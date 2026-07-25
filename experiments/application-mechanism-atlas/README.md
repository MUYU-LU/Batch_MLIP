# Application mechanism atlas

## Purpose

This stage asks a narrower question than another batch-size sweep: which changes
from independent ASE jobs to the tensor engine accelerate which task shapes,
why do they help, and where do they stop helping?

CUDA MPS with 32 independent ASE workers is the only new reference. Sequential
ASE is not rerun. Each point is one deterministic timing on one H100 using the
same signed jobs and checkpoint. A difference must exceed 5% to be called
practically meaningful. These are controlled engineering results without timing
uncertainty, not final paper estimates.

## Direct MPS comparison

| Task | Model | Tensor policy | Tensor (s) | MPS32 (s) | Speedup vs MPS32 | Tensor reserved / MPS device peak | Decision |
|:--|:--|:--|--:|--:|--:|--:|:--|
| EVAL-MIX-R256 | AtomBit | B128 | 0.944 | 1.225 | **1.30x** | 0.66 | tensor wins |
| EVAL-MIX-R256 | MACE | B128 | 1.790 | 3.353 | **1.87x** | 0.73 | tensor wins |
| NVE-MIX-R32, 1000 steps | AtomBit | B32, skin 0.5 A | 89.217 | 114.012 | **1.28x** | 0.34 | tensor wins |
| NVE-MIX-R32, 1000 steps | MACE | B32, skin 0.5 A | 83.102 | 90.678 | **1.09x** | 0.25 | tensor wins |
| STEPVAR-R256 BFGS | AtomBit | active drain B64 | 108.902 | 153.674 | **1.41x** | 0.68 | tensor wins |
| STEPVAR-R256 BFGS | MACE | active drain B64 | 175.635 | 171.581 | 0.98x | 0.60 | parity |

Tensor memory is PyTorch peak reserved memory; MPS memory is the total
device-level `nvidia-smi` peak of 32 processes. The ratio is useful for
operational comparison but the allocator scopes are not identical.

All 256 evaluation outputs were finite, all 32 NVE trajectories passed the
existing short-horizon validation and finite long-horizon checks, and every
optimization method converged 256/256 jobs.

`MIX` is an equal H46/H276 mixture: EVAL contains 128 of each size and NVE
contains 16 of each. Tensor/MPS atom throughput is 43,663/33,642 atoms/s for
AtomBit EVAL and 23,032/12,293 atoms/s for MACE EVAL. NVE reaches
57,747/45,188 and 61,996/56,816 atom-steps/s, respectively. Tensor phase
telemetry records model/neighbor time of 0.466/0.381 s and 1.321/0.371 s for
AtomBit/MACE EVAL, and 59.879/7.726 s and 54.053/6.359 s for NVE. Equivalent
component timings are not inferred for MPS.

## What changed from ASE

| Change | Circumstance where it helps | Measured explanation | Boundary |
|:--|:--|:--|:--|
| Pack graphs and evaluate one persistent model | One-shot pools large enough to form efficient batches | B128 reduces independent model replicas, Python dispatch, and fragmented small GPU launches; direct gain is 1.30x AtomBit and 1.87x MACE over MPS32 | Small pools and memory-limited large graphs cannot fill the batch |
| Keep positions, velocities, cells, and model graph state as tensors | Fixed-horizon MD replicas | B32 shares model launches and integration while retaining state on device; direct gain is 1.28x AtomBit and 1.09x MACE with roughly one-quarter to one-third of MPS memory | Current validation is NVE/NVT engineering scope; NPT is not implemented |
| Cache skin-expanded candidate neighbors | Repeated steps with modest displacement | Earlier ablations saved 3-14% in optimization; NVE reuses more than 90% of candidate graphs, making the neighbor backend only a 0-7% effect there | Cell/position motion beyond the validity bound rebuilds; larger skins inflate edges |
| Compact converged jobs out of the resident batch | Variable-horizon optimization with a long convergence tail | Avoids force/model work on inactive structures; current active-drain AtomBit is 1.41x faster than MPS32 | MACE STEPVAR is at MPS parity, so compaction does not overcome its adapter/model cost in every workload |
| Refill freed slots | Only when saved underfilled model calls exceed admission and state-rebuild cost | It did not do so here: AtomBit used 44% fewer model calls but was 3.5% slower and reserved 1.83x more memory; MACE used 51% fewer calls but was 2.9% slower and reserved 1.12x more | STEPVAR refill is practical parity but numerically slower; do not enable it by default |
| Preserve survivor topology during compaction | AtomBit refilling/compaction with cached neighbors | Prior ablation reduced neighbor search from 70.1 to 10.5 s and improved throughput 1.19x over rebuilding the resident graph | MACE has a separate `AtomicData` state and needs an equivalent native implementation |
| Use adaptive CUDA-dense neighbors instead of only matscipy | Rebuild-heavy EVAL and variable-cell AtomBit work | Prior direct ablation gained 1.69-5.02x for EVAL and 1.21-1.38x for AtomBit BFGS over matscipy batching | Dense search is quadratic and adds only 0-7% in cache-dominated NVE |
| Group equal-size full-BFGS linear algebra on GPU | Homogeneous groups with generalized dimension `D <= 256` | Prior H46 tests reduced optimizer time by 31-33% and total time by 1.25-1.26x by removing per-job Python/synchronization | H276 total gain was only about 1.05x; auto falls back above the validated dimension |
| Bucket and memory-plan batches | Mixed-size pools or near-memory-limit operation | Prevents OOM; held-out peak-memory error was 1.30% AtomBit and 0.36% MACE | Earlier simple mixed bucketing gains of 4.0-4.8% were below the practical gate |
| Shard closed pools across GPUs | Hundreds of costly jobs with enough work per shard | Prior R256 H276 results reached 5.21x AtomBit and 5.45x MACE on seven GPUs | Cold R32 pools gained only 1.06x and 1.34x because startup dominates |

The current direct measurements isolate complete execution policies, not every
component. Component claims in the table come only from prior paired ablations;
independent speedups are not multiplied.

## Next acceleration work

1. **Fixed-slot refill ablation.** Keep preallocated tensor/graph slots and
   overwrite only admitted jobs, instead of repacking a larger resident state.
   Compare active drain, current refill, and fixed-slot refill against MPS32 on
   STEPVAR with low, medium, and high step-count variance. This directly targets
   the measured refill overhead and 78.2 GiB AtomBit reserve peak.
2. **GPU cell-list neighbor construction.** Replace quadratic CUDA-dense
   candidate generation with GPU spatial bins while preserving matscipy edge
   and periodic-shift semantics. Test H46/H92/H184/H276 plus larger individual
   systems across EVAL, cache-miss NVE steps, and variable-cell optimization.
3. **Shared-topology finite-displacement path.** Build one neighbor template
   per parent crystal and update only geometry for its displacement replicas.
   This is a high-confidence application-specific opportunity that current MIX
   workloads do not measure.
4. **Persistent service and joint policy.** Reuse loaded models and worker
   processes across arriving pools, then select batch/edge limits, skin,
   drain/refill, buckets, and GPU count from telemetry. Evaluate regret against
   an offline measured oracle, using MPS32 as the independent-job alternative.
5. **Graph-break and kernel audit.** Measure `torch.compile`, fused
   scatter/radial operations, and reduced MACE graph conversion independently.
   Retain only changes that improve end-to-end time by more than 5%.

LBFGS is not listed as an acceleration of BFGS because it changes the optimizer
and potentially the relaxation path. It remains a separate algorithmic option
for large Hessians, not a fair same-method performance claim.

Machine-readable normalized results are in `results.json` and `results.csv`.
Raw outputs remain under the ignored
`runs/application_mechanism_atlas/current/` directory.
