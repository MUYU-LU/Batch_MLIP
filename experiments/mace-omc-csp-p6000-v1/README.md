# MACE OMC-CSP P6000: automatic batching versus MPS16

## Question

Does the frozen AtomBit OMC-CSP execution architecture transfer to a different
MLIP, MACE-OFF23-Small, and remain faster than 16 ASE/CUDA-MPS processes per
GPU on the same strict variable-cell BFGS workload?

The benchmark uses all 6,000 unique T2 CIFs exactly once. Both paths use the
same MACE model file, eight H100 GPUs, float64 model and optimizer state,
`FrechetCellFilter`, `fmax=0.01 eV/A`, `max_steps=3000`, and no tail recovery.
Each path was timed once, as requested.

## Execution paths

Automatic tensor execution uses 31 offline-planned resident chunks containing
109-256 structures. The outer scheduler uses model-specific atom/edge/BFGS
profiles, bucket-stratified initial placement, and complete-chunk work
stealing. Each GPU uses active compaction and active drain; multi-GPU refill is
disabled. MACE `AtomicData` graphs are rebuilt on every evaluation because the
cached variable-cell path has not yet passed the production gate.

MPS16 uses 16 persistent ASE BFGS processes per GPU, 128 processes in total.
Complete structures are assigned by static cost-balanced LPT. A worker-scoped
warning filter suppresses only repeated SciPy `logm` accuracy warnings; it does
not change the numerical calculation.

## Results

| Method | Execution/makespan (s) | Full script (s) | Systems/s | Converged |
|:--|--:|--:|--:|--:|
| Automatic tensor active drain | 3,190.83 | 3,200.59 | 1.880 | 5,963/6,000 |
| ASE/CUDA-MPS, 16 workers/GPU | 4,563.54 | 4,652.83 | 1.315 | 5,960/6,000 |

The automatic MACE path is **1.430x faster** using execution/makespan and
**1.454x faster** using full-script time. The result is smaller than the
corresponding AtomBit gain because MACE graph construction and dense BFGS
linear algebra consume most of the time saved by batched model inference.

The tensor workers are reasonably balanced (`max/mean=1.053`). MPS16 has
similar per-GPU aggregate balance (`max/mean=1.058`), but static assignment
leaves a terminal tail: complete ASE jobs cannot move after their unknown
convergence durations become visible.

## Numerical gate

The three-step P64 control matches ASE/MPS16 to `2.91e-11 eV` in energy,
`1.18e-11 A` in positions, `1.07e-11 A` in cells, and
`1.09e-12 eV/A^3` in stress. This validates the model adapter, force/stress
mapping, cell filter, and early BFGS trajectory.

The 3,000-step production endpoints are not identical. The convergence totals
differ by only three systems, or 0.05 percentage points, but 73 paired flags
differ: tensor alone converges 38 and MPS16 alone converges 35. Only two fail
in both paths. Absolute endpoint energy differences have median 0.0059, p95
6.88, p99 15.75, and maximum 36.49 meV/atom.

Four difficult structures were rerun through tensor B1. By maximum coordinate
difference, two B1 endpoints are closer to the production tensor result and
two are closer to ASE/MPS16. This rules out a simple batch-size error and is
evidence of local-minimum bifurcation over long, floating-point-sensitive BFGS
trajectories. The timing result therefore measures the same optimization
contract; it does not claim identical minimum selection.

## Memory and hotspots

The MACE capacity model is rejected for production. It predicts at most
72.22 GB under a 72.27-GB (85%) budget, but sequential heterogeneous chunks
raise PyTorch's reserved high-water mark to 82.16 GB, 96.6% of the device.
Peak allocated memory is 59.57 GB. The discrepancy is allocator retention and
fragmentation across chunk shapes, not live tensor storage alone. MPS16 peaks
at 38.56 GB in device-level sampling, but that measurement is not directly
comparable with PyTorch reserved memory.

Summed non-overlapping worker phases are:

| Phase | Worker-time share |
|:--|--:|
| MACE model forward | 32.95% |
| BFGS update | 32.37% |
| MACE `AtomicData` construction | 25.27% |
| MACE collation | 3.47% |
| State-to-ASE conversion | 1.19% |
| Graph transfer to GPU | 0.80% |
| Active compaction | 0.37% |

## Decision

Accept the **1.430x MPS16 acceleration** as performance evidence for this exact
MACE contract. Do not freeze MACE OMC-CSP v1 yet. The next bounded work is to
add an allocator-high-water-aware capacity margin and validate a persistent
MACE tensor-graph path that removes repeated `AtomicData` construction without
changing variable-cell endpoints. The frozen AtomBit policy is unchanged.

Machine-readable metrics and raw-artifact hashes are in
`results/summary.json`. The raw production results remain on the H100 host at
`/public/home/lmy/Batch_imple_project/mace_omc_csp_p6000_v1`.
