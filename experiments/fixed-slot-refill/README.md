# Fixed-slot in-flight refill

## Mechanism

TorchSim's
[`InFlightAutoBatcher`](https://torchsim.github.io/torch-sim/reference/torch_sim.autobatching.html)
removes converged states, admits pending states under a memory-scaling budget,
and restores original order. Its current implementation still calls `pop()`
followed by `concatenate_states()`.

This stage adopts the in-flight scheduling semantics but adds fixed storage for
homogeneous or planner-bucketed BFGS jobs:

- resident atom, cell, BFGS, and Frechet slots keep a stable shape;
- only completed slots are overwritten by pending jobs;
- survivor optimizer state stays in place;
- only replaced neighbor slots are marked invalid;
- unequal-size replacements and unfillable tails fall back to repacking.

`refill_storage="slots"` is explicit and opt-in. The existing
`refill_storage="repack"` behavior remains the default.

## Direct results

One deterministic timing was recorded per point on one H100. All methods
returned and converged 256/256 signed jobs.

| Model | Workload | Drain (s) | Repack (s) | Slots (s) | vs drain | vs repack | MPS32 (s) | vs MPS32 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| AtomBit | H46 R256 | 80.953 | 71.175 | **66.880** | **1.210x** | **1.064x** | 72.854 | **1.089x** |
| MACE | H46 R256 | 85.988 | 80.889 | **78.098** | **1.101x** | 1.036x | 81.421 | 1.043x |
| AtomBit | H276 STEPVAR | 105.325 | 102.772 | **99.673** | **1.057x** | 1.031x | 153.674 | **1.542x** |
| MACE | H276 STEPVAR | **168.035** | 170.591 | 172.025 | 0.977x | 0.992x | 171.581 | 0.997x |

The 5% gate selects slots for AtomBit H46 relative to drain, repack, and MPS32.
It selects slots over drain for MACE H46 and AtomBit H276, but not over repack
for those cases. MACE H276 selects drain/MPS parity, not refill.

The crossover follows total graph work. AtomBit H46 slots reduce model launches
from 1,076 to 663 with nearly unchanged graph evaluations. MACE H46 launches
fall from 1,052 to 731 with exactly 37,408 graph evaluations. MACE H276 refill
instead raises graph evaluations from 23,199 to 25,158. Refill is beneficial
when saved launch and optimizer overhead exceeds replacement graph work, not
merely because step counts vary.

## Allocator result

AtomBit H276 slots initially reserved 78.223 GiB while allocating 26.253 GiB.
Launching the identical point with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` produced:

| Configuration | Wall (s) | Peak allocated (GiB) | Peak reserved (GiB) |
|:--|--:|--:|--:|
| Standard allocator | 99.673 | 26.253 | 78.223 |
| Expandable segments | **99.334** | 26.253 | **29.477** |

The excess reserve was allocator fragmentation from long variable-shape
execution, not live fixed-slot state. Expandable segments should be part of the
recommended refill launcher on supported CUDA/PyTorch builds.

## Numerical scope

Every slot/repack comparison has identical convergence flags. Slot order changes
floating-point graph ordering, which full BFGS can amplify:

- MACE H46 has identical steps and at most `3.4e-10 eV/atom` endpoint error.
- AtomBit H46 has nine step mismatches and at most `0.367 meV/atom`.
- H276 reaches different converged minima for some jobs, with maxima of
  `3.00 meV/atom` AtomBit and `5.76 meV/atom` MACE.

The H276 values are throughput and convergence evidence, not same-minimum
equivalence evidence. Application ranking/hit-rate metrics remain necessary
before selecting unordered slots for CSP production.

Machine-readable results are in `results.json` and `results.csv`. Raw outputs
remain under `runs/fixed_slot_refill/`.
