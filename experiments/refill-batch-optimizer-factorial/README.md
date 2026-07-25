# Refill, resident batch, and optimizer factorial

## Question

The earlier refill comparison changed atom count while holding the resident
batch at B64. That could not distinguish an atom/edge effect from a resident
batch effect. This stage crosses both variables for the same signed 256-job
pools and uses one deterministic timing per point.

Stage 1 covers BFGS, AtomBit smooth-RMS float32, and MACE-OFF-Small float64 with
variable-cell Frechet optimization:

- atom counts: H46 and H276;
- resident batches: B32, B64, and B128;
- schedulers: active drain, refill with repacking, and refill with fixed slots;
- reference: 32 independent ASE BFGS workers under CUDA MPS.

## BFGS refill result

All entries converged 256/256 jobs.

| Model | Atoms | Batch | Drain (s) | Repack (s) | Slots (s) | Slots vs drain | Slot memory alloc/reserve (GiB) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| AtomBit | 46 | 32 | 108.46 | 84.81 | **81.66** | **1.328x** | 2.3 / 2.6 |
| AtomBit | 46 | 64 | 79.27 | 69.22 | **66.61** | **1.190x** | 4.5 / 5.1 |
| AtomBit | 46 | 128 | **60.55** | 60.98 | 62.85 | 0.963x | 9.0 / 10.1 |
| AtomBit | 276 | 32 | 127.17 | 103.29 | **100.92** | **1.260x** | 12.3 / 13.8 |
| AtomBit | 276 | 64 | 102.94 | 97.59 | **94.38** | **1.091x** | 24.3 / 27.3 |
| AtomBit | 276 | 128 | 92.66 | 93.30 | **92.54** | 1.001x | 48.6 / 54.5 |
| MACE | 46 | 32 | 119.51 | 92.89 | **90.35** | **1.323x** | 2.2 / 2.5 |
| MACE | 46 | 64 | 86.30 | 80.10 | **78.65** | **1.097x** | 4.3 / 4.9 |
| MACE | 46 | 128 | 74.77 | **74.05** | 74.12 | 1.009x | 8.5 / 9.8 |
| MACE | 276 | 32 | 187.83 | 158.00 | **157.55** | **1.192x** | 12.8 / 14.6 |
| MACE | 276 | 64 | 154.01 | **147.97** | 148.81 | 1.035x | 25.4 / 29.2 |
| MACE | 276 | 128 | **143.84** | 144.92 | 145.32 | 0.990x | 50.6 / 57.6 |

Refill benefit decreases consistently as resident batch increases. At B32 it
is useful for both atom counts and both MLIPs. At B128 it is neutral or
harmful. Atom count changes the magnitude and shifts the intermediate B64
case, but it is not a valid refill switch by itself.

Fixed slots preserve launch counts relative to repacking and save at most
3.9% here. They are a secondary storage optimization, not the source of the
large drain-to-refill gain.

## Best mode versus MPS

| Model | Atoms | Best tensor mode | Tensor (s) | MPS32 (s) | Speedup | Tensor reserve / MPS device (GiB) |
|:--|--:|:--|--:|--:|--:|--:|
| AtomBit | 46 | active B128 | **60.55** | 72.85 | **1.203x** | 9.3 / 31.6 |
| AtomBit | 276 | slots B128 | **92.54** | 132.37 | **1.430x** | 54.5 / 52.7 |
| MACE | 46 | repack B128 | **74.05** | 81.42 | **1.100x** | 9.8 / 31.6 |
| MACE | 276 | active B128 | **143.84** | 152.26 | **1.059x** | 57.4 / 52.7 |

The fastest overall action is therefore:

1. Select the largest resident batch that remains below the memory gate and
   lies on the favorable measured model-throughput curve.
2. At that capacity, enable refill only when launch savings exceed the cost of
   keeping a large resident batch full and replacing its state.
3. Use fixed slots only for equal-size or tightly bucketed jobs; otherwise use
   repacking.
4. Fall back to MPS when tensor batching is unsupported or the selected tensor
   point does not beat the application-specific parity gate.

For this 256-job matrix, B32 selects refill, B128 selects drain, and B64 depends
on MLIP and graph work. This is a batch-size, graph-cost, model-scaling, and
optimizer policy, not an atom-count policy.

## Numerical scope

Every slot/repack pair has identical convergence flags. Reordering can change
floating-point graph accumulation and the BFGS trajectory: the maximum endpoint
energy difference is `3.017 meV/atom` for AtomBit and `6.158 meV/atom` for
MACE. These timings establish throughput and convergence, not same-minimum
equivalence for CSP ranking.

Machine-readable results are in `results/results.json` and
`results/results.csv`. Raw outputs remain under
`runs/refill_batch_optimizer_factorial/`.

## FIRE verification

FIRE uses independent velocity, `dt`, `alpha`, positive-power counter, Frechet
state, and local step count for each resident job. The new refill path preserves
survivor state and initializes those tensors only for newly admitted jobs.
Fixed- and variable-cell trajectory tests pass before timing.

The BFGS limit of 500 steps was not sufficient for FIRE H46. A diagnostic run
converged only 240/256 AtomBit and 216/256 MACE jobs, so those timings were
rejected. The final FIRE matrix uses a 2,000-step ceiling; every point converges
256/256, with observed maxima below 1,000 steps.

| Model | Atoms | Batch | Drain (s) | Slots (s) | Refill speedup | Step median / p90 / max | Slot reserve (GiB) |
|:--|--:|--:|--:|--:|--:|:--|--:|
| AtomBit | 46 | 32 | 282.63 | **116.07** | **2.435x** | 84.5 / 381 / 942 | 2.5 |
| AtomBit | 46 | 128 | 114.73 | **90.25** | **1.271x** | 84.5 / 371.5 / 923 | 9.3 |
| AtomBit | 276 | 32 | 92.90 | **87.40** | **1.063x** | 59.5 / 93 / 115 | 12.6 |
| AtomBit | 276 | 128 | **76.39** | 77.78 | 0.982x | 60 / 93 / 115 | 49.8 |
| MACE | 46 | 32 | 314.78 | **123.29** | **2.553x** | 71.5 / 549 / 986 | 2.5 |
| MACE | 46 | 128 | 121.68 | **97.81** | **1.244x** | 71.5 / 549 / 986 | 9.6 |
| MACE | 276 | 32 | 124.99 | **112.36** | **1.112x** | 62 / 104 / 148 | 14.3 |
| MACE | 276 | 128 | **106.07** | 106.31 | 0.998x | 62 / 104 / 148 | 56.6 |

This separates convergence spread from atom count. H46 has a wide FIRE tail
and benefits from refill even at B128. H276 has a narrow step distribution and
does not benefit at B128. Batch size alone is still insufficient: the policy
needs an optimizer-specific estimate of the job-duration distribution.

## FIRE versus MPS

| Model | Atoms | Best tensor mode | Tensor (s) | MPS32 (s) | Tensor speedup |
|:--|--:|:--|--:|--:|--:|
| AtomBit | 46 | slots B128 | **90.25** | 121.01 | **1.341x** |
| AtomBit | 276 | active B128 | **76.39** | 79.14 | 1.036x |
| MACE | 46 | slots B128 | **97.81** | 112.03 | **1.145x** |
| MACE | 276 | active B128 | 106.07 | **76.30** | 0.719x |

The 5% gate selects tensor refill for H46, MPS/tensor parity for AtomBit H276,
and MPS for MACE H276. Refill cannot fix an MLIP/graph regime where independent
MPS execution is intrinsically faster.

FIRE and BFGS also exchange rank by workload. The best BFGS configuration is
faster for H46, while the best FIRE configuration is faster for H276. Optimizer
selection must therefore precede the scheduler decision and use application
quality metrics in addition to throughput.

All FIRE slot/drain pairs have identical convergence flags. MACE endpoints and
steps are numerically identical to reported precision. AtomBit H276 is also
effectively identical; AtomBit H46 has at most `0.188 meV/atom` endpoint energy
difference from floating-point graph reordering.

## BFGSLineSearch verification

BFGSLineSearch, registered also as `QuasiNewton`, performs independent
strong-Wolfe searches. Tensor execution batches compatible trial rounds but
does not admit pending jobs, because each resident system owns an inverse
Hessian and an asynchronous line-search task. The comparison therefore uses
active B128 directly against MPS32.

| Model | Atoms | Tensor B128 (s) | MPS32 (s) | Tensor speedup | Tensor reserve (GiB) |
|:--|--:|--:|--:|--:|--:|
| AtomBit | 46 | **149.20** | 206.00 | **1.381x** | 9.3 |
| AtomBit | 276 | **184.43** | 238.33 | **1.292x** | 49.8 |
| MACE | 46 | **164.36** | 195.15 | **1.187x** | 9.7 |
| MACE | 276 | 262.71 | **212.38** | 0.808x | 57.3 |

Every point converges 256/256. Smooth-RMS AtomBit now supports the
energy-based line search on both workloads, but BFGSLineSearch is slower than
the best FIRE or BFGS mode here. Tensor trial batching beats MPS for three
cases; MACE H276 remains an MPS regime because the large tensor graphs scale
poorly enough to outweigh the reduction in independent model evaluations.

## Combined policy

The sequential optimizer experiments support this decision order:

1. Select the optimizer using application quality and a small signed pilot;
   optimizer step distributions differ enough to exchange the performance
   ranking.
2. Measure model/graph throughput versus resident batch and enforce the memory
   gate.
3. For FIRE or BFGS, predict convergence spread and compare drain versus refill
   at the selected capacity. Refill is strongest for a large pending pool,
   small capacity, and a wide duration distribution.
4. Use fixed slots only inside equal-size or tight atom/edge buckets.
5. Compare the selected tensor point with MPS. Use MPS when the MLIP/graph
   regime does not batch efficiently, as observed for MACE H276 FIRE and
   BFGSLineSearch.

No rule based only on atom count or only on batch size is supported by the
measurements.
