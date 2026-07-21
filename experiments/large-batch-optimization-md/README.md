# Large-batch optimization and MD

## Scope

This experiment corrects the one-shot EVAL capacity screen by measuring the
persistent-state workloads that can benefit from larger resident batches:

- R256 variable-cell full BFGS with active compaction and immediate refill.
- R256 NVE with 100 warmup and 1000 measured Velocity-Verlet steps.
- AtomBit and MACE-OFF-Small on homogeneous H46 and H276 structures.
- Current tensor-state calculators with `neighbor_backend=auto`.

Each R256 workload is an exact cyclic repetition of the frozen R32 pool. The
reported ASE speedup uses the measured R32 sequential ASE rate multiplied by
eight and is therefore a derived identical-work estimate, not a second R256 ASE
run. Every point has one timing observation.

## Selected policy

The policy selects the smallest point within 2% of maximum throughput while
keeping measured memory below 85% of the H100 capacity.

| Task | Model | Workload | Selected B | Throughput | ASE speedup | Memory | Decision |
|:--|:--|:--|--:|--:|--:|--:|:--|
| BFGS | AtomBit | H46 | 256 | 3.091 systems/s | 12.53x | 18.51 GB allocated | still scales to full R256 |
| BFGS | AtomBit | H276 | 64 | 0.966 systems/s | 9.11x | 27.31 GB allocated | B128 is slower |
| BFGS | MACE | H46 | 256 | 2.648 systems/s | 14.32x | 18.11 GB allocated | still scales to full R256 |
| BFGS | MACE | H276 | 64 | 0.918 systems/s | 10.54x | 27.47 GB allocated | B128 gain is only 1.1% |
| NVE | AtomBit | H46 | 128 | 919 replica-steps/s | 13.17x | 22.95 GB reserved | B256 is equal but unsafe |
| NVE | AtomBit | H276 | 16 | 243 replica-steps/s | 3.98x | 45.36 GB reserved | no larger safe point |
| NVE | MACE | H46 | 256 | 1001 replica-steps/s | 18.24x | 25.72 GB reserved | B256 remains safe |
| NVE | MACE | H276 | 64 | 297 replica-steps/s | 6.37x | 38.22 GB reserved | B128 violates memory gate |

The AtomBit H276 NVE B16 selection comes from the current direct frontier. B64
and B128 both execute in this experiment, but reserve more than 98% of GPU
memory and are not production choices.

## BFGS scaling

Memory is peak PyTorch allocated memory; the older optimization runner did not
record peak reserved memory.

| Model | Workload | B | systems/s | vs current B32 | ASE speedup | Peak GB |
|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | H46 | 32 | 1.716 | 1.00x | 6.96x | 2.37 |
| AtomBit | H46 | 64 | 2.579 | 1.50x | 10.46x | 4.88 |
| AtomBit | H46 | 128 | 2.789 | 1.63x | 11.31x | 9.71 |
| AtomBit | H46 | 256 | 3.091 | 1.80x | 12.53x | 18.51 |
| AtomBit | H276 | 32 | 0.926 | 1.00x | 8.73x | 12.36 |
| AtomBit | H276 | 64 | 0.966 | 1.04x | 9.11x | 27.31 |
| AtomBit | H276 | 128 | 0.949 | 1.03x | 8.95x | 54.11 |
| MACE | H46 | 32 | 1.629 | 1.00x | 8.81x | 2.33 |
| MACE | H46 | 64 | 2.360 | 1.45x | 12.76x | 4.62 |
| MACE | H46 | 128 | 2.486 | 1.53x | 13.44x | 9.17 |
| MACE | H46 | 256 | 2.648 | 1.63x | 14.32x | 18.11 |
| MACE | H276 | 32 | 0.777 | 1.00x | 8.92x | 13.64 |
| MACE | H276 | 64 | 0.918 | 1.18x | 10.54x | 27.47 |
| MACE | H276 | 128 | 0.928 | 1.19x | 10.66x | 54.68 |

H46 gains from a larger pool because refill sustains useful resident work after
individual jobs converge. H276 saturates at B64: model work and independent
dense full-BFGS eigensolves dominate, so doubling memory gives no useful gain.

## NVE scaling

Memory is peak reserved memory, which captures the long-trajectory allocator
and graph-cache behavior that EVAL misses.

| Model | Workload | B | replica-steps/s | vs current B32 | ASE speedup | Reserved GB |
|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | H46 | 32 | 741 | 1.00x | 10.62x | 4.24 |
| AtomBit | H46 | 64 | 841 | 1.14x | 12.06x | 11.81 |
| AtomBit | H46 | 128 | 919 | 1.24x | 13.17x | 22.95 |
| AtomBit | H46 | 256 | 919 | 1.24x | 13.17x | 83.36 |
| AtomBit | H276 | 32 | 263 | 1.00x | 4.31x | 72.69 |
| AtomBit | H276 | 64 | 274 | 1.04x | 4.49x | 83.72 |
| AtomBit | H276 | 128 | 283 | 1.07x | 4.63x | 83.51 |
| MACE | H46 | 32 | 821 | 1.00x | 14.96x | 3.24 |
| MACE | H46 | 64 | 932 | 1.14x | 16.98x | 5.16 |
| MACE | H46 | 128 | 974 | 1.19x | 17.75x | 11.65 |
| MACE | H46 | 256 | 1001 | 1.22x | 18.24x | 25.72 |
| MACE | H276 | 32 | 284 | 1.00x | 6.09x | 18.89 |
| MACE | H276 | 64 | 297 | 1.05x | 6.37x | 38.22 |
| MACE | H276 | 128 | 308 | 1.09x | 6.62x | 75.42 |

## Comparison with older large-batch runs

At matching resident sizes, the current implementation changes wall time as
follows:

| Task and point | Current improvement |
|:--|--:|
| AtomBit H46 BFGS B128 | 18.4% faster |
| AtomBit H276 BFGS B64 | 22.2% faster |
| MACE H46 BFGS B128 | 21.4% faster |
| MACE H276 BFGS B64 | 12.6% faster |
| AtomBit H46 NVE B128 | 9.6% faster |
| AtomBit H276 NVE B128 | 13.4% faster |
| MACE H46 NVE B128 | 0.6% slower, inconclusive |
| MACE H276 NVE B128 | 7.8% faster |

The BFGS gain includes the final adaptive neighbors and MACE cached tensor-state
path. NVE reuses candidate graphs for most steps, so neighbor-backend changes
have less leverage.

## Correctness limits

Every BFGS point converges 256/256 jobs. MACE H46 endpoints are effectively
batch invariant, with maximum B64/B256 differences of `2.67e-9 eV/atom` and
`8.52e-6 A` position RMSD. Full BFGS remains trajectory-sensitive for H276:
although convergence flags match, B64/B128 differences reach `16.6 meV/atom`
and `0.958 A` for AtomBit, and `9.89 meV/atom` and `0.584 A` for MACE. These are
different converged minima, so H276 timings are throughput measurements rather
than same-minimum equivalence claims.

NVE maximum energy-drift values remain stable across batch sizes. The spread is
at most `3.64e-6 eV/atom` for AtomBit and `6.33e-13 eV/atom` for MACE. Existing
short-horizon ASE parity and R256 endpoint validation remain the trajectory
equivalence evidence; this capacity runner records final energy diagnostics but
not complete native endpoint structures.

Complete metrics are in `results.csv`; cross-batch BFGS and NVE diagnostics are
in `diagnostics.json`. Raw JSON and logs, including failed launcher preflights
and the superseded MACE rebuild control, remain on the benchmark server under
`runs/large_batch_optimization_md/`.
