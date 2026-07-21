# BFGS grouped-linear-algebra crossover

This decision-gated experiment measures the H92 (`D=285`) and H184 (`D=561`)
crossover between the previously selected H46 grouped path and H276 serial
path. H92 uses B256. H184 uses B128, the largest previously demonstrated
feasible resident size; B256 is a known OOM point for the corresponding
large-batch workload.

## Production result

Every point is one deterministic timing of the same cyclic R256 workload. All
eight runs converged 256/256 jobs. There is no timing uncertainty estimate.

| Model | Workload | Backend | Wall (s) | systems/s | Optimizer (s) | Model (s) | Neighbors (s) | Peak allocated (GB) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| AtomBit | H92 B256 | serial | 158.067 | 1.620 | 97.315 | 45.651 | 5.828 | 39.854 |
| AtomBit | H92 B256 | grouped | 137.528 | 1.861 | 77.877 | 45.552 | 5.431 | 39.854 |
| MACE | H92 B256 | serial | 159.183 | 1.608 | 103.670 | 40.530 | 5.231 | 36.465 |
| MACE | H92 B256 | grouped | 142.350 | 1.798 | 85.833 | 40.757 | 5.368 | 36.466 |
| AtomBit | H184 B128 | serial | 226.344 | 1.131 | 141.327 | 68.398 | 5.625 | 39.607 |
| AtomBit | H184 B128 | grouped | 212.942 | 1.202 | 127.768 | 68.292 | 5.665 | 39.604 |
| MACE | H184 B128 | serial | 228.687 | 1.119 | 151.992 | 58.883 | 6.489 | 36.712 |
| MACE | H184 B128 | grouped | 212.668 | 1.204 | 135.868 | 59.201 | 6.776 | 36.642 |

H92 passes both gates for both models: end-to-end throughput improves by 14.9%
for AtomBit and 11.8% for MACE, while optimizer time falls by 20.0% and 17.2%.
H184 passes the 5% end-to-end gate but does not meet the joint optimizer gate
because AtomBit reaches only 9.59%, below the declared 10% threshold. With one
timing this is a borderline result, so the conservative automatic decision is
serial. B128 was feasible but approached the practical allocator limit; B256
remains an inappropriate known-OOM point.

## Numerical comparison

| Model | Workload | Step mismatches | Max step difference | Max energy error (eV/atom) | Max position RMSD (A) | Max cell RMSD (A) |
|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | H92 B256 | 12 | 14 | 5.056e-4 | 0.0435 | 0.0324 |
| MACE | H92 B256 | 110 | 64 | 2.615e-3 | 0.2917 | 0.2624 |
| AtomBit | H184 B128 | 8 | 22 | 9.381e-4 | 0.0846 | 0.0481 |
| MACE | H184 B128 | 124 | 198 | 1.167e-2 | 0.7734 | 0.4052 |

Convergence flags match for all production jobs. A separate three-step H92
control isolates equation correctness before long-run trajectory amplification:

| Model | Energy (eV/atom) | fmax (eV/A) | Stress (eV/A^3) | Position (A) | Cell (A) |
|:--|--:|--:|--:|--:|--:|
| AtomBit | 0 | 0 | 0 | 0 | 5.55e-17 |
| MACE | 0 | 2.30e-13 | 1.09e-16 | 3.55e-15 | 8.24e-18 |

The fixed-step agreement confirms that the converged differences are
full-BFGS local-minimum sensitivity rather than a different update equation.

## Decision

The automatic grouped boundary advances from `D <= 256` to the directly
validated `D <= 285`, covering variable-cell H92. H184 (`D=561`) and H276
(`D=837`) remain serial. Explicit `linear_algebra_backend="grouped"` remains
available for diagnostics. The next conditional test is one heterogeneous
MIX-R256 confirmation of shape grouping and the updated automatic policy.

Raw production and fixed-step JSON/logs are retained on the benchmark server
under `runs/bfgs_grouped_crossover/`.
