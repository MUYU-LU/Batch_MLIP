# Grouped BFGS linear algebra

This experiment compares independent serial and grouped CUDA full-BFGS Hessian
updates/eigensolves on the selected H46 B256 and H276 B64 production points for
AtomBit and MACE-OFF-Small.

## Result

Every point is one deterministic timing of the same cyclic R256 workload. All
eight runs converged 256/256 jobs. There is no timing uncertainty estimate.

| Model | Workload | Backend | Wall (s) | systems/s | Optimizer (s) | Model (s) | Neighbors (s) | Peak allocated (GB) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| AtomBit | H46 B256 | serial | 86.289 | 2.967 | 50.774 | 20.121 | 6.760 | 18.506 |
| AtomBit | H46 B256 | grouped | 68.492 | 3.738 | 33.887 | 20.261 | 6.509 | 18.505 |
| MACE | H46 B256 | serial | 94.999 | 2.695 | 59.645 | 21.868 | 4.641 | 18.106 |
| MACE | H46 B256 | grouped | 76.021 | 3.367 | 40.976 | 21.757 | 4.568 | 18.106 |
| AtomBit | H276 B64 | serial | 269.243 | 0.951 | 172.676 | 76.274 | 7.267 | 27.315 |
| AtomBit | H276 B64 | grouped | 256.390 | 0.998 | 160.185 | 76.256 | 7.274 | 27.312 |
| MACE | H276 B64 | serial | 277.600 | 0.922 | 187.462 | 71.376 | 7.455 | 27.474 |
| MACE | H276 B64 | grouped | 264.135 | 0.969 | 174.587 | 70.931 | 7.636 | 27.449 |

H46 passes the 10% optimizer-time gate. Grouping reduces optimizer time by
33.3% for AtomBit and 31.3% for MACE, producing 1.260x and 1.250x end-to-end
throughput. H276 fails the gate: optimizer time falls by only 7.2% and 6.9%,
and end-to-end throughput improves by approximately 1.05x.

## Numerical comparison

| Model | Workload | Step mismatches | Max step difference | Max energy error (eV/atom) | Max position RMSD (A) | Max cell RMSD (A) |
|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | H46 B256 | 15 | 31 | 1.370e-3 | 0.1306 | 0.0974 |
| MACE | H46 B256 | 0 | 0 | 1.482e-9 | 4.061e-6 | 4.147e-6 |
| AtomBit | H276 B64 | 13 | 45 | 1.355e-3 | 0.2117 | 0.0289 |
| MACE | H276 B64 | 88 | 154 | 9.893e-3 | 0.5848 | 0.1860 |

Convergence flags match for every job. MACE H46 is effectively invariant.
AtomBit H46 differences remain within the previously measured deterministic
cross-batch sensitivity. H276 again demonstrates full-BFGS local-minimum
sensitivity and does not justify enabling the grouped path automatically.

## Decision

`linear_algebra_backend="auto"` uses grouped CUDA linear algebra only for
equal-sized groups with generalized dimension `D <= 256`; CPU, singleton, and
larger-Hessian groups use the serial ASE-compatible path. `"serial"` and
`"grouped"` remain explicit controls. The boundary deliberately covers only
the validated small-Hessian regime; H92/H184 interpolation is the next
calibration experiment.

The isolated eigensolver microbenchmark showed only 1.03x at H46 and 1.00x at
H276. The integrated H46 gain comes primarily from vectorizing Hessian updates
and removing per-system Python/CUDA synchronization, not from a faster
cuSOLVER eigendecomposition alone. Raw JSON and logs are retained on the
benchmark server under `runs/bfgs_grouped_linear_algebra/raw/`.

The benchmark runner records peak allocated memory, not peak reserved memory;
reserved memory is therefore not claimed for this experiment. The largest
peak allocated value is 27.474 GB, below the 85% H100 gate.
