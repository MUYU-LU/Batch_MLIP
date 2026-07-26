# Sparse CUDA cell-list neighbors

This experiment tests whether spatial binning can replace the quadratic
candidate tensor used by `cuda_dense` for large periodic structures. It does
not change model arithmetic, graph cutoff filtering, cache invalidation,
optimizer mathematics, or scheduling.

The candidate backend is exact rather than approximate: topology arithmetic is
float64, every potentially occupied neighboring bin is expanded, Cartesian
distance is checked against the requested cutoff, and output is restored to
the canonical Matscipy-compatible ordering.

Small molecular-crystal unit cells are an explicit negative control. When a
6 A cutoff leaves only one bin per lattice direction, a cell list has no
sparsity to exploit and should not replace the existing dense CUDA backend.
Replicated cells up to about 2,200 atoms establish the large-individual-system
crossover needed for optimization and MD policy.

## Correctness

All 14 neighbor-screen points match Matscipy exactly in directed edge order and
integer shifts. Unit tests cover randomized triclinic cells, unwrapped
coordinates, small cells with multiple images, heterogeneous selective
rebuilds, and partial-PBC rejection. Across six real-model production pairs,
`cuda_cell` and `cuda_dense` produce bitwise-identical energies, forces,
positions, cells, velocities, and NVE drift.

## Neighbor Results

Each point uses two warmups and three synchronized timings. Representative
median results are:

| cutoff (A) | system | B | bins | dense (ms) | cell (ms) | speedup | dense/cell peak (MB) |
|--:|:--|--:|:--|--:|--:|--:|:--|
| 6.0 | H46 | 32 | 1x2x1 | 10.294 | 8.569 | 1.20x | 47 / 244 |
| 6.0 | H276 | 16 | 3x3x3 | 8.518 | 6.978 | 1.22x | 171 / 341 |
| 6.0 | H1242 | 2 | 5x6x4 | 9.735 | 4.167 | 2.34x | 369 / 198 |
| 6.0 | H2208 | 2 | 7x7x7 | 24.492 | 4.484 | 5.46x | 569 / 238 |
| 4.5 | H276 | 16 | 5x5x5 | 8.417 | 5.803 | 1.45x | 170 / 133 |
| 4.5 | H1242 | 2 | 7x9x6 | 9.706 | 3.687 | 2.63x | 367 / 92 |
| 4.5 | H2208 | 2 | 10x10x10 | 24.468 | 3.883 | 6.30x | 564 / 133 |

Small 6 A cells are a negative regime: cell construction can be modestly
faster but uses more temporary memory because too few bins remain. Large
replicated cells pass both the 1.30x speed and 30% memory-reduction gates.

## Production Results

| task | workload | AtomBit speedup | MACE speedup | cell neighbor time reduction |
|:--|:--|--:|--:|--:|
| EVAL | H2208 B2 | 1.608x | 1.765x | 82.7-86.0% |
| 3-step variable-cell BFGS | H368 B4 | 1.076x | 1.075x | 42.5-47.0% |
| 10-step skin-zero NVE | H1242 B1 | 1.076x | 1.128x | 40.5-43.0% |

AtomBit and MACE NVE retain identical trajectories and energy drift:
`3.30e-5` and `6.26e-5 eV/atom`, respectively.

## Policy

`neighbor_backend="cuda_cell"` is available explicitly for full-rank
3D-periodic cells. `auto` retains the previous Matscipy/dense threshold and
promotes a dense-eligible rebuild to `cuda_cell` only when a uniform-occupancy
cell estimate, using the current occupied fractional span, predicts at least
98% candidate reduction. At 6 A this keeps H46 B32 and H276 B16 on
`cuda_dense` while selecting `cuda_cell` for H2208 B2.

The backend falls back through the existing policy for partial/nonperiodic
cells and singular periodic cells. The estimate is intentionally conservative
and calibrated on one H100; other GPU architectures require recalibration.
