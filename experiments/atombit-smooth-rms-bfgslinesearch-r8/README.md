# Smooth-RMS AtomBit BFGSLineSearch R8 validation

This experiment tests the native fp32 and fp64 smooth-RMS checkpoints on the
first eight frozen structures in each of H46, H92, H184, and H276. Each
structure is relaxed with variable-cell BFGSLineSearch (`fmax=0.05 eV/A`,
`max_steps=500`, `alpha=10`, `maxstep=0.2 A`) using common ASE, Active B1, and
Active B8. The matrix contains 32 unique structures and 192 optimization
attempts. Timings are one measured run on one NVIDIA H100 80 GB GPU.

## Results

All 192 attempts converged.

| dtype | workload | ASE (s) | Active B1 (s) | Active B8 (s) | B1/ASE | B8/ASE | ASE/B1/B8 peak GPU (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | H46 x8 | 93.032 | 65.343 | 17.744 | 1.424x | 5.243x | 0.176 / 0.176 / 0.668 |
| fp32 | H92 x8 | 127.581 | 97.278 | 22.860 | 1.312x | 5.581x | 0.277 / 0.279 / 1.364 |
| fp32 | H184 x8 | 259.395 | 160.725 | 47.974 | 1.614x | 5.407x | 0.546 / 0.547 / 2.667 |
| fp32 | H276 x8 | 435.370 | 126.192 | 36.332 | 3.450x | 11.983x | 0.595 / 0.602 / 3.223 |
| fp32 | all 32 | 915.379 | 449.539 | 124.910 | 2.036x | 7.328x | - |
| fp64 | H46 x8 | 96.275 | 65.784 | 18.791 | 1.464x | 5.123x | 0.280 / 0.281 / 1.260 |
| fp64 | H92 x8 | 151.882 | 87.698 | 22.372 | 1.732x | 6.789x | 0.484 / 0.483 / 2.655 |
| fp64 | H184 x8 | 431.518 | 169.291 | 58.398 | 2.549x | 7.389x | 1.015 / 1.020 / 5.251 |
| fp64 | H276 x8 | 423.870 | 129.443 | 43.990 | 3.275x | 9.636x | 1.121 / 1.129 / 6.315 |
| fp64 | all 32 | 1103.545 | 452.215 | 143.551 | 2.440x | 7.687x | - |

The aggregate fp32 throughputs are 0.03496, 0.07118, and 0.25618 systems/s for
ASE, B1, and B8. The corresponding fp64 values are 0.02900, 0.07076, and
0.22292 systems/s. Native fp32 is therefore suitable for this workload and is
faster and substantially smaller in memory than fp64.

## Correctness interpretation

A zero-step control compares energy, forces, stress, positions, and cell before
optimizer trajectory differences can accumulate. Every B1 and B8 parity case
passed. Across all sizes, the fp32 maxima were `4.20e-5 eV` in energy,
`3.81e-6 eV/A` in force, `2.79e-8 eV/A^3` in stress, `9.53e-7 A` in position,
and `1.49e-6 A` in cell. The fp64 maxima were `7.11e-14 eV`,
`7.11e-15 eV/A`, and `5.55e-17 eV/A^3`, with zero position and cell error.

All methods converge, but they do not always follow the same path into the same
local minimum. The maximum B1-versus-B8 endpoint energy differences for
H46/H92/H184/H276 are `0.00105/0.13625/0.02058/0.36572 eV` in fp32 and
`1.47e-6/0.19343/0.14478/0.03723 eV` in fp64. Exact initial parity rules out a
calculator mismatch. These endpoint differences reflect strong-Wolfe branch
sensitivity on a nonconvex, variable-cell landscape, so endpoint identity must
not be used as the implementation correctness criterion.

The machine-readable aggregate data and raw-output SHA-256 hashes are in
`results.json`. Raw outputs remain external because they contain full final
coordinates and are large.
