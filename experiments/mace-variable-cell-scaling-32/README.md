# MACE variable-cell scaling to batch 32

## Setup

- MACE-OFF23-Small in float64 on NVIDIA H100 80 GB GPUs.
- The same 32 unique T2 structures used by the AtomBit variable-cell test for
  each 46, 92, 184, and 276 atom group.
- Common sequential MACE ASE, masked native batching, and active-compacted
  native batching.
- Full Frechet variable-cell FIRE, `fmax=0.05 eV/A`, 500-step cap, zero
  pressure, and the model's 4.5 A cutoff.
- Three synchronized batch repeats and one synchronized ASE reference run.

## Timings

End-to-end seconds for each 32-structure pool; batch values are medians.

| atoms | method | B1 | B2 | B4 | B8 | B16 | B32 |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 46 | masked | 115.75 | 91.64 | 76.22 | 61.87 | 37.38 | 26.83 |
| 46 | active | 119.42 | 88.51 | 70.52 | 52.78 | 29.22 | 17.07 |
| 92 | masked | 100.30 | 79.57 | 64.32 | 51.76 | 38.26 | 41.44 |
| 92 | active | 92.03 | 71.11 | 53.38 | 38.32 | 23.57 | 16.59 |
| 184 | masked | 106.44 | 84.33 | 70.00 | 68.62 | 68.44 | 63.53 |
| 184 | active | 97.08 | 72.78 | 57.19 | 45.96 | 32.88 | 22.82 |
| 276 | masked | 59.50 | 41.99 | 29.02 | 25.45 | 23.72 | 25.68 |
| 276 | active | 57.58 | 38.17 | 25.38 | 18.94 | 15.39 | 13.78 |

Direct B32 comparison:

| atoms | ASE s | masked s | active s | masked/ASE | active/ASE | active/masked |
|---:|---:|---:|---:|---:|---:|---:|
| 46 | 144.06 | 26.83 | 17.07 | 5.37x | 8.44x | 1.57x |
| 92 | 120.53 | 41.44 | 16.59 | 2.91x | 7.27x | 2.50x |
| 184 | 114.77 | 63.53 | 22.82 | 1.81x | 5.03x | 2.78x |
| 276 | 69.69 | 25.68 | 13.78 | 2.71x | 5.06x | 1.86x |

Masked batching saturates before B32 for 92 and 276 atoms. Active compaction
continues to improve through B32 because it removes converged structures from
subsequent MACE graph construction and inference.

## Correctness

Every masked and active B1-B32 point passes the existing AtomBit final-state
gates against common MACE ASE. At B32, convergence counts match exactly at
27/32, 30/32, 30/32, and 32/32 in increasing atom-count order, and the maximum
step difference is zero for every group.

The worst B32 active discrepancies are `1.01e-8 eV/atom` in energy,
`6.72e-7 eV/A` in final maximum force, `1.25e-9 eV/A^3` in stress,
`5.76e-7 A` position RMSD, and `2.49e-7 A` cell RMSD. These are well below the
unchanged production tolerances.

At B32, active compaction avoids 68.1%, 74.1%, 74.6%, and 51.2% of masked graph
evaluations. Peak allocated GPU memory is 2.17, 4.29, 8.50, and 12.54 GiB.

## Scope

This is the same structure pool and optimization protocol as the AtomBit
variable-cell FIRE test, but it is not a direct model-speed comparison because
MACE uses float64 and a 4.5 A model cutoff while the historical AtomBit run used
float32 and 6.0 A. The reported speedups compare each MACE batched method only
against common MACE ASE.
