# Float64-model deterministic BFGS scaling

## Setup

- AtomBit checkpoint weights, model inference, autograd forces, stress, and graph
  state: float64.
- BFGS coordinates, Frechet log deformation, Hessians, and eigensolves: float64.
- Common ASE uses the same float64 model with its CPU float64 BFGS/Frechet path.
- Determinism: `torch.use_deterministic_algorithms(True)` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- One CPU thread per worker; seven H100 GPUs were visible.
- Same 32 unique manifest structures at 46, 92, 184, and 276 atoms as the
  mixed-precision experiment.
- Full anisotropic Frechet optimization, `fmax=0.05 eV/A`, 500-step cap,
  `alpha=70 eV/A^2`, `maxstep=0.2 A`, and 6.0 A cutoff.
- Three synchronized repeats per timing point.

## Results

End-to-end seconds for each 32-structure pool. Values are medians of three
complete deterministic relaxations.

| atoms | common ASE | B1 | B2 | B4 | B8 | B16 | B32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 131.59 | 128.54 | 89.47 | 61.67 | 41.97 | 29.65 | 22.69 |
| 92 | 152.70 | 142.44 | 90.35 | 67.25 | 48.42 | 38.69 | 33.07 |
| 184 | 222.87 | 140.12 | 103.89 | 84.61 | 62.83 | 54.38 | 47.19 |
| 276 | 240.52 | 83.15 | 69.30 | 60.63 | 45.69 | 44.68 | 42.30 |

| atoms | B1 vs ASE | B32 vs ASE | B32 float64/mixed time | B32 time/graph ratio | B32 memory |
|---:|---:|---:|---:|---:|---:|
| 46 | 1.02x | 5.80x | 1.111x | 1.089x | 4.31 GiB |
| 92 | 1.07x | 4.62x | 1.028x | 1.061x | 9.31 GiB |
| 184 | 1.59x | 4.72x | 1.041x | 1.102x | 17.99 GiB |
| 276 | 2.89x | 5.69x | 0.959x | 1.187x | 22.78 GiB |

`B32 float64/mixed time` compares this run with float32 model inference plus a
float64 optimizer. Values above one are slower. The 276-atom float64 wall time
is lower only because it used fewer graph evaluations. After normalizing by
graph evaluations, float64 is slower for every atom count: 8.9%, 6.1%, 10.2%,
and 18.7%, respectively.

Float64 B32 peak memory is 1.96-1.98 times the mixed-precision requirement.
The common ASE float64/mixed wall-time ratios are 1.025, 1.015, 1.005, and
0.792. The last ratio is also caused by a shorter convergence path rather than
a faster float64 model.

## Correctness

A two-forced-step B1 smoke comparison at 46 atoms gave `2.70e-11 eV` energy
difference, `3.43e-12 A` position RMSD, `4.41e-12 A` cell RMSD, and identical
step counts between common ASE and the native implementation.

All full-run convergence flags match common ASE. The 46-atom final states and
steps pass the long-run validation exactly. The 92-, 184-, and 276-atom final
states can occupy different nearby minima and have different step counts, as
already observed in the mixed-precision BFGS study. Those differences do not
invalidate the fixed-step equation check, but full-convergence wall times must
not be interpreted as isolated kernel throughput.

## Conclusion

Casting the float32-trained checkpoint to float64 does not add learned
information. On H100 it has a modest 6-19% normalized runtime cost for these
workloads, but almost doubles model/batch memory. Float32 model inference with
float64 BFGS/Frechet state remains the preferable production configuration.

## Artifacts

- `results.json`: merged performance, memory, and validation summary.
- `raw/scaling/`: eight complete production scaling artifacts.
- `raw/smoke/`: float64 ASE/native fixed-step smoke artifacts.
