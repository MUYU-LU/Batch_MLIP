# BFGS versus FIRE variable-cell scaling

> **Correctness update (2026-07-18):** the single-run BFGS final-state
> comparison below is superseded by `experiments/bfgs-b1-audit`. Repeated
> common ASE BFGS itself selects different minima because nondeterministic GPU
> reductions are amplified by full BFGS. Timings remain valid measurements;
> final-coordinate differences are not evidence of a large force error.

## Hypothesis

Active-compacted full BFGS should outperform sequential ASE BFGS, but its
dense per-system eigensolves should scale less effectively than batched FIRE.

## Setup

- Production AtomBit checkpoint, float32, uncompiled, on NVIDIA H100 80 GB GPUs.
- The same 32 unique fixed-manifest T2 structures at 46, 92, 184, and 276 atoms.
- Full anisotropic Frechet/log-deformation cell optimization at zero pressure.
- `fmax=0.05 eV/A`, 500-step cap, `maxstep=0.2 A`, and 6.0 A cutoff.
- BFGS uses `alpha=70 eV/A^2`; FIRE uses `dt=0.1` and `dtmax=1.0`.
- One ASE timing and three synchronized active-batch timings per point.
- One BLAS/Torch CPU thread per GPU worker. An uncontrolled ASE-BFGS pilot
  selected over 100 BLAS threads and was retained as a negative control.

## Timings

End-to-end time in seconds for each 32-structure pool. Active values are
medians of three runs; ASE is a sequential single run.

| atoms | optimizer | ASE | B1 | B2 | B4 | B8 | B16 | B32 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 46 | FIRE | 117.02 | 95.80 | 68.68 | 53.10 | 40.47 | 26.40 | 17.24 |
| 46 | BFGS | 97.38 | 88.59 | 67.86 | 51.19 | 37.35 | 29.97 | 24.58 |
| 92 | FIRE | 113.62 | 96.58 | 69.37 | 53.92 | 40.34 | 29.04 | 20.68 |
| 92 | BFGS | 127.15 | 110.99 | 83.97 | 63.13 | 54.29 | 46.74 | 44.89 |
| 184 | FIRE | 94.63 | 79.97 | 62.99 | 50.79 | 37.14 | 26.73 | 24.08 |
| 184 | BFGS | 193.64 | 94.95 | 75.97 | 60.36 | 48.42 | 42.65 | 39.10 |
| 276 | FIRE | 56.18 | 48.90 | 33.07 | 24.83 | 19.27 | 16.30 | 15.06 |
| 276 | BFGS | 280.70 | 85.79 | 65.68 | 56.14 | 45.28 | 45.07 | 39.23 |

Direct B32 comparison:

| atoms | BFGS speedup vs ASE BFGS | FIRE speedup vs ASE FIRE | FIRE speed advantage vs BFGS | BFGS peak memory |
|---:|---:|---:|---:|---:|
| 46 | 3.96x | 6.79x | 1.43x | 2.20 GiB |
| 92 | 2.83x | 5.49x | 2.17x | 4.69 GiB |
| 184 | 4.95x | 3.93x | 1.62x | 9.04 GiB |
| 276 | 7.16x | 3.73x | 2.61x | 11.43 GiB |

B32 BFGS timing ranges were 24.44-24.80, 43.99-45.79, 38.40-39.72,
and 39.22-41.19 seconds in increasing atom-count order. Active compaction
avoided 4,623, 4,324, 4,520, and 2,262 graph evaluations at B32.

## Correctness

Every common ASE and batched BFGS run converged 32/32 structures. However,
batched BFGS fails the existing final-state gate at every batch size. Across
all points, worst differences versus common ASE BFGS are 16.59 meV/atom in
energy, 0.888 A position RMSD, 0.653 A cell RMSD, and 158 optimizer steps.
The mismatch is already present at B1, where the worst energy difference is
9.17 meV/atom and position RMSD is 0.828 A, so active compaction alone is not
the cause. Exact float64 algorithm tests still match ASE; the production
failure is consistent with cumulative float32 full-BFGS trajectory sensitivity.

FIRE matches ASE convergence flags in every B32 group. Its strict final-state
gate passes at 184 atoms and fails the other groups primarily on the known
float32 energy/trajectory tolerance; its worst B32 energy error is 0.832
meV/atom. Therefore the BFGS wall-time ratios above are measurements, not a
scientifically validated replacement for common ASE BFGS on long relaxations.

Verification:

```text
python -m pytest -q
38 passed in 32.93s

python -m ruff check atombit_batch tests benchmarks
All checks passed!
```

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/benchmark_variable_cell_scaling.py \
  --optimizer OPTIMIZER --method METHOD --atom-count ATOMS \
  --pool-size 32 --batch-sizes 1,2,4,8,16,32 --repeats REPEATS \
  --device cuda:0 --output OUTPUT

python benchmarks/summarize_bfgs_fire_scaling.py \
  --raw-dir runs/bfgs_fire_scaling_raw \
  --output runs/bfgs_fire_scaling_summary.json
```

`raw/` contains all 16 benchmark artifacts plus both pilots. `results.json`
contains the merged timing and per-point validation records.

## Conclusion

Full BFGS benefits from graph batching versus sequential ASE BFGS, but FIRE is
the better current production optimizer: it is faster at every B32 size, uses
linear optimizer state rather than dense Hessians, and has materially smaller
final-state deviations. The next BFGS experiment should use float64 optimizer
state and force accumulation while keeping model evaluation float32, then test
fixed-step equivalence before repeating converged scaling. If memory and speed
remain limiting, implement LBFGS rather than optimize the dense eigensolve.
