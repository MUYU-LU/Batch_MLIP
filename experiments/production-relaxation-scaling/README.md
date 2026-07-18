# Production relaxation scaling

## Hypothesis

Batched FIRE will reduce fixed-pool relaxation wall time relative to sequential ASE FIRE while preserving convergence and final structures.

## Controls

- Production `model_epoch_15.pt` checkpoint, strict state-dict loading.
- Float32 autograd forces, fixed cells, 6.0 A cutoff, and zero neighbor skin.
- The same 16 hash-ranked structures at each exact atom count: 46, 92, 184, and 276.
- Every batch size processes all 16 structures, chunked at B1, B2, B4, B8, or B16.
- Matched FIRE parameters: `fmax=0.05 eV/A`, 500 maximum steps, `dt=0.1`, `dtmax=1.0`, `maxstep=0.2`, `Nmin=5`, `finc=1.1`, `fdec=0.5`, `alpha=0.1`, and `fa=0.99`.
- One sequential ASE timing and three batched timing trials with synchronized CUDA.

## Acceptance gates

- Identical convergence flags.
- Maximum residual-energy difference no greater than `5e-5 eV/atom`.
- Maximum final-force-maximum difference no greater than `2e-2 eV/A`; both paths must independently satisfy `fmax=0.05 eV/A`.
- Maximum final position RMSD no greater than `1e-2 A`.
- Optimizer convergence-step difference no greater than 15; identical FIRE trajectories are not required.
- OOM, nonconvergence, and validation failures remain in the result.

## Implementation correction

The initial production probe exposed an update-order mismatch in `batched_fire_relax`: the batched path decayed FIRE `alpha` before velocity mixing, while ASE mixes with the current `alpha` and then decays it for the next step. The order was corrected and a float64 ASE trajectory regression test was added. The corrected test agrees with ASE positions to `1e-12`.

## Results

All 64 ASE reference structures and all batched structures converged below `0.05 eV/A` within 500 steps. End-to-end relaxation speedup over sequential ASE FIRE was:

| atoms | B1 | B2 | B4 | B8 | B16 |
|---:|---:|---:|---:|---:|---:|
| 46 | 0.92x | 1.26x | 1.58x | 2.38x | 2.64x |
| 92 | 0.91x | 1.10x | 1.22x | 1.19x | 1.36x |
| 184 | 0.93x | 1.27x | 1.53x | **1.56x** | 1.52x |
| 276 | 0.93x | 1.31x | 1.44x | **1.45x** | 1.44x |

The original matrix used the stricter preliminary `2e-5 eV/atom` and 10-step gates. Nineteen of 20 points passed; 92-atom B8 reached the same convergence condition with `0.0043 A` RMSD and `4.65e-5 eV/atom` difference. A targeted rerun under the final declared gates passed with `2.4e-4 A` RMSD and zero step difference. Both artifacts are preserved.

At B16, inactive converged graphs consumed 59.2%, 68.8%, 52.1%, and 37.4% of graph evaluations for 46, 92, 184, and 276 atoms. This explains why relaxation speedup is much smaller than single-point speedup and why B8 can outperform B16 for larger structures.

## Commands

```bash
CUDA_VISIBLE_DEVICES='' /public/home/lmy/.conda/envs/lmy/bin/python -m pytest -q
CUDA_VISIBLE_DEVICES=0 /public/home/lmy/.conda/envs/lmy/bin/python benchmarks/benchmark_relaxation.py --output runs/production_relaxation_scaling.json
CUDA_VISIBLE_DEVICES=0 /public/home/lmy/.conda/envs/lmy/bin/python benchmarks/benchmark_relaxation.py --atom-counts 92 --sample-count 16 --batch-sizes 8 --repeats 1 --reference-repeats 1 --output runs/relax_revalidation_92_b8_final.json
/public/home/lmy/.conda/envs/lmy/bin/python -m ruff check atombit_batch src tests benchmarks/benchmark_relaxation.py
```

## Limitations

- Results cover one checkpoint, one H100, float32, fixed cells, and 16 structures per atom-count stratum.
- E0 is omitted because it does not affect forces or optimization trajectories.
- CUDA scatter reductions are not bitwise deterministic and can change a FIRE branch near the stopping threshold; final-state tolerances are therefore more meaningful than identical trajectories.
- Converged graphs remain in the batch and still incur model work until the slowest graph in the chunk converges.
- A zero neighbor skin matches the ASE reference but rebuilds every neighbor list at every step.

## Next experiment

Implement active-batch compaction and edge-count-aware bucketing, then repeat this fixed-pool experiment. Compaction should target the measured 37-69% wasted graph-evaluation fraction before attempting larger batch sizes.
