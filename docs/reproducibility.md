# Reproducibility Contract

Deterministic scheduling and deterministic numerical execution are separate
requirements. A sorted batch plan does not by itself control library random
states or CUDA kernel selection.

## Controlled Runs

OMC-CSP scheduler and CUDA-MPS comparisons use:

- process seed `20260729`;
- `PYTHONHASHSEED=20260729` before Python starts;
- Python `random`, NumPy global RNG, and all PyTorch RNGs seeded;
- `torch.use_deterministic_algorithms(True)`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization;
- cuDNN benchmark disabled and deterministic mode enabled;
- TF32 disabled and float32 matrix multiplication precision set to `highest`;
- one intra-op and one inter-op CPU thread per worker;
- spawned workers configured before model loading, warm-up, or CUDA use.

Use the same contract for the tensor implementation and CUDA-MPS baseline:

```bash
PYTHONHASHSEED=20260729 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python benchmark.py --deterministic --seed 20260729
```

`configure_reproducibility()` records whether `PYTHONHASHSEED` was already
present. Setting it from inside Python controls subsequently spawned workers
but cannot retroactively change the current interpreter's hash seed.

## Seed Roles

The process seed is a defensive execution control. It must not define physical
stochastic trajectories.

- The workload-selection seed remains `20260729`; execution seed sweeps reuse
  the identical signed manifest and never select different CIFs.
- BFGS and FIRE optimization are nonstochastic. Results must be invariant to
  process seeds `20260729`, `20260730`, and `20260731`.
- NVE velocity initialization and stochastic NVT use immutable per-job seeds
  from the workload manifest.
- Repartitioning an MD pool must preserve each job's seed and random state.
- A changed per-job MD seed is expected to generate a different trajectory.

## OMC-CSP Gate

Before timing the full workload matrix:

1. Run one fixed `P64` workload twice with seed `20260729`.
2. Run the same workload once with seeds `20260730` and `20260731`.
3. Require identical scheduling manifests for all four runs.
4. Require identical convergence flags and steps for a bitwise tier; otherwise
   apply the documented numerical endpoint tolerance and report the tier.
5. Run the same gate for Batch MLIP and four-worker CUDA MPS.

These are correctness runs, not timing repeats. Performance results still use
one timed execution per frozen workload unless the protocol explicitly changes.
