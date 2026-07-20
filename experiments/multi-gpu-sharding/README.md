# Independent-process multi-GPU sharding

## Hypothesis

One independent process and model replica per H100 should reduce fixed-workload
BFGS wall time, while deterministic largest-processing-time-first assignment by
generalized BFGS dimension should balance mixed 46/276-atom inputs.

The implementation adds model-independent `WorkerShard`, `balance_work`, and
`run_parallel_workers` interfaces. Child processes load and warm their own model,
signal readiness, start together, and return CPU payloads. Results are restored
to original input order. AtomBit and MACE preparation remains outside the generic
executor.

## Protocol

- Hardware: seven NVIDIA H100 80 GB GPUs (indices 0-6).
- Fixed main pool: 256 jobs; either 256 x 276 atoms or 128 x 46 plus 128 x 276.
- Worker counts: 1, 4, and 7; resident capacity B64; one timing per point.
- Optimizer: variable-cell BFGS with FrechetCellFilter, `fmax=0.05 eV/A`,
  `max_steps=500`, float64 optimizer state, immediate refill, active compaction.
- AtomBit: float32 model, cached topology, `skin=0.5 A`.
- MACE-OFF-Small: float64 model, cached tensor-state graph, `skin=0.5 A`.
- CUDA deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8` enabled.
- Timing starts only after every worker loads, warms, and synchronizes its model.
  Cold end-to-end time is reported separately.
- The user requested no timing repeats. Therefore no uncertainty estimate is
  available and assignment variance is visible directly in worker times.

## Main results

Optimization-only timings for the identical 256 input structures:

| Model | Workload | GPUs | Time (s) | systems/s | Speedup | Efficiency |
|---|---|---:|---:|---:|---:|---:|
| AtomBit | 276 atoms | 1 | 282.32 | 0.907 | 1.00x | 100.0% |
| AtomBit | 276 atoms | 4 | 83.56 | 3.064 | 3.38x | 84.5% |
| AtomBit | 276 atoms | 7 | 54.20 | 4.723 | 5.21x | 74.4% |
| AtomBit | mixed | 1 | 199.46 | 1.283 | 1.00x | 100.0% |
| AtomBit | mixed | 4 | 60.52 | 4.230 | 3.30x | 82.4% |
| AtomBit | mixed | 7 | 41.44 | 6.178 | 4.81x | 68.8% |
| MACE | 276 atoms | 1 | 291.77 | 0.877 | 1.00x | 100.0% |
| MACE | 276 atoms | 4 | 101.27 | 2.528 | 2.88x | 72.0% |
| MACE | 276 atoms | 7 | 53.56 | 4.780 | 5.45x | 77.8% |
| MACE | mixed | 1 | 209.07 | 1.224 | 1.00x | 100.0% |
| MACE | mixed | 4 | 69.76 | 3.669 | 3.00x | 74.9% |
| MACE | mixed | 7 | 45.32 | 5.649 | 4.61x | 65.9% |

All 3,072 main-point jobs converged. Every original index was returned exactly
once, sources remained in original order, and convergence flags match W1.

W7 is sublinear for two measured reasons. First, 36-37 homogeneous jobs per GPU
cannot fill B64. Model-call counts rise from 514 to 1,703 for AtomBit and from
493 to 1,791 for MACE. Second, equal atom count does not predict optimizer steps:
worker imbalance reaches 1.51x for homogeneous AtomBit and 2.26x at MACE W4.
For mixed W7, the analytic DOF-squared cost underprices small-job and per-system
overhead, producing 1.77-2.04x worker imbalance.

## Cold small-pool control

| Model | GPUs | Optimize (s) | End-to-end (s) | Optimize speedup | End-to-end speedup |
|---|---:|---:|---:|---:|---:|
| AtomBit | 1 | 29.87 | 54.12 | 1.00x | 1.00x |
| AtomBit | 7 | 12.04 | 51.29 | 2.48x | 1.06x |
| MACE | 1 | 38.71 | 80.06 | 1.00x | 1.00x |
| MACE | 7 | 14.36 | 59.81 | 2.70x | 1.34x |

Seven cold workers are not worthwhile for 32 AtomBit jobs. MACE gains only
1.34x end-to-end because its larger per-job cost better amortizes model startup.
The executor is retained, but GPU count remains explicit rather than automatic.

## Numerical limitation

Sharding changes batch membership and refill order. Deterministic CUDA makes each
configuration repeatable, but does not make different batch shapes bitwise equal.
BFGS amplifies small floating-point differences and a minority of structures enter
different local minima. Against W1, AtomBit W4/W7 has 18-23 step-count mismatches;
MACE has 37-70. Median final-state differences are zero or near machine precision,
but worst energy differences are 0.94-4.59 eV and worst position differences are
0.94-2.16 A. These are not force-path errors: B1, batch equivalence, ASE BFGS, and
MACE cached NVE tests pass. They do mean strict final-state reproducibility across
GPU counts is not provided by job-level sharding.

## Conclusion and next experiment

Independent-process sharding materially accelerates 256-job pools on both MLIPs.
The next step should combine a persistent worker pool with invariant work units.
Use fixed micro-pools that preserve batch/refill schedules across GPU counts, and
schedule those units with measured pilot cost (atom/edge count, observed steps,
and fixed per-system overhead). This can amortize 25-36 s model startup, improve
straggler balance, and provide strict result equivalence. GPU-native neighbor-list
work is not the immediate limiter in this matrix: total neighbor-search device
time stays approximately constant while model-call inflation and BFGS imbalance
grow with worker count.

Machine-readable summaries and SHA-256 hashes of every raw record are in
`results.json`; raw profiler/event logs are under `runs/multi_gpu_sharding/`.
