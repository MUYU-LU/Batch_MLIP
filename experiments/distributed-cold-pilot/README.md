# Distributed cold pilot

## Question

The persistent-executor benchmark used `multi_gpu_cold_start_jobs=256` on an
R256 pool. That override made one worker consume the entire first call while
the other initialized workers remained idle. The package default is C32. This
experiment tests whether the existing small pilot already solves the measured
underutilization and whether explicit duplicated shape warmup is justified.

## Method

- Signed R256 H46 and H276 workloads.
- Variable-cell BFGS, two steps per call, three calls per executor.
- AtomBit smooth-RMS float32 and MACE-OFF23 Small float64.
- C256 whole-pool calibration versus C32 retained pilot jobs followed by
  distributed production.
- One synchronized session per point; these are mechanism timings, not
  full-convergence application results or confidence intervals.

Raw records are produced by `benchmarks/benchmark_persistent_executor.py` and
summarized by `benchmarks/summarize_cold_pilot.py`.

## Results

| Model/workload | GPUs | C256 calls (s) | C32 calls (s) | Session speedup | Peak reserve C256/C32 |
|---|---:|---:|---:|---:|---:|
| AtomBit H46 B128 | 4 | 10.92, 1.86, 0.92 | 10.70, 0.90, 0.91 | 1.09x | 9.22/4.83 GiB |
| AtomBit H276 B64 | 2 | 14.90, 4.47, 4.08 | 12.53, 4.07, 4.06 | 1.14x | 24.76/24.88 GiB |
| MACE H46 B128 | 2 | 22.23, 2.58, 1.67 | 22.51, 1.68, 0.84 | 1.06x | 9.19/9.43 GiB |
| MACE H276 B64 | 2 | 28.24, 4.47, 3.63 | 25.18, 2.81, 2.81 | 1.18x | 26.93/26.97 GiB |

C32 does not satisfy the predeclared 20% first-call improvement gate: process
and model loading dominate H46, while H276 improves by 12–19%. It nevertheless
removes avoidable idle-worker shape initialization, improves every three-call
session by 6–18%, and sharply lowers AtomBit H46's first-call reservation.

MACE C32/C256 outputs agree to near machine precision. AtomBit H276 differs by
at most `1.91e-6 A` in positions, `6.49e-5 eV` in energy, and
`3.01e-4 eV/A` in forces, consistent with its established float32
cross-batch tolerance. Converged-step fields are identical in all two-step
comparisons.

## Rejected shape warmup

An experimental full-production-shape force/stress evaluation was dispatched
once to every worker after C32 capacity selection. For MACE H46 it removed the
second-call penalty, but increased the first call from 22.51 to 25.59 s. The
three-call session became 8.98% slower (27.26 versus 25.02 s), so the mechanism
was removed rather than exposed in the public API.

## Decision

Do not add a new distributed-calibration implementation: `BatchExecutor`
already retains C32 pilot outputs and dispatches the remaining work across all
workers without duplicate optimization. Change the benchmark default from the
artificial C256 override to the package C32 default, and retain a regression
test that the pilot and production sets cover each input exactly once while
both workers receive production work.

CUDA-MPS is intentionally excluded here because the existing MPS harness times
after worker/model startup, unlike this executor session. A defensible MPS
comparison requires a lifecycle-matched full-task experiment.
