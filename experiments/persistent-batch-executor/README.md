# Persistent batch executor

## Hypothesis

One long-lived process per GPU can reuse the loaded MLIP, CUDA context,
allocator state, and warmed tensor shapes across independent relaxation pools.
This should remove a lifecycle cost that batching, compaction, refill, and
neighbor caching do not address.

## Method

- Workload: signed H46 and H276 pools, 256 structures per call.
- Optimizer: variable-cell batched BFGS with `FrechetCellFilter`.
- Models: smooth-RMS AtomBit float32 and MACE-OFF23 Small float64.
- Comparison: three calls through the existing fresh process scheduler versus
  three calls through one `BatchExecutor`.
- Cold calibration: C256, an intentional whole-pool override used to isolate
  process/model reuse. This is not the package's C32 automatic default.
- GPU counts: 1, 2, and 4 for AtomBit H46; the model/size adaptation checks use
  two GPUs.
- Timing: synchronized end-to-end wall time, one measured session. The first
  call includes worker startup; the mean after the first call and final call
  expose amortized and fully shape-warmed behavior separately.

Run records are produced by `benchmarks/benchmark_persistent_executor.py`.
Matched summaries are produced by
`benchmarks/summarize_persistent_executor.py`. Raw tensor records remain in the
ignored `results/raw/` directory; `results.json` is the compact committed
record.

## Result

Persistent execution passed task-exactly-once, ordered reassembly, stable PID,
failure, and allocator-generation restart tests. It reduced steady-state wall
time in every measured pair without increasing steady-state peak reservation.
The benefit grows with worker count because the fresh path reloads one model
per GPU for every call.

| Model/workload | GPUs | Fresh post-first mean | Persistent post-first mean | Speedup | Peak reserved |
|---|---:|---:|---:|---:|---:|
| AtomBit H46 B128, 2 steps | 1 | 7.60 s | 2.00 s | 3.80x | 9.22 GiB |
| AtomBit H46 B128, converged | 2 | 38.01 s | 29.56 s | 1.29x | 9.29 GiB |
| AtomBit H46 B128, 2 steps | 4 | 14.86 s | 1.39 s | 10.70x | 9.22 GiB |
| AtomBit H276 B64, 2 steps | 2 | 13.28 s | 4.28 s | 3.11x | 24.76 GiB |
| MACE H46 B128, 2 steps | 2 | 22.13 s | 2.22 s | 9.97x | 9.20/9.19 GiB |
| MACE H276 B64, 2 steps | 2 | 25.30 s | 4.05 s | 6.24x | 27.05/26.93 GiB |

The table reports one synchronized session, not a statistical confidence
interval. At four GPUs, call two still warms production shapes on workers that
did not receive the cold-start chunk; the fully warm third call is 0.92 s.

The strongest scientific limitation is AtomBit long-trajectory repeatability.
Two independently launched deterministic float32 BFGS runs can converge to
different local endpoints, and the same variation occurs between repeated
fresh runs and repeated persistent runs. Therefore the full-convergence
fresh-versus-persistent endpoint gate is inconclusive for AtomBit. A two-step
AtomBit comparison agrees within the existing float32 tolerance, while MACE
two-step results agree to near machine precision. This change alters worker
lifecycle only; it does not alter model, graph, optimizer, or force kernels.

## Decision

Retain `BatchExecutor` for repeated independent pools. Do not use its first-call
timing as evidence of acceleration: starting every resident worker can make the
first call slower. Close the executor after the final pool to release worker
models and CUDA reservations.
