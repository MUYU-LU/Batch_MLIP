# Online autoscheduler

This experiment replaces the explicit research pilot in the normal relaxation
path with production-work autotuning. Cold-start observations come from jobs
that remain in the final result; structures are never run only for calibration.

Version 1 controls tensor capacity and refill and supports homogeneous
multi-GPU work stealing in one Python process. The explicit
`OptimizationPilot` path remains available for frozen, reproducible mechanism
studies.

The experiment is successful only if correctness, memory safety, cold-start
overhead, warm-cache reuse, and regret against the committed manual oracle are
reported together.

## Initial result

All points use variable-cell BFGS, deterministic algorithms, one timing run,
and the signed input order. Every reported workload converged completely.

| MLIP/workload | Cache state | Production queues | Wall s | Peak reserved GiB |
|---|---|---:|---:|---:|
| AtomBit H92 R32 | cold | 1/4/16/11 | 35.89 | 4.39 |
| AtomBit H92 R32 | warm | 32 | 17.23 | 8.60 |
| AtomBit H92 R256 | safe exploration | 64/128/64 | 91.01 | 46.81 |
| AtomBit H92 R256 | warm stable | 128/128 | 81.04 | 46.81 |
| MACE H92 R32 | cold | 1/4/16/11 | 32.03 | 2.98 |
| MACE H92 R32 | warm | 32 | 18.91 | 5.07 |
| AtomBit XATMOV88 R256 | H92 cache transfer | 128/128 | 20.23 | 37.29 |
| AtomBit H92 R256, 4 GPU | warm | 4 x 64 | 35.59 | 17.13/GPU |
| MACE H92 R32, 2 GPU | warm | 2 x 16 | 17.56 | 3.02/GPU |

The AtomBit H92 R32 warm result is within timing noise of the manually selected
17.62 s tensor point. H92 R256 is 0.3% slower than the manual 80.79 s B128
policy. The H92 policy transfers to the independent XATMOV88 organic-crystal
pool and improves on its prior 21.33 s selected tensor result. These are
single-run screens, so differences below 2% remain inconclusive.

An earlier controller admitted B192 after B64. It completed H92 R256 in
79.44 s but reserved 77.44 GiB against a 66.54 GiB safety budget. That result is
rejected even though it was fast. The retained rule limits unmeasured growth to
2x once a probe uses 20% of the budget and stops growth when measured reserved
memory reaches 65%. The rejected raw result is preserved with the accepted
artifacts under `results/raw`.

Cold H92 R32 is intentionally slower because four capacity probes consume most
of a 32-job pool. Cold-start cost is amortized for larger pools and eliminated
for matching future workloads.

Four-GPU AtomBit is 2.28x faster than the one-GPU warm R256 result, or 57%
parallel efficiency. Two-GPU MACE improves R32 by only 1.08x. Per-worker
chunks slow by roughly 35-40% under concurrent execution, identifying
same-process Python/CPU graph contention. Multi-GPU execution is functional and
load-balanced, but a process-isolated worker implementation is required before
claiming strong GPU-count scaling. Automatic MPS execution is also not yet
implemented.
