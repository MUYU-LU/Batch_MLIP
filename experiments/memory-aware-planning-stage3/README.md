# Stage 3 memory-aware planning and bucketing

## Implementation

The generic `BatchPlanner` profiles each input structure by atom count,
directed candidate-edge count, and full variable-cell BFGS dimension
`D = 3N + 9`. A non-negative peak-memory model is fitted as

```text
fixed + atoms * c_atom + edges * c_edge + sum(D^2) * c_hessian
```

The Hessian coefficient reserves at least one optimizer-dtype byte term for
every matrix element. The planner separates systems whose estimated costs differ
by more than a configurable ratio, then assigns each pending queue the largest
resident capacity below a byte budget. Original input indices are retained so
bucket results can be restored to input order.

## Calibration

Existing full-BFGS B64 peaks at 46, 92, 184, and 276 atoms were used for fitting.
All B128 measurements were held out for validation.

| model | maximum held-out B128 error |
|:--|--:|
| AtomBit | 1.30% |
| MACE-OFF-Small | 0.36% |

The online 256-system workload profiling step took 0.69 seconds for AtomBit and
0.37 seconds for MACE in isolated planner-only runs. Calibration graph profiling
is an offline operation and was excluded from optimization timing.

## Decision-gated benchmark

The workload contains 128 interleaved 46-atom and 128 interleaved 276-atom
structures. The planner used a 32 GiB allocation budget, maximum B128, and a
maximum 2x within-bucket cost ratio. Each point is one deterministic timing.

| model | policy | queues and capacities | time (s) | systems/s | peak allocated (GB) |
|:--|:--|:--|--:|--:|--:|
| AtomBit | fixed | mixed 256, B64 | 225.580 | 1.135 | 14.904 |
| AtomBit | planned | 276/B69 + 46/B128 | **215.210** | **1.190** | 27.935 |
| MACE | fixed | mixed 256, B64 | 234.302 | 1.093 | 15.883 |
| MACE | planned | 276/B79 + 46/B128 | **225.314** | **1.136** | 33.540 |

Planned bucketing improves AtomBit by 4.82% and MACE by 3.99%. Neither reaches
the required 5% throughput gate. Both stay within the 32 GiB budget
(`34.360 GB`). The conservative planned peak is `34.346 GB` for AtomBit and
`34.011 GB` for MACE. AtomBit overpredicts the executed peak by 23.0% because
the plan uses the largest `skin=0.5 A` candidate graph in each queue while its
calibration used mean physical-edge counts. MACE overpredicts by 1.4%.

## Numerical result

Every structure converges and convergence flags agree between fixed and planned
execution. Different queue composition changes floating-point reduction context
and therefore the optimization trajectory. AtomBit has 45 step-count differences
and MACE has 37; both can reach different nearby minima. Bucketing is therefore
a scheduling transformation with convergence-level equivalence, not bitwise
trajectory equivalence.

## Decision

The memory estimator, safety budget, bucketing, and order restoration are kept.
They provide a generic OOM-prevention and workload-inspection interface with
strong held-out memory prediction. Automatic planned execution is not made the
default because the single-run speedups miss the 5% gate.

The conditional small-pool and homogeneous endpoint matrix is stopped. The next
performance step should not simply enlarge resident batches; it should either
preserve cached topology during active compaction or proceed to independent
multi-GPU workers using the planner only for per-device safety.
