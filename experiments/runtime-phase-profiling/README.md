# Runtime phase profiling

## Change

`RuntimeProfiler` is an opt-in context manager that records internal calculator,
graph, optimizer, compaction, and refill phases without adding arguments to the
calculator or optimizer protocols. CUDA phases use deferred events and are
resolved with one synchronization when the context closes. CPU execution uses
`perf_counter`.

AtomBit reports common neighbor construction, graph-view creation, model
forward, and autograd. MACE reports state-to-ASE conversion, `AtomicData`
construction, PyG collation, device transfer, and model forward. Model-native
types remain confined to adapters.

Both variable-cell scaling benchmarks accept `--profile-runtime` and store the
phase samples, aggregate phase statistics, occupancy samples, graph sizes,
compaction events, and refill events in the point JSON.

## AtomBit screening result

The screening workload contains 128 fixed-manifest 46-atom jobs, resident B64,
20 forced variable-cell BFGS steps per system, float32 model inference, and
float64 optimizer state. Three complete runs were made with profiling disabled
and three with profiling enabled.

| measurement | median seconds |
|:--|--:|
| profiler disabled | 9.8714 |
| profiler enabled | 9.8594 |
| internal profiled span for median run | 9.8591 |

The median profiled run was 0.12% faster. This is indistinguishable from zero
overhead at the resolution of the experiment and passes the planned 1% overhead
gate; it is not interpreted as a speedup.

The complete output records are bitwise identical: positions, cells, energies,
forces, stresses, convergence flags, and step counts. Both runs performed 42
model evaluations, 2,688 graph evaluations, and 42 neighbor rebuilds.

## Phase decomposition

The non-overlapping top-level phases in the profiled AtomBit run are:

| phase | seconds | share |
|:--|--:|--:|
| full-BFGS update and eigensolves | 4.3617 | 44.2% |
| neighbor update | 3.1027 | 31.5% |
| model autograd | 1.2847 | 13.0% |
| model forward | 0.6407 | 6.5% |
| refill repack | 0.0103 | 0.1% |
| other orchestration | 0.4590 | 4.7% |

Within neighbor update, raw `matscipy` search accounts for 2.9502 seconds.
The result confirms that neighbor caching is worth a correctness pilot, but it
also shows that full-BFGS eigensolves are the largest single target.

This forced-step workload has one refill event after all first-wave systems
reach their step limit. It validates refill timing but does not represent the
one-at-a-time convergence pattern needed to choose a threshold refill policy.

## MACE smoke result

A four-job B2 CUDA smoke run in the dedicated MACE environment completed six
profiled evaluations and one refill. It recorded all MACE adapter graph phases,
model forward, BFGS update, and refill repacking. The small B2 workload is a
functional adapter check, not a MACE performance conclusion.

## Validation

- Default suite: 50 passed, two optional MACE tests skipped.
- Targeted BFGS, profiler, and package-layout tests: 23 passed.
- MACE-OFF-Small profiled CUDA benchmark: passed in `MACE_clean`.
- Ruff: passed.
- A regression test covers `refill_batch_size == pool_size`, an existing state
  aliasing bug exposed during the profiler screening.

## Decision

Stage 0 is successful: the profiler is generic, opt-in, numerically neutral,
and produces actionable phase accounting for both adapters. The next experiment
is the Stage 1 generic variable-cell neighbor-cache correctness pilot at B1 and
a small heterogeneous batch. Performance expansion to B64 is conditional on
that validation.
