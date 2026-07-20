# Agent Experiment Rules

These rules are mandatory for autonomous experiments on the task-aware batched atomistic runtime.

## 1. Protect the scientific reference

1. Never modify a frozen workload manifest in place. Create a new version.
2. Never modify the reference single-system implementation merely to make a candidate implementation agree.
3. Preserve classic FIRE, BFGS, force, stress, and MD parity tests as release gates.
4. Do not compare speed across configurations that fail the declared equivalence tier.
5. Do not hide nonconvergence, NaNs, OOM retries, or different final minima.

## 2. Distinguish evidence status

Every result is labeled as one of:

- exploratory;
- preliminary engineering;
- replicated controlled;
- held-out planner validation;
- application validation.

A single run cannot be labeled publication-grade. A superseded implementation remains archived but is not mixed with final results.

## 3. Freeze the workload before tuning

Before tuning a feature, record:

- workload ID/version and manifest hash;
- model/checkpoint hash;
- exact hardware and software environment;
- optimizer/integrator and all parameters;
- random seeds;
- timing scope;
- required equivalence tier.

Easy/hard optimization strata are computed once from the reference B1 run and frozen. Candidate implementations must not redefine difficulty.

## 4. Use paired and repeated experiments

- Compare configurations on the same jobs and seeds.
- Randomize or counterbalance run order.
- Use at least five independent process repetitions for long runs and ten for short/noisy runs.
- Synchronize the GPU at timing boundaries.
- Report mean, median, bootstrap 95% confidence interval, coefficient of variation, and failures.
- Treat repeated copies of the same structure as technical replicates; cluster analysis by unique structure.

## 5. Separate timing scopes

Always report:

1. model kernel;
2. warm steady-state end-to-end;
3. cold end-to-end.

Never compare a warm candidate against a cold baseline. Report GPU-seconds/job in addition to wall-clock speedup for multi-GPU studies.

## 6. Cache experiments

A cache change must demonstrate:

- zero missed active edges;
- energy/force/stress parity versus exact neighbor reconstruction;
- correct variable-cell invalidation;
- candidate/active edge inflation;
- rebuild interval and neighbor/filter/model time breakdown.

For AtomBit-like degree normalization, inactive skin edges must be filtered or excluded from degree calculation. A zero radial envelope alone is insufficient.

## 7. Refill experiments

Refill must be tested on completion-heterogeneous workloads. Record:

- active cost fraction over time;
- inactive model work;
- compaction and admission time;
- insertion-size distribution;
- compile shape variants;
- survivor cache retention.

One-by-one refill is accepted only when fixed-slot or topology-preserving replacement is actually low cost. Refill grouping without removing reconstruction overhead is not considered a mechanism improvement.

## 8. Batch and memory experiments

Do not equate batch size with compute load. Record atoms, active edges, candidate edges, and optimizer-state dimension. Select the best batch by accepted goodput under a memory safety constraint, not by maximum allocated memory.

Every OOM backoff is logged as a failure of the proposed plan, even when an automatic retry succeeds.

## 9. Multi-GPU experiments

Maintain two separate experiment classes:

- invariant-work strong scaling: fixed micro-pools and identical internal schedules;
- best-configuration throughput: the planner may change the execution plan.

Do not call the second one strong scaling. Also report cold and persistent-worker results separately.

## 10. Planner experiments

- Fit cost models and thresholds only on calibration workloads.
- Freeze the planner before held-out evaluation.
- Compare to a bounded empirical oracle.
- Report runtime regret, OOMs, correctness failures, and the selected-plan explanation.
- Do not tune the planner on the held-out test results.

## 11. MD experiments

NVE requires long-time drift and momentum tests, not only short timing. NVT requires temperature-distribution and kinetic-energy validation. Exact restart must include velocities, thermostat state, RNG state, topology cache, scheduler state, and step count.

NPT/NPH and melting-point claims remain disabled until stress and barostat validation pass.

## 12. Practical significance

The 5% rule is a practical-effect threshold, not a p-value. A speed feature becomes a default only when its paired median gain is at least 5%, the uncertainty interval supports improvement, and correctness passes. Safety features may remain enabled without a speed gain.

## 13. Required output per experiment

Each experiment must produce:

- immutable configuration;
- raw telemetry CSV/JSONL;
- environment report;
- validation report;
- summary statistics;
- failure log;
- final result manifest;
- concise conclusion stating what was and was not established.

