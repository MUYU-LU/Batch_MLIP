# Active task-aware baseline

The files beside this note are immutable reference copies from
`task_aware_batched_atomistics_protocol.zip` in packet version `1.0.0`
(2026-07-20). The archive SHA-256 is
`9ffd80b7086e2d1053f91873bfc9c6e44230b94718ce697ec7c11206df53e462`.

The packet's bundled code is historical version 0.1.2. It did not replace this
repository. Implementation work starts from active repository commit
`41694f704703a8d9a3b6de4e07b184c46be61595` (version 0.2.0).

## Foundation status

- Implemented: signed `WorkloadManifest`, derived `TaskProfile`, and registry-compatible
  `RunTelemetry` schemas.
- Implemented: deterministic T2 definitions for the initial homogeneous, mixed,
  fixed-cell, fixed-horizon, and model-specific step-variation workloads.
- Deferred: `EVAL-REPLAY50-v1` until genuine frozen trajectory frames exist.
- Deferred: execution benchmarks and planner-policy claims. This foundation changes no
  calculator, optimizer, integrator, or numerical result.

The signed JSON manifests under `benchmarks/workloads/manifests/` are authoritative.
Their CSV files are human-auditable projections and are not independent workload
definitions.
