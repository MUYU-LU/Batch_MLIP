# Task-aware experiment foundation

## Hypothesis

Signed workload identities plus a common telemetry contract can separate task
properties from execution-policy choices. This makes later cache, refill, packing,
batch-size, and multi-GPU comparisons reproducible without changing MLIP or optimizer
numerics.

## Implemented

- Imported the packet's protocol, workload catalog, planner policy, registry template,
  and evidence ledger as immutable references under `research/task-aware/`.
- Added typed `WorkloadManifest`, `WorkloadJob`, `TaskProfile`, and `RunTelemetry`
  schemas with JSON round trips and content-hash verification.
- Generated eight T2 workloads from 64 frozen CIF files. Each job records source and
  normalized-structure hashes, atom ordering, cell/PBC, constraints, duplicate group,
  arrival/order fields, and directed edge counts for four skins.
- Profiled AtomBit at its 6.0 A cutoff and MACE-OFF-Small at its 4.5 A cutoff.
- Constructed separate AtomBit and MACE step-variation workloads from the lowest and
  highest quartiles of their prior single-system ASE-BFGS step counts. Reference run
  artifacts and exact model hashes are recorded in manifest metadata.
- Added a semantic validator for source hashes, composition, ordering, edge-count
  monotonicity, strata balance, CSV projections, and derived profiles.

## Commands

```bash
pytest -q
python -m ruff check batch_mlip tests tools
PYTHONPATH=. python tools/generate_controlled_workloads.py
PYTHONPATH=. python tools/validate_controlled_workloads.py \
  --output experiments/task-aware-foundation/validation.json
```

Generation was run twice and all output checksums were identical. The result is in
`determinism.txt`; semantic validation details are in `validation.json`.

## Interpretation

This experiment creates no performance result. It establishes the controlled inputs
and instrumentation needed for the next measurements. `EVAL-REPLAY50-v1` remains
deferred because genuine frozen trajectory frames do not exist, and `OPT-FIXED50-v1`
is a frozen definition whose fixed-step execution path is not yet implemented.

The replicated R256 pools contain technical repeats of 32 or 64 unique structures.
They measure engine scheduling mechanisms, not dataset-level scientific
generalization. The B1 difficulty labels come from one prior run per model; the MACE
reference recorded `deterministic_algorithms=false`. They are valid frozen scheduling
strata, but they are not uncertainty estimates or endpoint-equivalence evidence.

## Next experiment

Use these exact manifests for a small calibration matrix that measures cache skin,
resident capacity, refill hysteresis, and one-versus-persistent-multi-GPU execution.
Hold optimizer settings and micro-pool identities invariant, write one `RunTelemetry`
record per run, and select policies by accepted-job goodput subject to correctness and
memory gates. Do not benchmark every combination blindly; use the task profiles to
prune inapplicable policies first.
