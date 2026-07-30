# Project Logic

This directory is the authoritative research map for Batch MLIP. Historical
protocols under `research/task-aware/` and individual experiment conclusions
remain useful provenance, but they do not override the status declarations
here.

## Central Question

Given a pool of structures, an MLIP, a user-selected numerical method, and a
set of available devices, select a scientifically valid execution plan that
minimizes total makespan without exceeding the memory budget.

```text
structures + MLIP + numerical method + available devices
                            |
                            v
               layered workload representation
              structure: atoms
              MLIP graph: active edges + cutoff
              task: state + cell/stress + horizon
              policy: candidate edges + cache
              hardware: calibrated time + memory
                            |
                            v
                  feasible-plan builder
             capacity, buckets, cache, refill,
                compaction, worker assignment
                            |
                            v
                  prior-guided selection
             accepted mechanisms + learned cost
                 model + conservative fallback
                            |
                            v
                    tensor execution
                            |
                            v
             makespan + memory + endpoints + telemetry
```

The four informal workload quadrants, small/large pool crossed with
small/large systems, are explanatory labels. Runtime decisions use continuous
descriptors:

- `m`: resident memory, including graphs, model temporaries, and numerical
  state;
- `w`: work per useful model evaluation;
- `h`: requested or predicted number of evaluations;
- `v`: topology volatility from atomic motion and cell deformation.

Atom count alone is never the workload definition. Nor is there a universal
formula that combines atoms, edges, and optimizer state before the MLIP, task,
execution policy, and hardware have been bound.

## Abstract Framework and OMC-CSP Instance

The abstract framework defines an action space. The runtime instance fixes that
action space from the supplied numerical contract. It does not infer an
application label from chemical composition.

For the current OMC-CSP study, `BFGS + FrechetCellFilter` over fully periodic
structures is detected computationally as
`periodic_variable_cell_relaxation`. OMC CSP is the study context attached to
that task, not a separate optimizer implementation.

The deterministic policy is applied in this order:

1. Preserve the model/task-independent structure identity and atom count.
2. Bind the MLIP graph profile: cutoff, active edges, force mode, and dtype.
3. Bind the OMC task auxiliary profile: variable-cell BFGS, stress,
   `D=3N+9`, dense state `D^2`, dense eigensolver work `D^3`, and variable
   horizon.
4. Bind graph execution policy separately: skin, candidate edges, cache, and
   neighbour backend.
5. Apply hardware-calibrated time/memory coefficients and form compatible
   buckets with a maximum within-bucket cost ratio of two.
6. Use one representative model forward and explicit dense-BFGS state
   allowance to pack resident queues below 85% of device memory.
7. On one device, merge memory waves into refill only when the packaged
   execution contract and workload evidence match; otherwise use active drain.
8. On multiple devices, subdivide safe queues for occupancy and work stealing;
   retain active drain because multi-GPU refill is not accepted.
9. Resolve `neighbor_backend="auto"` per rebuild and use the calculator's
   explicit skin/cache contract.
10. Record the complete decision plus observed convergence-step dispersion in
   `metadata["scheduling"]["policy_manifest"]`.

This composition is complete in the engineering sense that every branch has a
defined action or conservative fallback. It is not a claim that cache skin,
duration, or refill can already be predicted optimally for unseen workloads.

## Scope and Authority

The user owns the physical and numerical problem:

- structures and pool membership;
- MLIP and checkpoint;
- optimizer or integrator;
- convergence and thermodynamic parameters;
- available devices and required correctness tier.

The runtime may select:

- memory-safe resident capacity;
- cost-compatible buckets;
- graph backend and an already validated cache policy;
- active drain or an evidence-supported refill policy;
- compaction strategy;
- worker count up to the supplied device count;
- whole-pool or micro-pool assignment.

The runtime must not silently change the optimizer, precision contract,
potential, cutoff, convergence criterion, or physical ensemble.

## Evidence Layers

1. **Execution foundation:** graph isolation, calculator adapters, optimizer
   and integrator correctness, variable cells, constraints, and restart.
2. **Mechanism evidence:** paired ablations for batching, neighbors, caching,
   compaction, refill, dense BFGS algebra, allocation, and worker execution.
3. **Scheduler v1:** deterministic memory planning plus conservative
   evidence-matched policies. See `baseline_v1.yaml`.
4. **Workload transfer:** contract-consistent benchmarks over distinct OMC-CSP
   families, followed separately by molecular conformer-search workloads.
   `chemical_transfer.yaml` remains a future parent-selection protocol rather
   than the current workload definition.
5. **Applications:** CSP, conformer search, phonons, adsorption, bulk modulus,
   and multi-replica MD after the corresponding validation gates pass.

The experiment registry is complete by construction:
`evidence_registry.csv` contains one row for every directory under
`experiments/`, and a regression test rejects missing or duplicate entries.

Registry evidence statuses mean:

- `accepted_v1_prior`: directly supports a shipped scheduler-v1 decision;
- `validated_component`: correctness or interface capability, not a speed
  default;
- `benchmark_reference`: measured comparison retained as baseline evidence;
- `negative_result`: tested action rejected from automatic selection;
- `mixed_result`: some regimes passed and others failed;
- `superseded`: historically useful but replaced by a later implementation;
- `candidate`: validated locally but not yet committed into the frozen
  baseline.

`transfer_use` records whether an experiment defines a mechanism feature, the
frozen baseline, an action exclusion, or protocol only. It never authorizes
the old timing value as a new planner label.

## Current Claim Boundary

The project currently supports the following claim:

> A generic tensor runtime can execute independent MLIP relaxation pools with
> deterministic memory-bounded batching, active compaction, conservative
> evidence-gated refill, multi-device work distribution, and an inspectable
> task-to-execution policy manifest.

It does not yet support these stronger claims:

- universal prediction of the optimal execution plan for unseen chemistry;
- automatic multi-GPU refill;
- invariant internal schedules across GPU counts;
- automatic MPS fallback;
- automatic selection of cache skin or numerical algorithm;
- task-aware production scheduling for NVE, NVT, or NPT;
- application-level scientific improvement in CSP or conformer search.

## Reframed Experimental Narrative

The paper and engineering evaluation use four sections:

1. numerical correctness and scientific equivalence;
2. isolated acceleration-mechanism ablations;
3. end-to-end scheduler v1 against CUDA MPS;
4. planner transfer and regret on held-out workload families and, later,
   chemically held-out parent systems.

Old timings define mechanisms, exclusions, and the frozen v1 comparison. They
are not training labels for the chemical-transfer planner because their
models, workloads, timing scopes, hardware use, and execution contracts are
not uniformly matched.

## Immediate Work Sequence

1. Freeze exact unique OMC-CSP pools from the existing candidate families.
2. Validate nested `P64/P512/P2048` intra-family and inter-family manifests.
3. Classify those pools by atom, candidate-edge, and variable-cell BFGS costs.
4. Run paired scheduler-v1 and CUDA-MPS experiments under one frozen contract.
5. Derive and validate OMC-CSP policy decisions on held-out families.
6. Construct the molecular conformer-search workload as a separate instance.
7. Revisit parent-level chemical selection only when testing chemical transfer
   beyond the available workload families.
