# OMC-CSP Policy Composition

## Hypothesis

The accepted scheduling mechanisms can be composed into one deterministic
decision chain for periodic variable-cell relaxation. Unsupported decisions
must retain explicit conservative fallbacks instead of being inferred from
chemistry or hidden timing pilots.

## Scope

This is a policy-composition and reporting change. It does not modify BFGS,
FIRE, graph construction, convergence criteria, or the structures assigned to
an execution chunk. No speedup is claimed.

The manifest distinguishes:

- the generic scheduling architecture;
- the detected numerical task instance;
- general structure atoms from MLIP active edges;
- task auxiliary state from graph execution-policy candidate edges;
- hardware-bound coefficients from the unbound workload profile;
- outer bucketing and device assignment;
- inner graph, capacity, compaction, and drain/refill decisions;
- evidence sources and conservative fallback reasons.

The frozen workload manifests remain task-independent selections of structures.
Model, task, graph-execution, and eventually hardware bindings are stored as
separate hashed planning sidecars. The validated AtomBit variable-cell BFGS
sidecars and their exact contract are recorded in `layered-profiles.yaml`.

OMC is an application label supplied by the study, not something inferred from
atomic numbers at runtime. The detected execution task is therefore reported
as periodic variable-cell relaxation.

## Validation

Pure policy tests cover homogeneous and mixed, single- and multi-wave plans.
Integration tests require both automatic single-device and multi-device
results to expose the same manifest schema. The existing full test suite
remains the numerical-correctness gate.
