# AtomBit OMC-CSP v1 Freeze

## Status

This document defines the frozen AtomBit OMC-CSP production instance of the
generic Batch MLIP framework. `baseline_v1.yaml` is the machine-readable
authority. Individual experiment documents provide provenance but do not
override this decision.

The freeze fixes execution policy, not scientific inputs. Users continue to
choose structures, AtomBit checkpoint, optimizer, cell treatment, convergence
criteria, and devices.

## Bound Contract

The validated production contract is:

- AtomBit smooth-RMS fp32 model evaluation;
- float64 optimizer state;
- BFGS or FIRE selected explicitly by the user;
- optional `FrechetCellFilter` for variable-cell relaxation;
- model cutoff `6.0 A` and explicit skin/cache configuration;
- deterministic process, NumPy, Torch, CUDA, and CPU-thread settings;
- NVIDIA H100 80-GB hardware for the signed capacity policy.

Changing the model, cutoff, dtype, optimizer, or hardware invalidates the
signed AtomBit capacity match and invokes a conservative fallback. It does not
reuse AtomBit coefficients under another MLIP.

## Accepted Automatic Workflow

For a manifest-backed multi-GPU OMC-CSP pool:

1. Validate the immutable planning profile against the manifest, calculator,
   optimizer, dtype, cutoff, cell/stress mode, and neighbor contract.
2. Bind the signed H100 capacity policy when the exact execution fingerprint
   matches; otherwise use the bounded representative memory probe.
3. Build cost-compatible buckets from atom, candidate-edge, model-work, and
   dense BFGS `D^2` state descriptors.
4. Pack resident chunks below the 85% planning budget with the measured growth
   margin. Runtime reserved-memory high-water up to 91% is accepted for long
   trajectories.
5. Materialize source structures lazily with task-aware CPU loader processes
   and bounded global prefetch.
6. Dispatch one bucket-stratified initial wave, then use descending-cost work
   stealing across persistent GPU workers.
7. Execute active drain with active compaction. Converged structures are
   removed; a GPU requests another complete resident chunk after drain.
8. Rebuild exact candidate graphs with neighbor policy v2 and preserve valid
   cached topology between rebuilds.
9. Reassemble results in immutable manifest order and record the complete
   policy, timings, memory, convergence, and phase telemetry.

## Accepted Defaults

- Multi-GPU inner policy: active drain.
- Multi-GPU refill: disabled.
- Outer queue: bucket-stratified initial wave followed by work stealing.
- Compaction: active and topology preserving.
- Materialization: manifest-lazy global prefetch.
- Neighbor selection: `auto`, using the validated H100 candidate-graph rule.
- Allocator: expandable segments only for matched AtomBit variable-cell
  BFGS/FIRE; native otherwise.
- Tail recovery: never silent; an application benchmark must request and
  report it explicitly.

## Explicit Experiments, Not Defaults

The following mechanisms remain callable for controlled experiments but are
not eligible for automatic selection:

- `manifest_multi_gpu_refill_policy="local_compatible"`;
- `manifest_multi_gpu_refill_policy="streaming_compatible"`;
- heterogeneous arena refill;
- automatic MPS fallback;
- optimizer or precision substitution.

Private refill regressed at G2/G4. Bounded streaming recovered the regression
and improved G8, but missed the unchanged promotion gates. Those results do
not change the active-drain default.

## Production Evidence

The strict unique-T2 P6000 comparison uses eight H100 GPUs, variable-cell BFGS,
`fmax=0.01`, `max_steps=3000`, and no tail recovery:

| Method | Makespan | Converged | Peak memory |
|:--|--:|--:|--:|
| Frozen active drain | 2245.09 s | 5986/6000 | 76.68 GB reserved |
| ASE/CUDA-MPS16 | 4706.15 s | 5985/6000 | 77.70 GB sampled |

The frozen execution is 2.096x faster. The methods have identical source
coverage but not identical trajectories or convergence flags. This supports a
makespan claim under a common optimization contract, not local-minimum
identity.

## Transfer Boundary

The following framework components transfer unchanged to MACE:

- manifests and source materialization;
- optimizer and cell-filter interfaces;
- active compaction;
- outer work stealing;
- telemetry and result reassembly.

MACE must provide its own calculator adapter metadata, cutoff/graph profile,
allocator evidence, memory/time coefficients, signed capacity policy, and
correctness gates. The AtomBit planning profile and capacity coefficients are
not transferable.

## Change Control

After the implementation commit is recorded in `baseline_v1.yaml`, changes to
the frozen AtomBit path require all of:

1. an isolated hypothesis and experiment record;
2. unchanged scientific inputs in the paired comparison;
3. convergence, endpoint, memory, and failure reporting;
4. the complete unit/integration suite;
5. an explicit baseline-version decision.

MACE adaptation must not modify this AtomBit baseline implicitly.
