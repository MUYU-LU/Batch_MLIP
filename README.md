# Batch MLIP

A model-independent interface for **true batched geometry optimization and molecular dynamics** with graph MLIPs. Native adapters are included for AtomBit-style models and MACE.

The engine concatenates independent atomic systems into one heterogeneous graph batch and performs one model forward pass per simulation step. ASE is used at the boundary for structure I/O. An adaptive matscipy/ASE CPU or dense CUDA backend constructs neighbour lists, while PyTorch owns batched model evaluation, optimizer state, and MD integration.

## What is included

- A reusable package under `batch_mlip/`.
- Compatibility copies of the uploaded model under the original `src.*` namespace.
- Exact source snapshots under `original_uploads/`.
- Fixed/variable-cell batched FIRE and full BFGS, plus steepest descent.
- Fixed-cell NVE velocity-Verlet and NVT Langevin BAOAB.
- Heterogeneous atom counts, cells, PBC flags, and per-system MD parameters.
- Autograd or direct forces, E0 offsets, and strain-gradient stress evaluation.
- `FixAtoms` support.
- Neighbour-list skins and rebuild accounting.
- Exact ordered CUDA neighbour construction with adaptive CPU fallback.
- extxyz trajectories, JSONL diagnostics, tensor checkpoints, and summary JSON.
- YAML-driven CLI, deterministic toy models, tests, benchmarks, and an agent protocol.

## Repository layout

```text
batch_mlip/             Canonical public package
atombit_batch/          Thin compatibility namespace for the former package name
  core/                 Batch state, calculator contract, types, neighbors
  optimization/         FIRE, BFGS, cell filters, optimizer registry
  dynamics/             Molecular-dynamics integrators
  models/               MLIP adapter, loaders, reference models
  interfaces/           Python API, CLI/configuration, reporting
  profiling/            Opt-in phase timing and runtime event collection
  planning/             Memory calibration and heterogeneous workload bucketing
  workloads/            Signed workload identities and task descriptors
src/                    Uploaded AtomBit code in checkpoint-compatible namespace
original_uploads/       Immutable source snapshots
configs/                Runnable YAML configurations
examples/               Python API and checkpoint-loader examples
data/                   Small demo extxyz batch
benchmarks/              Scaling and profiling scripts
experiments/             Reproducible experiment specifications
research/                Imported research protocols and active-baseline notes
runs/                    Generated outputs; ignored by Git
tests/                   Correctness and regression tests
docs/                    Architecture, validation, and roadmap
AGENTS.md                Rules for autonomous experimental agents
```

New code should import from `batch_mlip`. Flat paths such as
`batch_mlip.filters` remain available, and the former `atombit_batch` package
name forwards to the same implementations for scripts, configs, and serialized
models created before version 0.2.

## Installation

Use a dedicated environment. Install the PyTorch build appropriate for your CPU/CUDA platform first when necessary, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional faster CPU neighbour lists:

```bash
python -m pip install -e '.[performance]'
```

## Verify the packet

```bash
pytest -q
batch-mlip validate configs/relax_toy.yaml
```

The included tests check:

- batch versus single-system energies and forces;
- absence of cross-system edges;
- neighbour-list skin behavior;
- per-system FIRE convergence;
- `FixAtoms` behavior;
- NVE energy drift;
- per-system Langevin parameters;
- the uploaded `src.model.AtomBitModel` running with `num_graphs > 1`;
- the YAML CLI.

## Run the included examples

```bash
batch-mlip run configs/relax_toy.yaml
batch-mlip run configs/nve_toy.yaml
batch-mlip run configs/nvt_toy.yaml
```

Outputs are written under `runs/`, including final structures, trajectories, diagnostics, and summaries.

The direct Python API is demonstrated by:

```bash
python examples/python_api.py
```

## Use a serialized complete model

A YAML model factory can return any `torch.nn.Module` that accepts the generic graph fields. For a checkpoint containing the complete module:

```yaml
model:
  factory: examples.atombit_loader:load_pickled_model
  kwargs:
    checkpoint: checkpoints/model.pt
    key: model
  cutoff: 6.0
  force_mode: autograd
```

## Use an AtomBit state dictionary

Edit `configs/atombit_model_example.yaml` so it exactly matches the trained architecture, then edit `configs/relax_atombit_template.yaml`:

```yaml
model:
  factory: examples.atombit_loader:load_atombit_state_dict
  kwargs:
    checkpoint: checkpoints/model.pt
    model_config: configs/atombit_model.yaml
    state_dict_key: state_dict
    strict: true
  cutoff: 6.0
  force_mode: autograd
```

Run:

```bash
batch-mlip validate configs/relax_atombit_template.yaml
batch-mlip run configs/relax_atombit_template.yaml
```

Validation should precede long optimization or MD runs.

## Model interface

The engine sends an attribute container with:

| Field | Shape | Meaning |
|---|---:|---|
| `z` | `[N]` | Atomic numbers |
| `pos` | `[N, 3]` | Concatenated positions in Å |
| `cell` | `[B, 3, 3]` | One row-vector cell per graph |
| `edge_index` | `[2, E]` | Directed local edges |
| `shifts_int` | `[E, 3]` | Integer periodic-image shifts |
| `batch` | `[N]` | Atom-to-graph map |
| `num_graphs` | scalar | Number of systems |

The model must return one total energy per graph, as `[B]`, `[B, 1]`, or a dictionary containing `energy`. A dictionary may also contain direct forces under `force` or `forces` with shape `[N, 3]`.

## Calculator-style Python API

The public structure-level API uses one model-independent calculator for
single-point evaluation, relaxation, and MD:

```python
import torch
from ase.io import read

from batch_mlip import (
    AtomBitBatchCalculator,
    FrechetCellFilter,
    evaluate,
    molecular_dynamics,
    relax,
)

systems = read("structures.extxyz", index=":")
calculator = AtomBitBatchCalculator(
    model,
    cutoff=6.0,
    skin=0.5,
    device="cuda",
    dtype=torch.float32,
    force_mode="autograd",
    e0_dict=e0_dict,
    neighbor_backend="auto",  # auto | matscipy | cuda_dense
)

single_points = evaluate(systems, calculator)
relaxed = relax(
    systems,
    calculator,
    fmax=0.03,
    max_steps=1000,
    active_compaction=True,
)
cell_relaxed = relax(
    systems,
    calculator,
    cell_filter=FrechetCellFilter(pressure_GPa=0.0),
    fmax=0.03,
    smax=0.0006,
    max_steps=1000,
)
trajectory_end = molecular_dynamics(
    relaxed.structures,
    calculator,
    ensemble="nve",
    timestep_fs=0.5,
    n_steps=100,
)
```

Each result exposes `.structures`, an input-ordered list of ASE `Atoms` with a
`SinglePointCalculator` containing the final energy and forces. Integrators use
only the `BatchCalculator` contract; model-specific graph and output conversion
belongs in calculator adapters.

Internal phase timing is opt-in and does not change calculator or optimizer
signatures:

```python
from batch_mlip import RuntimeProfiler, relax

with RuntimeProfiler(device=calculator.device) as profiler:
    result = relax(
        systems,
        calculator,
        optimizer="bfgs",
        refill_batch_size=64,
    )

profile = profiler.summary()
print(profile["phases"])
```

Controlled experiments use signed workload manifests rather than ad hoc filename
lists. Load and verify a manifest, derive model-specific task costs, and project
runtime profiling into the common registry schema as follows:

```python
from batch_mlip import TaskProfile
from batch_mlip.profiling import RunTelemetry, runtime_profile_registry_fields
from batch_mlip.workloads import read_workload_manifest, topology_key

manifest = read_workload_manifest(
    "benchmarks/workloads/manifests/OPT-H276-R256-v1.json"
)
task = TaskProfile.from_manifest(
    manifest,
    active_edge_key=topology_key(6.0, 0.0),
    candidate_edge_key=topology_key(6.0, 0.5),
)
timings = runtime_profile_registry_fields(profile)
telemetry = RunTelemetry.create(
    run_id="example-001",
    study_id="skin-calibration",
    workload_id=manifest.workload_id,
    workload_manifest_sha256=manifest.manifest_sha256,
    model_name="AtomBit",
    code_commit="<git-commit>",
    algorithm="bfgs",
    cell_mode=manifest.cell_mode,
    gpu_count=1,
    worker_mode="single-process",
    cold_or_warm="warm",
    repeat_index=0,
    equivalence_tier="K2",
    validation_pass=True,
    **timings,
)
```

The frozen suite and its model-specific profiles are indexed by
`benchmarks/workloads/index.json`. Regenerate and validate it with
`PYTHONPATH=. python tools/generate_controlled_workloads.py` and
`PYTHONPATH=. python tools/validate_controlled_workloads.py`.

Static force evaluation and fixed-horizon NVE workloads use the same signed
manifest runner for every native `BatchCalculator`:

```bash
batch-mlip-workload configs/run_controlled_workload_template.yaml
```

The YAML selects a calculator factory, model options, resident batch size, and
output paths without changing task definitions. Each run writes input-ordered
final structures, runtime phase data, telemetry, and a concise summary.
`wall_time_s` and throughput cover only the synchronized measured region;
`end_to_end_time_s` also includes verified dataset loading and model/physical
warm-up, but not calculator construction or output serialization. Peak allocated
and reserved GPU memory cover the measured region.

CUDA events are resolved once when the context exits. The variable-cell
benchmark scripts accept `--profile-runtime` and store the full phase samples
and scheduler events in their JSON point results.

An ordinary ASE calculator can be used as a correctness/reference fallback:

```python
from batch_mlip import ASECalculatorAdapter, relax

calculator = ASECalculatorAdapter(existing_ase_calculator)
result = relax(systems, calculator, fmax=0.03)
```

`ASECalculatorAdapter` evaluates structures sequentially. It makes existing
ASE MLIPs functionally compatible, but true acceleration requires a native
batch adapter for that MLIP.

MACE models use the optional native adapter rather than the sequential ASE
fallback:

```python
import torch

from batch_mlip import MACEBatchCalculator, relax

calculator = MACEBatchCalculator.from_off(
    model="small",
    device="cuda:0",
    dtype=torch.float64,
    graph_mode="cached",
    skin=0.5,
)
result = relax(systems, calculator, optimizer="bfgs", fmax=0.03)
```

Install the `mace` optional dependency or use an environment containing
`mace-torch`. MACE-OFF checkpoints use the Academic Software License and do
not permit commercial use. `graph_mode="cached"` projects the persistent common
tensor state directly into MACE and filters a skin candidate graph to the exact
model cutoff before every forward. `graph_mode="rebuild"` is the default and
retains MACE `AtomicData` construction for compatibility. Both modes use MACE's
direct forces, stress convention, cutoff, element table, and heads.

The opt-in integration suite runs the fixed T2 structures through common ASE,
masked batching, and active batching with both FIRE and BFGS:

```bash
python -m pip install -e '.[mace,dev]'
make test-mace
```

This test requires CUDA, the MACE-OFF-Small checkpoint, and the extracted
`data/T2_test/structures` dataset. The ordinary test suite skips it because
MACE is an optional dependency. If pytest and MACE are in separate compatible
environments, pass the MACE site-packages directory explicitly:

```bash
make test-mace PYTHON=/path/to/pytest/python \
  MACE_SITE_PACKAGES=/path/to/mace/environment/lib/python3.11/site-packages
```

The reproducible B1-B32 ASE/masked/active optimization benchmark is implemented
in `benchmarks/benchmark_mace_variable_cell_scaling.py`; its fixed-pool results
are recorded under `experiments/mace-variable-cell-scaling-32/`.
The B64 cached-versus-rebuild BFGS experiment is recorded under
`experiments/mace-tensor-state-cache/`.

`cell_filter=None` is the default and preserves fixed-cell FIRE. Passing
`FrechetCellFilter` optimizes atomic positions and full-rank periodic
cells together using log-deformation coordinates. Pressure is specified in
GPa and is positive in compression; `smax` is in eV/Angstrom^3. Variable-cell
FIRE requires calculator stress. Active compaction removes converged graph and
cell optimizer state while preserving original output order.

## Extensible optimizer interface

`relax()` accepts either a registered optimizer name or a direct object that
implements the runtime-checkable `BatchOptimizer` protocol:

```python
from batch_mlip import BatchedFIRE, create_optimizer, relax

# Registered-name convenience path.
result = relax(systems, calculator, optimizer="fire", fmax=0.03)

# Configured optimizer object; call-time options override object defaults.
optimizer = BatchedFIRE(dt_start=0.05, dt_max=0.5)
result = relax(systems, calculator, optimizer=optimizer, fmax=0.03)

# Equivalent explicit factory construction.
optimizer = create_optimizer("fire", dt_start=0.05, dt_max=0.5)
```

A third-party batched optimizer declares its optional capabilities and returns
a `RelaxationResult` from `run`:

```python
from batch_mlip import OptimizerCapabilities, register_optimizer

class BatchedLBFGS:
    def capabilities(self):
        return OptimizerCapabilities(
            variable_cell=True,
            active_compaction=True,
        )

    def run(self, state, calculator, **options):
        # Implement batched LBFGS state, updates, compaction, and result here.
        ...

register_optimizer("lbfgs", BatchedLBFGS)
result = relax(systems, calculator, optimizer="lbfgs", fmax=0.03)
```

The built-in `BatchedFIRE` and `BatchedBFGS` support variable cells and active
compaction. Full BFGS stores an independent dense Hessian for every active
structure and follows ASE's update, eigensolve, and row-wise step clipping:

```python
from batch_mlip import BatchedBFGS

result = relax(
    systems,
    calculator,
    optimizer=BatchedBFGS(alpha=70.0, max_step=0.2),
    cell_filter=FrechetCellFilter(),
    active_compaction=True,
    fmax=0.05,
    smax=None,
)
```

For workloads larger than the desired GPU-resident batch, BFGS can refill
converged slots from a pending queue while preserving each survivor's Hessian
and Frechet state:

```python
result = relax(
    workload,
    calculator,
    optimizer="bfgs",
    cell_filter=FrechetCellFilter(),
    refill_batch_size=64,
    refill_policy="immediate",
    fmax=0.05,
    smax=None,
)
```

The step limit applies independently from the time each queued structure
enters. Finished Hessians are released, results retain workload order, and
neighbor graphs for pending structures are built only when those structures
enter the resident batch.

`linear_algebra_backend` accepts `"auto"`, `"serial"`, or `"grouped"`. The
automatic policy groups equal-sized CUDA Hessians only when `D <= 285`. H46 and
H92 reduce optimizer time by 17-33%; H184 misses the joint two-model gate and
larger Hessians use the serial path. Singleton groups always use the serial
path, which preserves B1 agreement with ASE.

ASE's line-search variant is available under either conventional name:

```python
result = relax(
    systems,
    calculator,
    optimizer="quasinewton",  # alias: bfgslinesearch
    cell_filter=FrechetCellFilter(),
    active_compaction=True,
    fmax=0.05,
)
```

`quasinewton` and `bfgslinesearch` construct the same optimizer, matching ASE's
class alias. Each structure owns an inverse Hessian and independent strong-Wolfe
state, while simultaneous trial requests share a model batch. One accepted
optimizer step can require multiple model evaluations. Active refill is not yet
supported for this optimizer and is rejected by the capability interface.

`refill_policy` accepts `"drain"`, `"immediate"`, or `"threshold"`.
Immediate is the measured default. Threshold refill also accepts
`refill_low_watermark` and `refill_min_chunk`, but it is workload-dependent and
did not beat immediate refill by the project performance gate.

For heterogeneous workloads, `BatchPlanner` provides calibrated memory-safe
queues without coupling planning to a particular MLIP or optimizer:

```python
from batch_mlip import BatchPlanner

planner = BatchPlanner(
    coefficients,
    memory_budget_bytes=32 * 1024**3,
    max_batch_size=128,
    max_cost_ratio=2.0,
)
plan = planner.plan(
    structures,
    cutoff=calculator.cutoff,
    skin=calculator.skin,
)
```

Each planned bucket reports original system indices, a resident capacity, and a
predicted peak. Calibration uses `fit_memory_coefficients` with measured batch
peaks. Planning is an explicit safety tool; it is not automatically applied by
`relax` because workload-dependent speedups did not pass the project 5% gate.

The BFGS Hessian costs `O(D^2)` memory and its eigensolve costs `O(D^3)` for
`D = 3N` fixed-cell or `D = 3N + 9` variable-cell degrees of freedom. It is a
strong ASE-compatible optimizer for small and medium structures; LBFGS remains
the scalable follow-up for large systems.

`BatchedGradientDescent` is a fixed-cell reference and rejects those options.
Registering an optimizer does not adapt an ordinary ASE optimizer
automatically: ASE classes operate on one `Atoms`/filter state and require a
dedicated batched implementation to retain acceleration.

## Low-level Python API

```python
import torch
from ase.io import read, write

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator, batched_fire_relax

systems = read("structures.extxyz", index=":")
state = AseGraphBatch.from_ase(
    systems,
    cutoff=6.0,
    skin=0.5,
    device="cuda",
    dtype=torch.float32,
)
potential = AtomBitBatchCalculator(
    model,
    device="cuda",
    dtype=torch.float32,
    force_mode="autograd",
    e0_dict=e0_dict,
)
result = batched_fire_relax(
    state,
    potential,
    fmax=0.03,
    max_steps=1000,
)
write("relaxed.extxyz", result.state.to_ase(result.evaluation, wrap=True))
```

## Force modes

- `autograd`: differentiate the graph energies. This is the default and the preferred starting point for NVE dynamics.
- `direct`: use the model's direct force head. This is faster when the head exists, but it may not be exactly conservative.
- `auto`: use direct forces when returned, otherwise autograd.

Do not add E0 both inside the model and in `AtomBitBatchCalculator`; choose one location.

## Neighbour-list policy

With `skin: 0`, the list is rebuilt every force evaluation. With a positive skin, edges are built to `cutoff + skin` and rebuilt after any atom moves more than `skin / 2` from the reference positions. The supplied AtomBit envelope becomes zero at the physical cutoff, so extra skin edges do not contribute.

The baseline builder runs on CPU through matscipy when installed, otherwise ASE. GPU-native PBC cell lists are a priority experiment rather than an unverified default.

## Current scientific scope

Implemented:

- independent fixed-cell and optional Frechet variable-cell systems;
- heterogeneous sizes and cells;
- FIRE, full BFGS, and gradient descent;
- NVE/NVT fixed-cell MD;
- `FixAtoms` for fixed-cell optimization and MD;
- per-system time steps, temperatures, friction, and FIRE parameters;
- finite-difference-validated strain-gradient stress calculation.

Not yet implemented:

- NPT dynamics (the public ensemble slot is reserved but raises explicitly);
- SHAKE/RATTLE or general ASE constraints;
- GPU-native periodic neighbour lists;
- multi-GPU sharding.

These are tracked in `docs/roadmap.md` and designed as controlled experiments rather than hidden behavior.

## Autonomous experimentation

Agents should read `AGENTS.md` before modifying the code. The required loop is:

1. Establish a tested baseline.
2. Register one falsifiable hypothesis.
3. Change one primary variable.
4. Run correctness tests before benchmarks.
5. Store commands, environment, metrics, and artifacts.
6. Compare against the baseline and record failures as well as wins.

Use:

```bash
python tools/run_experiment.py experiments/baseline/experiment.yaml
python tools/compare_runs.py runs/experiments/<run-a>/manifest.json runs/experiments/<run-b>/manifest.json
```

## Provenance and licensing

The exact uploaded files are preserved under `original_uploads/`. No license was supplied for them; see `NOTICE.md` before redistribution.
