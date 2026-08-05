# Batch MLIP

A model-independent interface for **true batched geometry optimization and molecular dynamics** with graph MLIPs. Native adapters are included for AtomBit-style models and MACE.

The engine concatenates independent atomic systems into one heterogeneous graph batch and performs one model forward pass per simulation step. ASE is used at the boundary for structure I/O. Adaptive matscipy/ASE CPU, dense CUDA, and sparse CUDA cell-list backends construct neighbour lists, while PyTorch owns batched model evaluation, optimizer state, and MD integration.

## What is included

- A reusable package under `batch_mlip/`.
- Compatibility copies of the uploaded model under the original `src.*` namespace.
- Exact source snapshots under `original_uploads/`.
- Fixed/variable-cell batched FIRE and full BFGS, plus steepest descent.
- Fixed-cell NVE velocity-Verlet and NVT Langevin BAOAB, plus isotropic MTK NPT.
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
research/                Authoritative project map and historical protocols
runs/                    Generated outputs; ignored by Git
tests/                   Correctness and regression tests
docs/                    Architecture, validation, and roadmap
AGENTS.md                Rules for autonomous experimental agents
```

## Project logic

The authoritative research framing is
[research/project/README.md](research/project/README.md). It separates the
validated execution foundation, mechanism evidence, frozen scheduler-v1
baseline, and the next chemical-transfer study. The accompanying evidence
registry classifies every experiment directory and prevents historical timings
collected under incompatible contracts from becoming planner-training labels.

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
- ASE-parity, restart, drift, and batching checks for isotropic MTK NPT;
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

## Optimize a structure pool automatically

`optimize_pool()` is the production entry point for an in-memory pool. The
caller supplies structures, a native batch calculator, and devices; the
function performs workload profiling, cost bucketing, resident-batch planning,
multi-GPU assignment, active compaction/drain, result reassembly, and worker
shutdown:

```python
import torch
from ase.io import read

from batch_mlip import (
    MACEBatchCalculator,
    ReproducibilityConfig,
    configure_reproducibility,
    optimize_pool,
)


def main():
    # Install the frozen execution contract before MACE initializes CUDA.
    configure_reproducibility(ReproducibilityConfig())
    structures = read("candidates.extxyz", index=":")
    calculator = MACEBatchCalculator.from_off(
        model="small",
        device="cuda:0",
        dtype=torch.float64,
        graph_mode="cached",
        skin=0.5,
    )
    result = optimize_pool(
        structures,
        calculator,
        optimizer="bfgs",
        cell_filter="frechet",
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        policy="auto",
        fmax=0.01,
        max_steps=3000,
    )
    print(result.structures)
    print(result.converged)
    print(result.schedule)
    print(result.metadata["optimize_pool"]["capacity_planning"])


if __name__ == "__main__":
    main()
```

The frozen AtomBit OMC-CSP path uses the same entry point and loads the
training checkpoint without a benchmark-specific helper:

```python
from batch_mlip import AtomBitBatchCalculator

calculator = AtomBitBatchCalculator.from_checkpoint(
    "AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt",
    e0="meta_e0_data_OMC_r6_single.pt",
    device="cuda:0",
    dtype=torch.float32,
    cutoff=6.0,
    skin=0.5,
    force_mode="autograd",
    neighbor_backend="auto",
)
result = optimize_pool(
    structures,
    calculator,
    optimizer="bfgs",
    cell_filter="frechet",
    devices=["cuda:0", "cuda:1"],
    policy="auto",
    fmax=0.01,
    max_steps=3000,
)
```

`examples/optimize_atombit_omc_csp.py` is the complete file-backed command-line
example. The exact smooth-RMS fp32 checkpoint, float64 BFGS state, 6.0 A
cutoff, 0.5 A skin, Frechet cell filter, H100, and allocator contract select
the packaged AtomBit capacity policy; a mismatch uses the recorded probe
fallback instead.

`policy="auto"` uses a packaged capacity model only when the checkpoint,
adapter options, optimizer, cell filter, precision, allocator, PyTorch/CUDA
versions, GPU model, and memory budget match its signed contract. Otherwise it
automatically performs a representative memory probe and records the mismatch
reason. Use `policy="probe"` to request that conservative path explicitly, or
pass a signed `HardwareCapacityPolicy`/JSON path. `cell_filter=None` selects
fixed-cell optimization. Because GPU workers use Python's spawn method, invoke
the one-shot interface from a file-backed `__main__`; retain `BatchExecutor`
directly when processing several pools with the same calculator and devices.

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
    neighbor_backend="auto",  # auto | matscipy | cuda_dense | cuda_cell
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
npt_end = molecular_dynamics(
    relaxed.structures,
    calculator,
    ensemble="npt_mtk",
    timestep_fs=0.5,
    n_steps=100,
    temperature_K=300.0,
    pressure_eV_per_A3=0.0,
    thermostat_damping_fs=50.0,
    barostat_damping_fs=500.0,
)
```

Each result exposes `.structures`, an input-ordered list of ASE `Atoms` with a
`SinglePointCalculator` containing the final energy and forces. Integrators use
only the `BatchCalculator` contract; model-specific graph and output conversion
belongs in calculator adapters.

For repeated independent relaxation pools, keep one initialized calculator
process per GPU with `BatchExecutor`:

```python
from batch_mlip import AutoSchedulerConfig, BatchExecutor

config = AutoSchedulerConfig(
    memory_safety_fraction=0.85,
    max_batch_size=256,
)
with BatchExecutor(
    calculator,
    devices=["cuda:0", "cuda:1"],
    auto_config=config,
) as executor:
    first = executor.relax(
        first_pool,
        optimizer="bfgs",
        cell_filter=FrechetCellFilter(),
        fmax=0.03,
        max_steps=500,
    )
    second = executor.relax(
        second_pool,
        optimizer="bfgs",
        cell_filter=FrechetCellFilter(),
        fmax=0.03,
        max_steps=500,
    )

    crystal_result = executor.relax_manifest(
        crystal_manifest,
        "/path/to/cif/root",
        crystal_planning_profile,
        optimizer="bfgs",
        cell_filter=FrechetCellFilter(),
        fmax=0.03,
        max_steps=500,
    )
```

Each call profiles atoms and candidate edges, performs one representative model
forward, and packs deterministic chunks to at most 85% of device memory. It
does not run trial optimizations or require a timing-policy cache. The first
call starts and warms the worker generation; compatible later calls reuse the
same worker PIDs and model instances. A change that requires an incompatible
CUDA allocator policy closes that generation and starts a new one. Leaving the
context releases all worker processes and GPU reservations.

`relax_manifest` on the executor retains the same GPU workers and model
instances across compatible signed pools. Its parent-owned CPU loader keeps a
global cost-ordered buffer of at most one running plus one prefetched chunk per
active worker. Prefetched chunks remain unassigned until a GPU finishes, so
pending work stealing is preserved. A matching signed capacity policy removes
the representative forward; a mismatch retains the bounded probe fallback.
The CPU loader pool is also reused while its selected process count remains
compatible.

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

The built-in `BatchedFIRE` and `BatchedBFGS` support variable cells, active
compaction, and bounded active refill. Full BFGS stores an independent dense
Hessian for every active structure and follows ASE's update, eigensolve, and
row-wise step clipping:

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

For workloads larger than the desired GPU-resident batch, FIRE or BFGS can
refill converged slots from a pending queue. FIRE preserves velocity, time
step, mixing, positive-power counter, and Frechet state; BFGS preserves each
survivor's Hessian and Frechet state:

```python
result = relax(
    workload,
    calculator,
    optimizer="fire",  # "bfgs" uses the same scheduler controls
    cell_filter=FrechetCellFilter(),
    refill_batch_size=64,
    refill_policy="immediate",
    refill_storage="slots",
    refill_interval=1,
    fmax=0.05,
    smax=None,
)
```

The step limit applies independently from the time each queued structure
enters. Finished optimizer state is released, results retain workload order,
and neighbor graphs for pending structures are built only when those
structures enter the resident batch.

`refill_storage="slots"` overwrites completed equal-atom-count resident slots
and falls back to repacking for unequal sizes or an unfillable tail. It is most
useful after size/edge bucketing. On supported CUDA/PyTorch builds, long refill
runs should be launched with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid measured allocator
fragmentation. The default remains `"repack"` pending broader application
validation.

`refill_interval` optionally accumulates finished, immutable slots before
physically replacing or repacking them. Its default is `1` (immediate refill).
Values above one are experimental: they reduce scheduler mutations but spend
extra model work on frozen graphs, and the measured H46 matrix did not justify
automatic selection.

`refill_tail_compaction_threshold` is a separate experimental control. It
keeps immediate admission while pending jobs exist, then compacts the
queue-empty tail geometrically at the requested occupancy fraction. The
measured 50% and 75% thresholds did not produce a reproducible speedup, so the
default remains `None`.

The automatic process launcher selects expandable CUDA allocator segments for
AtomBit variable-cell FIRE and BFGS with changing graph sizes:

```python
from batch_mlip import AutoSchedulerConfig, FrechetCellFilter, relax

result = relax(
    structures,
    calculator,
    optimizer="fire",  # BFGS uses the same measured allocator rule
    scheduling="auto",
    devices=["cuda:0"],  # add more devices for one process per GPU
    auto_config=AutoSchedulerConfig(),
    cell_filter=FrechetCellFilter(),
    fmax=0.05,
)
print(result.metadata["scheduling"]["allocator"])
```

The launcher installs the allocator before each spawned worker initializes
CUDA and reports the selected policy, reason, effective environment, and
reported backend. It sets both compatibility variables because the validated
PyTorch 2.9.1 environment responds only to the deprecated spelling. MACE,
fixed-cell optimization, and optimizers without matched evidence remain on the
native allocator. Override the conservative rule with
`AutoSchedulerConfig(cuda_allocator_policy="native" | "expandable_segments")`.

Calls without `devices=[...]` run in the current Python process and cannot
safely change an allocator after CUDA initialization. For those calls, set
both variables before Python starts:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python your_relaxation.py
```

Matched variable-cell FIRE diagnostics reduced peak reserve from
`77.4-78.2 GiB` to `28.9-35.3 GiB` at B64/B128 on BOQWIN116, XAFPAY172, and
ROFB296 without a throughput regression. Larger safe points reached B256 for
BOQWIN/XAFPAY and B128 for ROFB. MACE did not benefit in the prior matched
screening.

BFGS also accepts the experimental `refill_storage="arena"` mode for
heterogeneous residents. It alternates between two reusable compact graph
stores, preserving per-job Hessians and neighbor-cache state without padded
model inputs. On the measured 256-job MIX4 B64 workload it reduced refill
packing by 11.6% and total time by 0.8%, which is below the automatic-selection
gate for AtomBit. On matched MACE-OFF-Small, arena packing was 15.8% slower;
the 1.1% lower wall time came from variation in graph/model phases. FIRE does
not accept this mode, and the planner never selects it.

`linear_algebra_backend` accepts `"auto"`, `"serial"`, `"grouped"`, or
`"cholesky"`. On CUDA, the automatic policy groups equal-sized Hessians and
uses a Cholesky solve while they are positive definite. That solve is
mathematically equivalent to ASE's eigenbasis expression; any failed
factorization falls back to ASE's absolute-eigenvalue solve for only the
affected systems. `"grouped"` keeps the eigen-only batched implementation and
`"serial"` keeps the per-system ASE-compatible implementation.

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
Immediate is the implementation default when refill is explicitly requested.
Threshold refill also accepts
`refill_low_watermark` and `refill_min_chunk`, but it is workload-dependent and
did not beat immediate refill by the project performance gate. The current
STEPVAR MPS comparison found both refill policies numerically slower than
active drain, so refill is not a universal throughput default.

Normal users do not need to choose a batch size or construct a planner:

```python
result = relax(
    structures,
    calculator,
    optimizer="bfgs",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],  # omit for one GPU
    cell_filter=FrechetCellFilter(),
    fmax=0.05,
    smax=None,
)
print(result.schedule)
```

For a large, signed file-backed pool, use the corresponding source-backed
entry point instead of loading every CIF into the parent process:

```python
from batch_mlip import read_planning_profile, relax_manifest
from batch_mlip.workloads import read_workload_manifest

manifest = read_workload_manifest("workload.json")
profile = read_planning_profile("planning-profile.json")
result = relax_manifest(
    manifest,
    "/path/to/cif/root",
    profile,
    calculator,
    optimizer="bfgs",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    cell_filter=FrechetCellFilter(),
    fmax=0.05,
    smax=None,  # use ASE FrechetCellFilter convergence semantics
)
```

`relax_manifest` verifies the manifest/profile binding, plans without a timing
pilot, and keeps source loading bounded. One-GPU calls materialize each planned
resident batch directly without an outer worker process. Multi-GPU CUDA calls
use a global prefetch queue so unassigned CIF chunks can load while the current
GPU wave runs, then dispatch complete cost-balanced chunks to isolated CUDA
workers. Results retain manifest order, and the returned object is the same
`OptimizationResult` used by `relax`. This path is the accepted default for
signed large-pool OMC-CSP workloads; ordinary in-memory use remains unchanged.

Manifest loading also has a bounded CPU-process policy. The default
`manifest_loader_processes="auto"` uses one process below the validated medium
gate, two at pool size at least 512 with at least 3,000 atom-records per active
GPU, and four at pool size at least 2,048 with at least 32,000 atom-records per
active GPU, subject to available host CPUs. This preserves ordering, avoids
forking an initialized CUDA process, and requires no timing pilot. Pass a
positive integer to override the offline decision.

For consecutive pools, prefer `BatchExecutor.relax_manifest`. The ordinary
entry point closes its workers after one pool; the executor retains them across
pools while using the same global prefetch queue. Set
`manifest_prefetch_chunks_per_worker=0` to retain persistence while disabling
the overlap buffer.

For the exact packaged AtomBit/H100 variable-cell BFGS contract,
`relax_manifest` also selects a signed offline reserved-memory model and
performs zero representative model probes. The model is matched against the
checkpoint state, calculator, optimizer, cell filter, graph policy, allocator,
software, hardware, and planning sidecar. Any mismatch automatically restores
the representative-probe capacity path.

Automatic scheduling is the ordinary default when the calculator exposes a
cutoff. The user selects structures, calculator, optimizer, and optional
devices; the runtime owns the remaining decisions:

| Step | Automatic decision |
|---|---|
| 1 | Preserve structure atoms, then bind MLIP active edges, task auxiliary state, and graph-policy candidate edges separately. |
| 2 | Preserve compatible outer buckets, then use an exact-contract signed byte model or the safe probe fallback to pack each GPU to the 85% memory budget. |
| 3 | Always remove converged structures from later model evaluations. |
| 4 | Select refill only from scientifically accepted matching evidence; otherwise drain safely. |
| 5 | On multiple GPUs, balance memory-safe chunks through work stealing. |

No timing sweep or trial relaxation is performed. The compact `summary`
reports only the selected batch mode, devices, resident capacities, memory
fraction, compaction, work stealing, and any refill fallback reason. Detailed
profiling and evidence records remain available in the surrounding
`metadata["scheduling"]` mapping.

Use `scheduling="single_batch"` only for an explicitly unmanaged batch.
Supplying manual refill controls also preserves this expert path for backward
compatibility. `scheduling="autotune"` and explicit `BatchPlanner` inputs are
advanced experimental interfaces, not competing production defaults.

The runtime records atoms and active MLIP edges as the base compute interface.
It separately records cutoff/force mode, task auxiliary state, stress/cell
mode, skin/cache candidate edges, and hardware. For variable-cell BFGS,
`D=3N+9`, dense state is `D²`, and dense eigensolver work is represented by
`D³`. A scalar estimate exists only after coefficients have been calibrated
for that exact model/task/policy/hardware contract.

When no exact signed capacity policy matches, the runtime performs one
representative model forward containing up to four of the largest structures.
It combines the measured model/graph peak with the explicit dense-BFGS
allowance and packs largest-cost-first chunks within 85% of available GPU
memory.

For the packaged AtomBit/H100 manifest contract, a signed peak-reserved-byte
model replaces that forward. It charges the fixed model/allocator term once
per resident batch and the calibrated active- and candidate-edge terms per
system, then applies the same 1.10 optimization-growth margin used by the
probe path. The task profiler's outer buckets are retained; the hardware model
only chooses inner resident chunks. Neither path runs an optimization pilot,
relaxes a structure twice, or performs a batch-size timing sweep.

Automatic optimization always uses active compaction. It normally uses active
drain, but BFGS can select immediate slot refill from a packaged offline policy
when the calculator, model state, precision, optimizer, cell filter, hardware,
allocator, pool size, resident capacity, and workload descriptors match
contract-identical evidence. No timing pilot is run. Unmatched cases and
records that failed the speed, memory, convergence, or endpoint gates use
active drain. Every productive chunk reports the refill decision, predicted
peak, actual allocated peak, actual reserved peak, and resident count in
`result.metadata["scheduling"]`.

For a mixed outer bucket, the scheduler may extract an exact optimizer-shape
subgroup only when that subgroup spans more than one memory-safe resident wave
and independently matches accepted refill evidence. Variable-cell full BFGS
uses `(atom_count, (3N + 9)^2)` as this compatibility key. Accepted subgroups
retain submitted order and use fixed slots; unmatched structures remain in
their original active-drain chunks. Thus exact-shape extraction cannot
fragment a mixed bucket unless refill has a predicted payback, and it never
requires a timing pilot or trial relaxation.

The packaged refill policy currently applies to current-process single-GPU
automatic scheduling. Multi-GPU automatic execution continues to shard
memory-safe active-drain chunks; per-worker refill is not inferred from
single-GPU pool evidence.

Refill policy v2 combines the original nine-family R256 matrix with a separate
pool-transfer matrix at R128 and R512. The locked single-GPU transfer rule
predicted all four held-out speed outcomes (`1.121-1.433x`) with no convergence,
memory, or endpoint failures. Selection still requires an exact measured pool
size and resident capacity; it does not interpolate.

Per-worker refill was also tested under static G2/G4 sharding. Its GPU-count
rule predicted only one of two held-out speed outcomes, and the selected G4
BOQWIN point exceeded the endpoint gate. Multi-GPU automatic execution
therefore remains active drain. Historical repeated-structure controls,
multi-GPU transfer failures, and results from different execution contracts
are not used for refill selection.

`AutoSchedulerConfig` exposes the 0.85 memory fraction, safety margin, offline
capacity-policy enable flag, probe size, and absolute-budget test override.
Multiple homogeneous GPUs share the same plan. The first dispatch wave is
bucket-stratified so one expensive bucket cannot occupy every GPU; remaining
memory-safe chunks are pulled in descending predicted-cost order. Active
optimizer states never migrate between GPUs. In-memory automatic execution
uses threads for short queues. When at least eight
pending chunks per active device can amortize spawn startup, it uses one
isolated persistent process per GPU; each process keeps its calculator and
optimizer alive while pulling later chunks. Override this conservative rule with
`AutoSchedulerConfig(multi_gpu_worker_backend="process" | "thread")`.
Non-serializable custom adapters fall back to threads during preflight, before
any production job starts. Signed multi-GPU manifest workloads instead use
the bounded global-prefetch process executor directly; in-process MPS dispatch
remains a future execution layer.

The former production-learning controller is retained only for controlled
experiments as `scheduling="autotune"`. That explicit mode grows capacities
from completed production chunks and stores compatible decisions in
`~/.cache/batch_mlip/autoscheduler-v1.json`; it is not used by
`scheduling="auto"`.

The earlier nearest-workload throughput table remains rejected. Its static
descriptor rule passed only 6 of 12 cases on the second independent holdout,
so the runtime still does not infer a throughput knee from a chemically
similar family. The packaged policy is different: it is a continuous
reserved-memory regression over signed layered graph features, makes no
throughput claim, and is enabled only by exact execution-contract matching.
On three non-fit P2048 families it performed zero probes, satisfied every
predicted chunk bound, used at most 63.58% of H100 memory, and ran 1.036x
faster than the probe-backed path and 2.700x faster than frozen MPS.

Capacity safety passed, but changing resident chunk composition produced six
sparse endpoint-tolerance exceptions among 6,144 jobs relative to the
probe-backed tensor schedule. All jobs converged. Capacity acceptance and
strict schedule-invariant endpoint reproducibility are therefore reported
separately in
[`experiments/omc-csp-scheduler-epoch3`](experiments/omc-csp-scheduler-epoch3).

For frozen experiments, `BatchPlanner` still provides explicitly calibrated
memory-safe queues without coupling planning to a particular MLIP or optimizer:

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
peaks. Apply the plan through the same calculator-style relaxation interface:

```python
result = relax(
    structures,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    planner=planner,
    cell_filter=FrechetCellFilter(),
    active_compaction=True,
    fmax=0.05,
)
print(result.metadata["scheduling"])
```

The explicit planner uses the whole pool when its calibrated allocation is
within both the byte budget and maximum resident count. Otherwise, it executes
cost-compatible memory-safe queues, using active refill only when the selected
optimizer supports it.

Supplying an `OptimizationPilot` activates the task-aware layer:

```python
import json

from batch_mlip import OptimizationPilot, TaskAwarePolicy

pilot = OptimizationPilot.from_dict(
    json.load(open("optimizer-pilot.json", encoding="utf-8"))
)
result = relax(
    structures,
    calculator,
    optimizer="bfgs",
    scheduling="auto",
    planner=planner,
    pilot=pilot,
    policy=TaskAwarePolicy(),
    system_profiles=cached_profiles,
    cell_filter=FrechetCellFilter(),
)
```

It compares measured drain/refill capacities and records a tensor-versus-MPS
recommendation. `relax` executes the tensor fallback; an MPS recommendation
requires an external worker dispatcher. Refill is not extrapolated to an
unmatched atom/edge regime by default. Cached `SystemProfile` values avoid
repeating CPU topology profiling, and completed buckets are offloaded before
the next bucket. The allocated-memory planner still requires a separate
reserved-memory safety measurement near device capacity.

The measured task/mechanism policy, direct CUDA MPS comparison, and untested
acceleration backlog are recorded in
`experiments/application-mechanism-atlas/README.md`.
The held-out task-aware policy validation, including negative results, is in
`experiments/task-aware-policy-validation/README.md`.

For a strict ordinary-ASE reference, pass the MLIP's native ASE calculator
explicitly. A `BatchCalculator` is not silently converted into an ASE
calculator because that would not preserve the native calculator path:

```python
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from batch_mlip import relax_ase

reference = relax_ase(
    structures,
    native_ase_calculator,
    optimizer="bfgs",
    cell_filter=ASEFrechetCellFilter,
    fmax=0.05,
    max_steps=500,
)
```

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

The baseline builder runs on CPU through matscipy when installed, otherwise
ASE. `cuda_dense` performs exact memory-bounded quadratic construction on CUDA.
`cuda_cell` uses exact float64 periodic spatial bins for full-rank 3D-periodic
cells and restores Matscipy-compatible edge ordering. `auto` first applies the
existing CPU/CUDA launch crossover, then selects the cell list only when cell
geometry and occupied fractional span predict at least 98% fewer candidates
than dense image expansion.
Partial/nonperiodic or singular cells retain the existing dense/CPU paths.

`neighbor_backend="nvalchemi"` is an optional explicit NVIDIA Warp control
available through the `nvidia` package extra. It reproduces canonical topology
but did not pass the end-to-end performance gate and is therefore never selected
by `auto`.

## Current scientific scope

Implemented:

- independent fixed-cell and optional Frechet variable-cell systems;
- heterogeneous sizes and cells;
- FIRE, full BFGS, BFGSLineSearch/QuasiNewton, and gradient descent;
- NVE/NVT fixed-cell MD and isotropic MTK NPT;
- `FixAtoms` for fixed-cell optimization and MD;
- per-system time steps, temperatures, friction, and FIRE parameters;
- finite-difference-validated strain-gradient stress calculation.
- exact Matscipy, CUDA dense-pair, and CUDA periodic cell-list neighbour
  construction with per-rebuild automatic backend selection;
- deterministic multi-GPU chunking, bucket-stratified initial dispatch,
  pending work stealing, and bounded global manifest prefetch;
- persistent model-owning GPU workers through `BatchExecutor`.

Not yet implemented:

- anisotropic or partially periodic NPT cell dynamics;
- SHAKE/RATTLE or general ASE constraints;
- a validated multi-GPU refill policy;
- automatic recovery/repacking after an unexpected production OOM.

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
