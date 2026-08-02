# Configuration Reference

Every run configuration uses `schema_version: 1`.

## Top level

```yaml
task: relax | nve | nvt_langevin | npt_mtk
input: path/to/input.extxyz
output: path/to/final.extxyz
runtime: {...}
model: {...}
relax: {...}  # relaxation only
md: {...}     # MD only
reporting: {...}
```

## Runtime

```yaml
runtime:
  device: cuda
  dtype: float32
  skin: 0.5
  neighbor_backend: auto  # auto | matscipy | cuda_dense
  reproducibility:
    seed: 20260729
    deterministic_algorithms: true
    deterministic_warn_only: false
    cublas_workspace_config: ":4096:8"
    cudnn_benchmark: false
    cudnn_deterministic: true
    allow_tf32: false
    cpu_threads: 1
    interop_threads: 1
```

`auto` uses the CPU backend for small rebuilds and the integrated dense CUDA
backend above a cutoff-aware work threshold. `matscipy` forces CPU construction
(with ASE fallback outside matscipy's validated fully periodic path), while
`cuda_dense` requires CUDA and raises for unsupported degenerate periodic cells.

The reproducibility block seeds Python, NumPy, and PyTorch and fixes the
supported deterministic library controls. For strict controlled experiments,
set `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` before starting Python; see
[`reproducibility.md`](reproducibility.md).

## Automatic capacity

The Python API uses `AutoSchedulerConfig` for automatic relaxation capacity:

```python
AutoSchedulerConfig(
    memory_safety_fraction=0.85,
    memory_growth_margin=1.10,
    offline_hardware_capacity_enabled=True,
    manifest_loader_processes="auto",
    manifest_prefetch_chunks_per_worker=1,
)
```

For a signed manifest and matching planning profile, the packaged offline
capacity policy is selected only when its complete model/task/graph/allocator/
software/H100 contract matches. It performs no representative model forward.
An unmatched or disabled policy uses the bounded representative-probe path.
Both paths retain the outer workload buckets and apply the same memory-growth
margin.

`optimize_pool(..., policy="auto")` applies the same logic to ordinary
in-memory ASE structures. It searches all packaged signed policies by exact
calculator and model-state identity, then validates the complete runtime
contract. It never borrows a capacity from a different MLIP, checkpoint,
optimizer, graph mode, CUDA stack, or GPU. Use `policy="probe"` to force the
fallback, or pass a signed policy object or JSON path explicitly.

For signed file-backed workloads, `manifest_loader_processes="auto"` selects
either one or four worker-local CIF parsing processes from pool size, total
atom-record pressure per active GPU, and available host CPUs. It does not run a
timing sweep. A positive integer is an explicit expert override. The process
pool uses `spawn` and is closed deterministically with its GPU worker.

`manifest_prefetch_chunks_per_worker=1` is used by
`BatchExecutor.relax_manifest`: at most one extra chunk per active worker is
materialized in a global host-side queue while GPUs execute the current wave.
The chunks are not assigned to a device until it becomes idle. Set the value to
zero to disable overlap without disabling persistent GPU workers.

## Model

```yaml
model:
  factory: package.module:function
  kwargs: {}
  cutoff: 6.0
  force_mode: autograd
  e0: path/to/e0.json
  call_kwargs: {}
```

The factory must return a `torch.nn.Module`. `cutoff` can be omitted when available as `model.cfg.cutoff` or `model.cutoff`.

## FIRE

```yaml
relax:
  optimizer: fire
  fmax: 0.03
  max_steps: 1000
  dt_start: 0.1
  dt_max: 1.0
  max_step: 0.2
  alpha_start: 0.1
  n_min: 5
  f_inc: 1.1
  f_dec: 0.5
  f_alpha: 0.99
  callback_interval: 10
```

## NVE

```yaml
md:
  timestep_fs: 0.5
  n_steps: 10000
  initialize_velocities: true
  initial_temperature_K: 300
  initialization_seed: 1234
  remove_initial_com: true
  force_exact_initial_temperature: true
  callback_interval: 10
```

## Langevin NVT

Add:

```yaml
  temperature_K: 300
  friction_per_fs: 0.01
  seed: 1235
  remove_com_each_step: false
```

Scalars can be replaced by length-`B` sequences for time step, temperature, and friction.

## Isotropic MTK NPT

```yaml
md:
  timestep_fs: 0.5
  n_steps: 10000
  initialize_velocities: true
  initial_temperature_K: 300
  initialization_seed: 1234
  temperature_K: 300
  pressure_GPa: 0.0
  thermostat_damping_fs: 50.0
  barostat_damping_fs: 500.0
  thermostat_chain_length: 3
  barostat_chain_length: 3
  thermostat_substeps: 1
  barostat_substeps: 1
  callback_interval: 10
```

Set `task: npt_mtk`. Pressure may instead be supplied as
`pressure_eV_per_A3`; do not set both units. Time step, temperature, pressure,
and both damping times accept scalars or length-`B` sequences. The implemented
Martyna-Tobias-Klein cell degree of freedom scales each fully periodic cell
isotropically. It does not change cell shape and rejects partial or nonperiodic
systems.

## Reporting

```yaml
reporting:
  trajectory: runs/job/trajectory.extxyz
  diagnostics: runs/job/diagnostics.jsonl
  checkpoint: runs/job/latest_state.pt
  summary: runs/job/summary.json
  wrap: false
```
