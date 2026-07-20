# Configuration Reference

Every run configuration uses `schema_version: 1`.

## Top level

```yaml
task: relax | nve | nvt_langevin
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
```

`auto` uses the CPU backend for small rebuilds and the integrated dense CUDA
backend above a cutoff-aware work threshold. `matscipy` forces CPU construction
(with ASE fallback outside matscipy's validated fully periodic path), while
`cuda_dense` requires CUDA and raises for unsupported degenerate periodic cells.

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

## Reporting

```yaml
reporting:
  trajectory: runs/job/trajectory.extxyz
  diagnostics: runs/job/diagnostics.jsonl
  checkpoint: runs/job/latest_state.pt
  summary: runs/job/summary.json
  wrap: false
```
