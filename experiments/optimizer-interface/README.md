# Extensible optimizer interface

## Hypothesis

A runtime optimizer protocol and validated registry can remove the hard-coded
optimizer switch without changing the validated FIRE numerical kernels.

## Design

- `BatchOptimizer` defines `capabilities()` and `run(state, calculator, **options)`.
- `OptimizerFactory` defines construction from keyword defaults.
- `BatchedFIRE` and `BatchedGradientDescent` wrap existing implementations.
- `create_optimizer()` constructs registered names; `register_optimizer()` adds
  third-party batched implementations.
- `relax()` accepts either a registered string or a direct optimizer object.
- Capability validation prevents unsupported variable-cell or active-compaction
  requests from being silently ignored.

This experiment intentionally does not reinterpret ordinary ASE optimizer
classes as batched optimizers. They require per-system Python objects and would
not retain native graph batching or active-state compaction.

## Validation

Baseline before the change:

```text
python -m pytest -q
19 passed in 23.06s
```

Final checks:

```text
python -m pytest -q
26 passed in 32.66s

python -m ruff check atombit_batch tests benchmarks
All checks passed!

python -m pip wheel . --no-deps --wheel-dir runs/optimizer_interface_checks/dist
Successfully built atombit-batch-lab
```

Production AtomBit validation used two fixed 46-atom T2 manifest structures,
float32 on an H100, three FIRE steps, active compaction, and both fixed-cell
and full Frechet variable-cell modes. The registered `"fire"` path and a direct
`BatchedFIRE()` object had identical convergence tensors, step counts, and
active-batch-size histories.

The object-path maximum differences were `2.38e-6 eV` in energy and
`1.42e-5 eV/A` in force for fixed cells, and `2.38e-6 eV`, `4.28e-5 eV/A`,
`9.54e-7 A` in positions, and `2.98e-8 eV/A^3` in stress for variable cells.
A repeated registered-name control produced differences of the same order, so
these are GPU float32 reduction variation rather than dispatch changes. Exact
values and tolerances are stored in `results.json`.

## Files changed

- `atombit_batch/optimization/registry.py`: protocols, capabilities, built-in objects,
  registry, and factory.
- `atombit_batch/interfaces/api.py`: string-or-object optimizer dispatch.
- `atombit_batch/interfaces/cli.py`: registry-backed YAML dispatch.
- `atombit_batch/__init__.py`: public exports.
- `tests/test_optimizer_interface.py`: extension and failure-mode tests.
- `benchmarks/validate_optimizer_interface.py`: production equivalence check.

## Limitations and next step

The protocol accepts native batched optimizers; it cannot make an ordinary ASE
optimizer batched automatically. FIRE currently remains the only built-in
optimizer from this interface experiment; the subsequent BFGS integration adds
`BatchedBFGS` with variable cells and active compaction. LBFGS remains the next
scalable optimizer, including per-system history compaction and
fixed/variable-cell ASE references.
