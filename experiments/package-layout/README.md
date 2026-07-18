# Functional package layout

## Hypothesis

Moving the flat implementation modules into five responsibility-based
subpackages should improve navigation without changing user-facing behavior.

## Layout

```text
atombit_batch/
  core/          batch state, calculator contracts, shared types and graph tools
  optimization/  FIRE, BFGS, cell filters and optimizer registration
  dynamics/      molecular-dynamics integrators
  models/        MLIP adapters, loaders and reference models
  interfaces/    Python API, CLI/configuration and reporting
```

The package root continues to export the documented API. Legacy submodule names
are installed as aliases during package import so existing scripts, YAML model
factory strings, and old serialized toy models remain loadable without leaving
forwarding files in the root directory.

## Validation

Compatibility checks:

- all 17 former submodule imports resolve to their canonical new module;
- root public classes/functions retain canonical identities;
- legacy `atombit_batch.toy_models` YAML factories build successfully;
- a model serialized with the old toy-model module path deserializes;
- existing YAML run/validation and optimizer registry tests pass;
- isolated-wheel `atombit-batch` and `python -m atombit_batch` entry points pass.

Package and regression checks:

```text
python -m pytest -q
38 passed in 33.49s

python -m ruff check atombit_batch tests benchmarks
All checks passed!
```

The clean wheel contains 24 package Python files. Its only root Python files are
`__init__.py` and `__main__.py`; its console entry is
`atombit_batch.interfaces.cli:main`. The initial wheel incorrectly included
stale flat modules from a generated `build/` directory. That negative result
was retained in `checks/wheel.log`; after deleting generated build metadata,
`checks/clean_wheel.log` records the clean wheel.

Production smoke inference loaded the AtomBit checkpoint through the new
`core.state` and `models.potential` paths for a fixed 46-atom T2 sample. It
returned finite energy shape `[1]`, force shape `[46, 3]`, and energy
`-0.67355156 eV`. Machine-readable details are in `results.json` and
`production_smoke.json`.

## Compatibility boundary

Ordinary imports such as `from atombit_batch.cli import run_config` remain
supported. The installed console command and `python -m atombit_batch` are the
supported execution forms. Direct execution of the internal legacy alias with
`python -m atombit_batch.cli` is replaced by `python -m atombit_batch` because
module aliases do not provide a second executable loader specification.
