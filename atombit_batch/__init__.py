"""Compatibility namespace for the package formerly named ``atombit_batch``.

New code should import from :mod:`batch_mlip`. This module intentionally
forwards both public symbols and historical submodule paths without copying the
implementation.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

import batch_mlip as _implementation

_CANONICAL_MODULES = (
    "core",
    "core.calculator",
    "core.math_utils",
    "core.neighbors",
    "core.state",
    "core.types",
    "dynamics",
    "dynamics.integrators",
    "interfaces",
    "interfaces.api",
    "interfaces.cli",
    "interfaces.config",
    "interfaces.reporting",
    "models",
    "models.loaders",
    "models.mace",
    "models.potential",
    "models.toy_models",
    "optimization",
    "optimization.bfgs",
    "optimization.cell_filters",
    "optimization.fire",
    "optimization.registry",
)

for _module_name in _CANONICAL_MODULES:
    _module = _importlib.import_module(f"batch_mlip.{_module_name}")
    _sys.modules[f"{__name__}.{_module_name}"] = _module
    if "." not in _module_name:
        globals()[_module_name] = _module

for _module_name in (
    "api",
    "bfgs",
    "calculator",
    "cli",
    "config",
    "filters",
    "loaders",
    "math_utils",
    "md",
    "neighbors",
    "optimize",
    "optimizers",
    "potential",
    "reporting",
    "state",
    "toy_models",
    "types",
):
    _module = getattr(_implementation, _module_name)
    _sys.modules[f"{__name__}.{_module_name}"] = _module
    globals()[_module_name] = _module

__all__ = list(_implementation.__all__)
globals().update({name: getattr(_implementation, name) for name in __all__})
__version__ = _implementation.__version__
