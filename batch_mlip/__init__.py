"""Torch-native batched optimization and MD for graph MLIPs."""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from .core.calculator import ASECalculatorAdapter, BatchCalculator
from .core.state import AseGraphBatch
from .core.types import (
    BatchEvaluation,
    EvaluationResult,
    GraphData,
    MDResult,
    RelaxationResult,
)
from .dynamics.integrators import (
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from .interfaces.api import evaluate, molecular_dynamics, relax
from .interfaces.reporting import (
    CompositeReporter,
    ExtXYZReporter,
    JSONLReporter,
    TorchStateCheckpointReporter,
    build_reporter,
)
from .models.mace import MACEBatchCalculator, load_mace_off_batch
from .models.potential import AtomBitBatchCalculator, BatchedPotential
from .optimization.bfgs import batched_bfgs_relax
from .optimization.cell_filters import BatchedFrechetCellFilter, FrechetCellFilter
from .optimization.fire import (
    FIREConfig,
    batched_fire_relax,
    batched_gradient_descent,
    max_force_per_system,
)
from .optimization.registry import (
    BatchedBFGS,
    BatchedFIRE,
    BatchedGradientDescent,
    BatchOptimizer,
    OptimizerCapabilities,
    OptimizerFactory,
    available_optimizers,
    create_optimizer,
    register_optimizer,
)
from .planning import (
    BatchPlan,
    BatchPlanner,
    CalibrationObservation,
    MemoryCoefficients,
    PlannedBucket,
    SystemProfile,
    fit_memory_coefficients,
)
from .profiling import RuntimeProfiler


def _install_legacy_module_aliases() -> None:
    """Keep pre-reorganization import paths and serialized models loadable."""

    from .core import calculator, math_utils, neighbors, state, types
    from .dynamics import integrators
    from .interfaces import api, cli, config, reporting
    from .models import loaders, potential, toy_models
    from .optimization import bfgs, cell_filters, fire, registry

    aliases: dict[str, _ModuleType] = {
        "api": api,
        "bfgs": bfgs,
        "calculator": calculator,
        "cli": cli,
        "config": config,
        "filters": cell_filters,
        "loaders": loaders,
        "math_utils": math_utils,
        "md": integrators,
        "neighbors": neighbors,
        "optimize": fire,
        "optimizers": registry,
        "potential": potential,
        "reporting": reporting,
        "state": state,
        "toy_models": toy_models,
        "types": types,
    }
    package = _sys.modules[__name__]
    for legacy_name, module in aliases.items():
        qualified_name = f"{__name__}.{legacy_name}"
        _sys.modules.setdefault(qualified_name, module)
        setattr(package, legacy_name, module)


_install_legacy_module_aliases()

__all__ = [
    "AseGraphBatch",
    "ASECalculatorAdapter",
    "BatchEvaluation",
    "BatchCalculator",
    "BatchOptimizer",
    "BatchPlan",
    "BatchPlanner",
    "BatchedBFGS",
    "BatchedFIRE",
    "BatchedFrechetCellFilter",
    "FrechetCellFilter",
    "BatchedGradientDescent",
    "AtomBitBatchCalculator",
    "BatchedPotential",
    "CompositeReporter",
    "CalibrationObservation",
    "EvaluationResult",
    "ExtXYZReporter",
    "FIREConfig",
    "GraphData",
    "JSONLReporter",
    "MACEBatchCalculator",
    "MemoryCoefficients",
    "MDResult",
    "OptimizerCapabilities",
    "OptimizerFactory",
    "PlannedBucket",
    "RelaxationResult",
    "RuntimeProfiler",
    "SystemProfile",
    "TorchStateCheckpointReporter",
    "available_optimizers",
    "batched_bfgs_relax",
    "batched_fire_relax",
    "batched_gradient_descent",
    "batched_langevin_baoab",
    "batched_velocity_verlet",
    "build_reporter",
    "create_optimizer",
    "evaluate",
    "fit_memory_coefficients",
    "initialize_maxwell_boltzmann",
    "load_mace_off_batch",
    "max_force_per_system",
    "molecular_dynamics",
    "register_optimizer",
    "relax",
]

__version__ = "0.2.0"
