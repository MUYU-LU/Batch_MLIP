"""Torch-native batched optimization and MD for graph MLIPs."""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from .core.calculator import ASECalculatorAdapter, BatchCalculator
from .core.neighbors import NeighborBackend
from .core.state import AseGraphBatch
from .core.types import (
    BatchEvaluation,
    EvaluationResult,
    GraphData,
    MDResult,
    RelaxationResult,
)
from .dynamics.integrators import (
    LangevinBAOABState,
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from .dynamics.mtk import IsotropicMTKState, batched_isotropic_mtk
from .execution import (
    CudaAllocatorPlan,
    CudaAllocatorPolicy,
    MultiGPUExecution,
    MultiGPUTaskExecution,
    ParallelWorkerError,
    PersistentTaskExecution,
    PersistentTaskPool,
    TaskResult,
    TaskWorker,
    TaskWorkerResult,
    WorkerResult,
    WorkerShard,
    balance_work,
    run_parallel_task_workers,
    run_parallel_workers,
    select_cuda_allocator,
)
from .interfaces.api import evaluate, molecular_dynamics, relax
from .interfaces.ase_reference import relax_ase
from .interfaces.executor import BatchExecutor
from .interfaces.reporting import (
    CompositeReporter,
    ExtXYZReporter,
    JSONLReporter,
    TorchStateCheckpointReporter,
    build_reporter,
)
from .models.mace import MACEBatchCalculator, load_mace_off_batch
from .models.potential import (
    AtomBitBatchCalculator,
    BatchedPotential,
    load_atombit_batch,
)
from .optimization.bfgs import batched_bfgs_relax
from .optimization.bfgs_line_search import batched_bfgs_line_search_relax
from .optimization.cell_filters import BatchedFrechetCellFilter, FrechetCellFilter
from .optimization.fire import (
    FIREConfig,
    batched_fire_relax,
    batched_gradient_descent,
    max_force_per_system,
)
from .optimization.registry import (
    BatchedBFGS,
    BatchedBFGSLineSearch,
    BatchedFIRE,
    BatchedGradientDescent,
    BatchedQuasiNewton,
    BatchOptimizer,
    OptimizerCapabilities,
    OptimizerFactory,
    available_optimizers,
    create_optimizer,
    register_optimizer,
)
from .planning import (
    AutoBatchAction,
    AutoBatchObservation,
    AutoPolicyCache,
    AutoScheduler,
    AutoSchedulerConfig,
    AutoWorkloadBucket,
    AutoWorkloadPlan,
    BatchPlan,
    BatchPlanner,
    BatchTimingPoint,
    CachedAutoPolicy,
    CalibrationObservation,
    DeterministicMemoryProbe,
    DeterministicRelaxationChunk,
    DeterministicRelaxationPlan,
    MemoryCoefficients,
    OnlineCapacityController,
    OptimizationPilot,
    PilotRegime,
    PlannedBucket,
    RefillPrediction,
    RelaxationSchedule,
    ScheduledRelaxationBatch,
    SystemProfile,
    TaskAwarePolicy,
    execution_fingerprint,
    fit_memory_coefficients,
    load_refill_policy,
    model_state_sha256,
    optimizer_dof_bytes,
    plan_deterministic_relaxation,
    plan_relaxation_execution,
    plan_task_aware_relaxation,
    predict_refill,
    profile_auto_workload,
    profile_model_work,
    select_probe_indices,
)
from .profiling import RunTelemetry, RuntimeProfiler
from .workloads import (
    TaskProfile,
    WorkloadExecutionResult,
    WorkloadJob,
    WorkloadManifest,
    WorkloadRunSpec,
    build_task_aware_holdout_workloads,
    execute_workload,
    materialize_workload,
)


def _install_legacy_module_aliases() -> None:
    """Keep pre-reorganization import paths and serialized models loadable."""

    from .core import calculator, math_utils, neighbors, state, types
    from .dynamics import integrators
    from .interfaces import api, cli, config, reporting
    from .models import loaders, potential, toy_models
    from .optimization import bfgs, bfgs_line_search, cell_filters, fire, registry

    aliases: dict[str, _ModuleType] = {
        "api": api,
        "bfgs": bfgs,
        "bfgs_line_search": bfgs_line_search,
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
    "AutoBatchAction",
    "AutoBatchObservation",
    "AutoPolicyCache",
    "AutoScheduler",
    "AutoSchedulerConfig",
    "AutoWorkloadBucket",
    "AutoWorkloadPlan",
    "BatchEvaluation",
    "BatchExecutor",
    "BatchCalculator",
    "BatchOptimizer",
    "BatchPlan",
    "BatchPlanner",
    "BatchTimingPoint",
    "BatchedBFGS",
    "BatchedBFGSLineSearch",
    "BatchedFIRE",
    "BatchedFrechetCellFilter",
    "FrechetCellFilter",
    "BatchedGradientDescent",
    "BatchedQuasiNewton",
    "AtomBitBatchCalculator",
    "BatchedPotential",
    "CompositeReporter",
    "CalibrationObservation",
    "CachedAutoPolicy",
    "CudaAllocatorPlan",
    "CudaAllocatorPolicy",
    "DeterministicMemoryProbe",
    "DeterministicRelaxationChunk",
    "DeterministicRelaxationPlan",
    "EvaluationResult",
    "ExtXYZReporter",
    "FIREConfig",
    "GraphData",
    "IsotropicMTKState",
    "JSONLReporter",
    "LangevinBAOABState",
    "MACEBatchCalculator",
    "MultiGPUTaskExecution",
    "MemoryCoefficients",
    "OptimizationPilot",
    "OnlineCapacityController",
    "MDResult",
    "MultiGPUExecution",
    "NeighborBackend",
    "OptimizerCapabilities",
    "OptimizerFactory",
    "ParallelWorkerError",
    "PersistentTaskExecution",
    "PersistentTaskPool",
    "PlannedBucket",
    "PilotRegime",
    "RelaxationSchedule",
    "RelaxationResult",
    "RuntimeProfiler",
    "RunTelemetry",
    "SystemProfile",
    "TaskAwarePolicy",
    "TaskResult",
    "TaskWorker",
    "TaskWorkerResult",
    "ScheduledRelaxationBatch",
    "TorchStateCheckpointReporter",
    "TaskProfile",
    "WorkerResult",
    "WorkerShard",
    "WorkloadJob",
    "WorkloadManifest",
    "WorkloadExecutionResult",
    "WorkloadRunSpec",
    "available_optimizers",
    "balance_work",
    "batched_bfgs_relax",
    "batched_bfgs_line_search_relax",
    "batched_fire_relax",
    "batched_gradient_descent",
    "batched_isotropic_mtk",
    "batched_langevin_baoab",
    "batched_velocity_verlet",
    "build_reporter",
    "build_task_aware_holdout_workloads",
    "create_optimizer",
    "evaluate",
    "execution_fingerprint",
    "execute_workload",
    "fit_memory_coefficients",
    "initialize_maxwell_boltzmann",
    "load_mace_off_batch",
    "load_atombit_batch",
    "max_force_per_system",
    "materialize_workload",
    "molecular_dynamics",
    "optimizer_dof_bytes",
    "plan_deterministic_relaxation",
    "plan_relaxation_execution",
    "plan_task_aware_relaxation",
    "RefillPrediction",
    "load_refill_policy",
    "model_state_sha256",
    "predict_refill",
    "profile_auto_workload",
    "profile_model_work",
    "register_optimizer",
    "relax",
    "relax_ase",
    "run_parallel_workers",
    "run_parallel_task_workers",
    "select_cuda_allocator",
    "select_probe_indices",
]

__version__ = "0.2.0"
