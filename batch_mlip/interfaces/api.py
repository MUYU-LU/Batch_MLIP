"""Public structure-level API shared by relaxation and MD."""

from __future__ import annotations

import copy
import math
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, PriorityQueue
from threading import Lock
from typing import Any, Literal

import torch
from ase import Atoms

from ..core.calculator import BatchCalculator
from ..core.types import (
    BatchEvaluation,
    EvaluationResult,
    MDResult,
    RelaxationResult,
)
from ..dynamics.integrators import batched_langevin_baoab, batched_velocity_verlet
from ..execution import TaskWorker, run_parallel_task_workers
from ..optimization.registry import BatchOptimizer, create_optimizer
from ..planning.auto import (
    AutoBatchAction,
    AutoBatchObservation,
    AutoScheduler,
    AutoSchedulerConfig,
)
from ..planning.execution import (
    RelaxationSchedule,
    plan_relaxation_execution,
)
from ..planning.memory import BatchPlanner, SystemProfile
from ..planning.policy import (
    OptimizationPilot,
    TaskAwarePolicy,
    plan_task_aware_relaxation,
)


def _normalize_systems(systems: Atoms | Sequence[Atoms]) -> list[Atoms]:
    normalized = [systems] if isinstance(systems, Atoms) else list(systems)
    if not normalized:
        raise ValueError("systems must contain at least one ASE Atoms object")
    if not all(isinstance(atoms, Atoms) for atoms in normalized):
        raise TypeError("every system must be an ASE Atoms object")
    return normalized


def evaluate(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    compute_stress: bool = False,
) -> EvaluationResult:
    """Evaluate structures in one calculator call and preserve input order."""

    state = calculator.create_state(_normalize_systems(systems))
    evaluation = calculator(state, compute_stress=compute_stress)
    return EvaluationResult(state=state, evaluation=evaluation)


def relax(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    optimizer: str | BatchOptimizer = "fire",
    scheduling: Literal["single_batch", "auto"] = "single_batch",
    planner: BatchPlanner | None = None,
    pilot: OptimizationPilot | None = None,
    policy: TaskAwarePolicy | None = None,
    system_profiles: Sequence[SystemProfile] | None = None,
    auto_config: AutoSchedulerConfig | None = None,
    devices: Sequence[str | torch.device] | None = None,
    **optimizer_kwargs: Any,
) -> RelaxationResult:
    """Relax structures directly or through an automatic execution schedule.

    With no planner or pilot, ``scheduling="auto"`` profiles and tunes
    production chunks internally. Supplying a planner retains the explicit
    reproducible scheduling paths used by controlled experiments.
    """

    resolved = create_optimizer(optimizer) if isinstance(optimizer, str) else optimizer
    if not isinstance(resolved, BatchOptimizer):
        raise TypeError(
            "optimizer must be a registered name or implement BatchOptimizer"
        )
    normalized = _normalize_systems(systems)
    resolved_devices = _normalize_devices(devices)
    if resolved_devices and scheduling != "auto":
        raise ValueError("devices are only used with scheduling='auto'")
    if resolved_devices and resolved_devices[0] != calculator.device:
        calculator = calculator.clone_to(resolved_devices[0])
    if scheduling == "auto":
        if calculator.cutoff is None:
            raise ValueError("automatic scheduling requires a calculator cutoff")
        if optimizer_kwargs.get("refill_batch_size") is not None:
            raise ValueError(
                "automatic scheduling controls refill_batch_size; do not set it"
            )
        capabilities = resolved.capabilities()
        if len(resolved_devices) > 1:
            if planner is not None or pilot is not None or policy is not None:
                raise ValueError(
                    "multi-GPU automatic scheduling does not accept an "
                    "explicit planner or pilot"
                )
            if system_profiles is not None:
                raise ValueError(
                    "multi-GPU automatic scheduling profiles the workload internally"
                )
            config = auto_config or AutoSchedulerConfig()
            if not config.cache_enabled:
                raise ValueError(
                    "multi-GPU cold-start coordination requires the policy cache"
                )
            options = dict(optimizer_kwargs)
            if getattr(capabilities, "active_compaction", False):
                options.setdefault("active_compaction", True)
            return _execute_multi_device_auto_relaxation(
                normalized,
                calculator,
                resolved,
                resolved_devices,
                config,
                options,
            )
        if planner is None:
            if pilot is not None or policy is not None:
                raise ValueError(
                    "an explicit task-aware pilot requires a calibrated BatchPlanner"
                )
            if system_profiles is not None:
                raise ValueError(
                    "system_profiles currently require an explicit BatchPlanner"
                )
            options = dict(optimizer_kwargs)
            if getattr(capabilities, "active_compaction", False):
                options.setdefault("active_compaction", True)
            scheduler = AutoScheduler(
                calculator,
                resolved,
                options,
                config=auto_config,
                supports_refill=getattr(capabilities, "active_refill", False),
            )
            return _execute_online_auto_relaxation(
                normalized,
                calculator,
                resolved,
                scheduler,
                options,
            )
        if auto_config is not None:
            raise ValueError(
                "auto_config is only used by zero-configuration automatic scheduling"
            )
        if pilot is None:
            if policy is not None:
                raise ValueError("a task-aware policy requires an optimization pilot")
            if system_profiles is not None:
                raise ValueError(
                    "cached system_profiles require an optimization pilot"
                )
            schedule = plan_relaxation_execution(
                planner,
                normalized,
                cutoff=calculator.cutoff,
                skin=calculator.skin,
                supports_refill=getattr(capabilities, "active_refill", False),
            )
        else:
            if not getattr(capabilities, "active_compaction", False):
                raise ValueError(
                    "task-aware scheduling requires active-compaction support"
                )
            if optimizer_kwargs.get("active_compaction") is False:
                raise ValueError(
                    "task-aware scheduling requires active_compaction=True"
                )
            if optimizer_kwargs.get("refill_policy", "immediate") != "immediate":
                raise ValueError(
                    "the task-aware refill model currently requires "
                    "refill_policy='immediate'"
                )
            schedule = plan_task_aware_relaxation(
                planner,
                normalized,
                cutoff=calculator.cutoff,
                skin=calculator.skin,
                pilot=pilot,
                policy=policy,
                supports_refill=getattr(capabilities, "active_refill", False),
                refill_storage=optimizer_kwargs.get("refill_storage", "auto"),
                optimizer_name=(
                    optimizer if isinstance(optimizer, str) else type(resolved).__name__
                ),
                system_profiles=system_profiles,
            )
        return _execute_relaxation_schedule(
            normalized,
            calculator,
            resolved,
            schedule,
            optimizer_kwargs,
        )
    if scheduling != "single_batch":
        raise ValueError(f"unsupported relaxation scheduling {scheduling!r}")
    if planner is not None:
        raise ValueError("planner is only used with scheduling='auto'")
    if pilot is not None or policy is not None:
        raise ValueError("pilot and policy are only used with scheduling='auto'")
    if auto_config is not None:
        raise ValueError("auto_config is only used with scheduling='auto'")
    if system_profiles is not None:
        raise ValueError(
            "system_profiles are only used with scheduling='auto' and a pilot"
        )
    return _run_optimizer(
        normalized,
        calculator,
        resolved,
        optimizer_kwargs,
    )


def _normalize_devices(
    devices: Sequence[str | torch.device] | None,
) -> tuple[torch.device, ...]:
    if devices is None:
        return ()
    if isinstance(devices, (str, torch.device)):
        normalized = (torch.device(devices),)
    else:
        normalized = tuple(torch.device(device) for device in devices)
    if not normalized:
        raise ValueError("devices must not be empty")
    labels = [str(device) for device in normalized]
    if len(set(labels)) != len(labels):
        raise ValueError("devices must be unique")
    if len({device.type for device in normalized}) != 1:
        raise ValueError("devices must all use the same device type")
    cuda_hardware = []
    for device in normalized:
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        cuda_hardware.append((properties.name, properties.total_memory))
    if cuda_hardware and len(set(cuda_hardware)) != 1:
        raise ValueError(
            "automatic multi-GPU scheduling currently requires homogeneous GPUs"
        )
    return normalized


def _run_optimizer(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    optimizer_kwargs: dict[str, Any],
) -> RelaxationResult:
    capabilities = optimizer.capabilities()
    lazy_refill = (
        optimizer_kwargs.get("refill_batch_size") is not None
        and getattr(capabilities, "active_refill", False)
    )
    state = (
        calculator.create_state(systems, build_neighbors=False)
        if lazy_refill
        else calculator.create_state(systems)
    )
    return optimizer.run(state, calculator, **optimizer_kwargs)


def _combine_relaxation_results(
    indexed_results: list[tuple[tuple[int, ...], RelaxationResult]],
    *,
    workload_size: int,
    calculator: BatchCalculator,
) -> RelaxationResult:
    slots: list[tuple[RelaxationResult, int, Atoms] | None] = [
        None
    ] * workload_size
    for indices, result in indexed_results:
        structures = result.structures
        for local_index, (global_index, atoms) in enumerate(
            zip(indices, structures, strict=True)
        ):
            slots[global_index] = (result, local_index, atoms)
    if any(slot is None for slot in slots):
        raise RuntimeError("scheduled relaxation did not return every system")
    ordered = [slot for slot in slots if slot is not None]
    state = calculator.create_state(
        [slot[2] for slot in ordered],
        build_neighbors=False,
    )
    output_device = state.device
    state.neighbor_rebuild_count = sum(
        result.state.neighbor_rebuild_count
        for _, result in indexed_results
    )
    force_blocks = [
        result.evaluation.forces[result.state.atom_slice(local_index)]
        for result, local_index, _ in ordered
    ]
    stress_available = all(
        result.evaluation.stress is not None for result, _, _ in ordered
    )
    max_stress_available = all(
        result.max_stress is not None for result, _, _ in ordered
    )
    evaluation = BatchEvaluation(
        energy=torch.stack(
            [
                result.evaluation.energy[local_index]
                for result, local_index, _ in ordered
            ]
        ).to(output_device),
        forces=torch.cat(force_blocks).to(output_device),
        stress=(
            torch.stack(
                [
                    result.evaluation.stress[local_index]
                    for result, local_index, _ in ordered
                    if result.evaluation.stress is not None
                ]
            ).to(output_device)
            if stress_available
            else None
        ),
    )
    return RelaxationResult(
        state=state,
        evaluation=evaluation,
        converged=torch.stack(
            [result.converged[local_index] for result, local_index, _ in ordered]
        ).to(output_device),
        converged_step=torch.stack(
            [
                result.converged_step[local_index]
                for result, local_index, _ in ordered
            ]
        ).to(output_device),
        max_force=torch.stack(
            [result.max_force[local_index] for result, local_index, _ in ordered]
        ).to(output_device),
        max_stress=(
            torch.stack(
                [
                    result.max_stress[local_index]
                    for result, local_index, _ in ordered
                    if result.max_stress is not None
                ]
            ).to(output_device)
            if max_stress_available
            else None
        ),
        steps=max(result.steps for _, result in indexed_results),
        model_evaluations=sum(
            result.model_evaluations for _, result in indexed_results
        ),
        graph_evaluations=sum(
            result.graph_evaluations for _, result in indexed_results
        ),
        active_batch_sizes=tuple(
            size
            for _, result in indexed_results
            for size in result.active_batch_sizes
        ),
    )


def _offload_relaxation_result(result: RelaxationResult) -> RelaxationResult:
    """Move a completed scheduled bucket off GPU before the next bucket."""

    return RelaxationResult(
        state=result.state.to("cpu"),
        evaluation=BatchEvaluation(
            energy=result.evaluation.energy.cpu(),
            forces=result.evaluation.forces.cpu(),
            stress=(
                None
                if result.evaluation.stress is None
                else result.evaluation.stress.cpu()
            ),
        ),
        converged=result.converged.cpu(),
        converged_step=result.converged_step.cpu(),
        max_force=result.max_force.cpu(),
        max_stress=(
            None if result.max_stress is None else result.max_stress.cpu()
        ),
        steps=result.steps,
        model_evaluations=result.model_evaluations,
        graph_evaluations=result.graph_evaluations,
        active_batch_sizes=result.active_batch_sizes,
        metadata=dict(result.metadata),
    )


def _empty_device_cache(device: torch.device) -> None:
    if device.type != "cuda":
        return
    with torch.cuda.device(device):
        torch.cuda.empty_cache()


def _execute_relaxation_schedule(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    schedule: RelaxationSchedule,
    optimizer_kwargs: dict[str, Any],
) -> RelaxationResult:
    indexed_results = []
    offload_completed = len(schedule.batches) > 1
    for batch in schedule.batches:
        options = dict(optimizer_kwargs)
        if schedule.decision == "task_aware_pilot_policy":
            options.setdefault("active_compaction", True)
        if batch.active_refill:
            options["refill_batch_size"] = batch.resident_capacity
            options.setdefault("refill_policy", "immediate")
            options.setdefault("refill_storage", batch.refill_storage)
        result = _run_optimizer(
            [systems[index] for index in batch.system_indices],
            calculator,
            optimizer,
            options,
        )
        if offload_completed and result.state.device.type == "cuda":
            result_device = result.state.device
            result = _offload_relaxation_result(result)
            _empty_device_cache(result_device)
        indexed_results.append((batch.system_indices, result))
    if len(indexed_results) == 1:
        result = indexed_results[0][1]
    else:
        result = _combine_relaxation_results(
            indexed_results,
            workload_size=len(systems),
            calculator=calculator,
        )
    batch_metadata = []
    for batch in schedule.batches:
        record = {
            "system_count": len(batch.system_indices),
            "resident_capacity": batch.resident_capacity,
            "active_refill": batch.active_refill,
        }
        if schedule.decision == "task_aware_pilot_policy":
            record["refill_storage"] = batch.refill_storage
            record["predicted_seconds"] = batch.predicted_seconds
        batch_metadata.append(record)
    result.metadata["scheduling"] = {
        "policy": "auto",
        "decision": schedule.decision,
        "total_predicted_bytes": schedule.total_predicted_bytes,
        "memory_budget_bytes": schedule.plan.memory_budget_bytes,
        "profiling_seconds": schedule.plan.profiling_seconds,
        "batches": batch_metadata,
        **schedule.metadata,
    }
    return result


def _timed_online_batch(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    optimizer_kwargs: dict[str, Any],
    action: AutoBatchAction,
    *,
    memory_safety_fraction: float,
) -> tuple[RelaxationResult, AutoBatchObservation]:
    device = calculator.device
    baseline_allocated = None
    baseline_reserved = None
    memory_budget = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        memory_budget = baseline_reserved + int(
            free_bytes * memory_safety_fraction
        )
        torch.cuda.reset_peak_memory_stats(device)

    options = dict(optimizer_kwargs)
    if action.active_refill:
        options["refill_batch_size"] = action.resident_capacity
        options.setdefault("refill_policy", "immediate")
        options.setdefault("refill_storage", action.refill_storage)
    started = time.perf_counter()
    result = _run_optimizer(
        systems,
        calculator,
        optimizer,
        options,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    system_evaluations = sum(result.active_batch_sizes)
    if system_evaluations <= 0 or result.model_evaluations <= 0:
        raise RuntimeError(
            "online scheduling requires optimizer active-batch telemetry"
        )
    occupancy = system_evaluations / (
        action.resident_capacity * result.model_evaluations
    )
    observation = AutoBatchObservation(
        system_count=len(systems),
        resident_capacity=action.resident_capacity,
        active_refill=action.active_refill,
        wall_seconds=wall_seconds,
        system_evaluations=system_evaluations,
        model_evaluations=result.model_evaluations,
        mean_active_occupancy=min(1.0, occupancy),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        baseline_allocated_bytes=baseline_allocated,
        baseline_reserved_bytes=baseline_reserved,
        memory_budget_bytes=memory_budget,
    )
    return result, observation


def _execute_online_auto_relaxation(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    scheduler: AutoScheduler,
    optimizer_kwargs: dict[str, Any],
) -> RelaxationResult:
    """Tune future queues from completed production work without repeats."""

    plan = scheduler.plan(systems)
    indexed_results: list[tuple[tuple[int, ...], RelaxationResult]] = []
    bucket_metadata = []
    batch_metadata = []
    controller_seconds = 0.0
    for bucket in plan.buckets:
        controller = scheduler.controller(plan, bucket)
        pending = list(bucket.system_indices)
        cache_hit = controller.cache_hit
        while pending:
            decision_started = time.perf_counter()
            action = controller.next_action(len(pending))
            controller_seconds += time.perf_counter() - decision_started
            indices = tuple(pending[: action.system_count])
            del pending[: action.system_count]
            result, observation = _timed_online_batch(
                [systems[index] for index in indices],
                calculator,
                optimizer,
                optimizer_kwargs,
                action,
                memory_safety_fraction=scheduler.config.memory_safety_fraction,
            )
            if result.state.device.type == "cuda":
                result_device = result.state.device
                result = _offload_relaxation_result(result)
                _empty_device_cache(result_device)
            indexed_results.append((indices, result))
            decision_started = time.perf_counter()
            controller.observe(observation, remaining=len(pending))
            controller_seconds += time.perf_counter() - decision_started
            batch_metadata.append(
                {
                    "system_count": len(indices),
                    "resident_capacity": action.resident_capacity,
                    "active_refill": action.active_refill,
                    **observation.to_dict(),
                }
            )
        scheduler.save(controller)
        selected = controller.cached_result()
        bucket_metadata.append(
            {
                "system_count": len(bucket.system_indices),
                "mean_atom_count": bucket.mean_atom_count,
                "mean_edge_count": bucket.mean_edge_count,
                "mean_dof_squared": bucket.mean_dof_squared,
                "homogeneous_atom_count": bucket.homogeneous_atom_count,
                "cache_hit": cache_hit,
                "selected_resident_capacity": selected.resident_capacity,
                "selected_capacity_source": selected.capacity_source,
                "selected_active_refill": selected.active_refill,
                "selected_throughput": selected.throughput,
            }
        )
    result = _combine_relaxation_results(
        indexed_results,
        workload_size=len(systems),
        calculator=calculator,
    )
    result.metadata["scheduling"] = {
        "policy": "auto",
        "decision": "online_autotune",
        "fingerprint": plan.fingerprint,
        "fingerprint_fields": plan.fingerprint_fields,
        "cache_path": str(scheduler.cache.path),
        "cache_enabled": scheduler.cache.enabled,
        "profiling_seconds": plan.profiling_seconds,
        "controller_seconds": controller_seconds,
        "batches": batch_metadata,
        "buckets": bucket_metadata,
        "mps_requires_external_dispatch": False,
        "recommended_worker_mode": "tensor",
    }
    return result


@dataclass(frozen=True)
class _PendingAutoChunk:
    indices: tuple[int, ...]
    estimated_cost: float
    bucket_index: int


@dataclass
class _ProcessAutoWorkerRunner:
    """Persistent child-owned calculator and optimizer for planned chunks."""

    systems: list[Atoms]
    calculator: BatchCalculator
    optimizer: BatchOptimizer
    config: AutoSchedulerConfig
    optimizer_kwargs: dict[str, Any]

    def __call__(self, chunk: _PendingAutoChunk) -> RelaxationResult:
        result = relax(
            [self.systems[index] for index in chunk.indices],
            self.calculator,
            optimizer=self.optimizer,
            scheduling="auto",
            auto_config=self.config,
            **self.optimizer_kwargs,
        )
        if result.state.device.type == "cuda":
            device = result.state.device
            torch.cuda.synchronize(device)
            result = _offload_relaxation_result(result)
            _empty_device_cache(device)
        return result


@dataclass
class _ProcessAutoWorkerPreparer:
    """Serializable factory that materializes model state inside each child."""

    systems: list[Atoms]
    calculator_template: BatchCalculator
    optimizer_factory: type[Any] | None
    optimizer_options: dict[str, Any] | None
    optimizer_template: BatchOptimizer | None
    config: AutoSchedulerConfig
    optimizer_kwargs: dict[str, Any]

    def __call__(self, worker: TaskWorker) -> _ProcessAutoWorkerRunner:
        torch.set_num_threads(self.config.multi_gpu_process_cpu_threads)
        device = torch.device(worker.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        calculator = self.calculator_template.clone_to(device)
        if self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(**(self.optimizer_options or {}))
        elif self.optimizer_template is not None:
            optimizer = _clone_optimizer(self.optimizer_template)
        else:  # pragma: no cover - construction guarantees one representation
            raise RuntimeError("process optimizer representation is missing")
        compute_stress = self.optimizer_kwargs.get("cell_filter") is not None
        warm_state = calculator.create_state([self.systems[0]])
        calculator(warm_state, compute_stress=compute_stress)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        return _ProcessAutoWorkerRunner(
            systems=self.systems,
            calculator=calculator,
            optimizer=optimizer,
            config=self.config,
            optimizer_kwargs=self.optimizer_kwargs,
        )


def _profile_cost_for_dispatch(profile: SystemProfile) -> float:
    return max(
        1.0,
        profile.atom_count
        + profile.edge_count
        + math.sqrt(profile.dof_squared),
    )


def _uniform_pending_sample(
    indices: tuple[int, ...],
    count: int,
) -> tuple[int, ...]:
    if count >= len(indices):
        return indices
    return tuple(
        indices[position * len(indices) // count]
        for position in range(count)
    )


def _clone_optimizer(optimizer: BatchOptimizer) -> BatchOptimizer:
    options = getattr(optimizer, "options", None)
    if isinstance(options, Mapping):
        return type(optimizer)(**dict(options))
    return copy.deepcopy(optimizer)


def _prepare_process_workers(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    config: AutoSchedulerConfig,
    optimizer_kwargs: dict[str, Any],
) -> tuple[_ProcessAutoWorkerPreparer | None, str | None]:
    """Build and pickle-check a CPU template before production work starts."""

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if not main_file:
        return None, "spawn requires a file-backed __main__ module"
    try:
        calculator_template = calculator.clone_to("cpu")
        model = getattr(calculator_template, "model", None)
        if isinstance(model, torch.nn.Module):
            model.share_memory()
        optimizer_options = getattr(optimizer, "options", None)
        if isinstance(optimizer_options, Mapping):
            optimizer_factory: type[Any] | None = type(optimizer)
            serialized_options: dict[str, Any] | None = dict(optimizer_options)
            optimizer_template: BatchOptimizer | None = None
        else:
            optimizer_factory = None
            serialized_options = None
            optimizer_template = optimizer
        preparer = _ProcessAutoWorkerPreparer(
            systems=systems,
            calculator_template=calculator_template,
            optimizer_factory=optimizer_factory,
            optimizer_options=serialized_options,
            optimizer_template=optimizer_template,
            config=config,
            optimizer_kwargs=optimizer_kwargs,
        )
        pickle.dumps(preparer)
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    return preparer, None


def _execute_multi_device_auto_relaxation(
    systems: list[Atoms],
    calculator: BatchCalculator,
    optimizer: BatchOptimizer,
    devices: tuple[torch.device, ...],
    config: AutoSchedulerConfig,
    optimizer_kwargs: dict[str, Any],
) -> RelaxationResult:
    """Cold-tune once, then let GPU workers pull compatible pending chunks."""

    total_started = time.perf_counter()
    capabilities = optimizer.capabilities()
    scheduler = AutoScheduler(
        calculator,
        optimizer,
        optimizer_kwargs,
        config=config,
        supports_refill=getattr(capabilities, "active_refill", False),
    )
    plan = scheduler.plan(systems)
    indexed_results: list[tuple[tuple[int, ...], RelaxationResult]] = []
    cold_indices: set[int] = set()
    cold_records = []

    # Only the primary worker pays cold-start exploration. Its structures are
    # production jobs and remain in the final result.
    for bucket_index, bucket in enumerate(plan.buckets):
        controller = scheduler.controller(plan, bucket)
        if controller.cache_hit:
            continue
        indices = _uniform_pending_sample(
            bucket.system_indices,
            config.multi_gpu_cold_start_jobs,
        )
        cold_indices.update(indices)
        started = time.perf_counter()
        cold_result = _execute_online_auto_relaxation(
            [systems[index] for index in indices],
            calculator,
            optimizer,
            scheduler,
            optimizer_kwargs,
        )
        elapsed = time.perf_counter() - started
        if cold_result.state.device.type == "cuda":
            result_device = cold_result.state.device
            cold_result = _offload_relaxation_result(cold_result)
            _empty_device_cache(result_device)
        indexed_results.append((indices, cold_result))
        cold_records.append(
            {
                "bucket_index": bucket_index,
                "system_count": len(indices),
                "wall_seconds": elapsed,
                "schedule": cold_result.metadata["scheduling"],
            }
        )

    by_index = {profile.index: profile for profile in plan.profiles}
    pending_queue: PriorityQueue[tuple[float, int, _PendingAutoChunk]] = (
        PriorityQueue()
    )
    serial = 0
    planned_chunks = []
    execution_chunks: list[_PendingAutoChunk] = []
    for bucket_index, bucket in enumerate(plan.buckets):
        pending = [
            index
            for index in bucket.system_indices
            if index not in cold_indices
        ]
        if not pending:
            continue
        policy = scheduler.cache.find(
            plan.fingerprint,
            bucket,
            config,
        )
        if policy is None:
            raise RuntimeError(
                "cold-start scheduling did not produce a reusable bucket policy"
            )
        per_gpu_share = math.ceil(len(pending) / len(devices))
        capacity = max(
            1,
            min(
                policy.resident_capacity,
                per_gpu_share,
                config.max_batch_size,
            ),
        )
        for start in range(0, len(pending), capacity):
            indices = tuple(pending[start : start + capacity])
            cost = sum(
                _profile_cost_for_dispatch(by_index[index])
                for index in indices
            )
            chunk = _PendingAutoChunk(
                indices=indices,
                estimated_cost=cost,
                bucket_index=bucket_index,
            )
            pending_queue.put((-cost, serial, chunk))
            execution_chunks.append(chunk)
            planned_chunks.append(
                {
                    "bucket_index": bucket_index,
                    "system_count": len(indices),
                    "resident_target": capacity,
                    "estimated_cost": cost,
                }
            )
            serial += 1

    worker_count = min(len(devices), len(planned_chunks))
    worker_records: list[dict[str, Any]] = []
    requested_backend = config.multi_gpu_worker_backend
    selected_backend = "thread"
    fallback_reason = None
    process_preparer = None
    preparation_started = time.perf_counter()
    process_has_enough_work = len(planned_chunks) >= (
        config.multi_gpu_process_min_chunks_per_device * worker_count
    )
    if (
        planned_chunks
        and requested_backend == "auto"
        and not process_has_enough_work
    ):
        fallback_reason = (
            "process workers require at least "
            f"{config.multi_gpu_process_min_chunks_per_device} pending chunks "
            "per active device to amortize spawn startup"
        )
    elif planned_chunks and requested_backend != "thread":
        process_preparer, fallback_reason = _prepare_process_workers(
            systems,
            calculator,
            optimizer,
            config,
            optimizer_kwargs,
        )
        if process_preparer is not None:
            selected_backend = "process"
        elif requested_backend == "process":
            raise TypeError(
                "process multi-GPU workers require a serializable calculator "
                f"and optimizer: {fallback_reason}"
            )
    preparation_seconds = time.perf_counter() - preparation_started

    parallel_started = time.perf_counter()
    if planned_chunks:
        if selected_backend == "process":
            if process_preparer is None:  # pragma: no cover - narrowed above
                raise RuntimeError("process worker preparation was not initialized")
            execution = run_parallel_task_workers(
                execution_chunks,
                [chunk.estimated_cost for chunk in execution_chunks],
                [str(device) for device in devices[:worker_count]],
                process_preparer,
            )
            task_results = {
                task.task_index: task for task in execution.task_results
            }
            for task_index, task_result in task_results.items():
                chunk = execution_chunks[task_index]
                indexed_results.append((chunk.indices, task_result.payload))
            for worker_result in execution.worker_results:
                completed = []
                for task_index in worker_result.task_indices:
                    chunk = execution_chunks[task_index]
                    task_result = task_results[task_index]
                    completed.append(
                        {
                            "bucket_index": chunk.bucket_index,
                            "system_count": len(chunk.indices),
                            "estimated_cost": chunk.estimated_cost,
                            "wall_seconds": task_result.run_seconds,
                            "schedule": task_result.payload.metadata["scheduling"],
                        }
                    )
                worker_records.append(
                    {
                        "worker_id": worker_result.worker.worker_id,
                        "device": worker_result.worker.device,
                        "startup_seconds": worker_result.startup_seconds,
                        "wall_seconds": worker_result.run_seconds,
                        "chunks": completed,
                    }
                )
            worker_startup_wall_seconds = execution.startup_wall_seconds
            worker_run_wall_seconds = execution.run_wall_seconds
            process_end_to_end_seconds = execution.end_to_end_wall_seconds
        else:
            clone_started = time.perf_counter()
            worker_calculators = []
            worker_optimizers = []
            for device in devices[:worker_count]:
                worker_calculators.append(
                    calculator
                    if device == calculator.device
                    else calculator.clone_to(device)
                )
                worker_optimizers.append(_clone_optimizer(optimizer))
            preparation_seconds += time.perf_counter() - clone_started
            result_lock = Lock()

            def run_worker(
                worker_id: int,
                worker_calculator: BatchCalculator,
                worker_optimizer: BatchOptimizer,
            ) -> dict[str, Any]:
                worker_started = time.perf_counter()
                completed = []
                while True:
                    try:
                        _, _, chunk = pending_queue.get_nowait()
                    except Empty:
                        break
                    chunk_started = time.perf_counter()
                    try:
                        chunk_result = relax(
                            [systems[index] for index in chunk.indices],
                            worker_calculator,
                            optimizer=worker_optimizer,
                            scheduling="auto",
                            auto_config=config,
                            **optimizer_kwargs,
                        )
                        if chunk_result.state.device.type == "cuda":
                            result_device = chunk_result.state.device
                            chunk_result = _offload_relaxation_result(chunk_result)
                            _empty_device_cache(result_device)
                        with result_lock:
                            indexed_results.append((chunk.indices, chunk_result))
                        completed.append(
                            {
                                "bucket_index": chunk.bucket_index,
                                "system_count": len(chunk.indices),
                                "estimated_cost": chunk.estimated_cost,
                                "wall_seconds": time.perf_counter() - chunk_started,
                                "schedule": chunk_result.metadata["scheduling"],
                            }
                        )
                    finally:
                        pending_queue.task_done()
                return {
                    "worker_id": worker_id,
                    "device": str(worker_calculator.device),
                    "startup_seconds": 0.0,
                    "wall_seconds": time.perf_counter() - worker_started,
                    "chunks": completed,
                }

            thread_run_started = time.perf_counter()
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="batch-mlip-gpu",
            ) as pool:
                futures = [
                    pool.submit(
                        run_worker,
                        worker_id,
                        worker_calculators[worker_id],
                        worker_optimizers[worker_id],
                    )
                    for worker_id in range(worker_count)
                ]
                worker_records.extend(future.result() for future in futures)
            worker_startup_wall_seconds = 0.0
            worker_run_wall_seconds = time.perf_counter() - thread_run_started
            process_end_to_end_seconds = None
    else:
        worker_startup_wall_seconds = 0.0
        worker_run_wall_seconds = 0.0
        process_end_to_end_seconds = None
    parallel_seconds = time.perf_counter() - parallel_started
    executed = [
        index for indices, _ in indexed_results for index in indices
    ]
    if sorted(executed) != list(range(len(systems))):
        raise RuntimeError(
            "multi-GPU scheduling duplicated or omitted input systems"
        )
    result = _combine_relaxation_results(
        indexed_results,
        workload_size=len(systems),
        calculator=calculator,
    )
    result.metadata["scheduling"] = {
        "policy": "auto",
        "decision": "online_autotune_multi_gpu",
        "devices": [str(device) for device in devices],
        "gpu_count": len(devices),
        "active_gpu_count": worker_count,
        "fingerprint": plan.fingerprint,
        "cache_path": str(scheduler.cache.path),
        "profiling_seconds": plan.profiling_seconds,
        "worker_backend_requested": requested_backend,
        "worker_backend": selected_backend,
        "worker_backend_fallback_reason": fallback_reason,
        "worker_preparation_seconds": preparation_seconds,
        "worker_startup_wall_seconds": worker_startup_wall_seconds,
        "worker_run_wall_seconds": worker_run_wall_seconds,
        "worker_process_end_to_end_seconds": process_end_to_end_seconds,
        "cold_start": cold_records,
        "planned_chunks": planned_chunks,
        "workers": sorted(
            worker_records,
            key=lambda record: record["worker_id"],
        ),
        "parallel_seconds": parallel_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "pending_work_stealing": True,
        "recommended_worker_mode": "tensor",
        "mps_requires_external_dispatch": False,
    }
    return result


def molecular_dynamics(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    ensemble: Literal[
        "nve", "nvt", "nvt_langevin", "npt", "npt_mtk"
    ] = "nve",
    **md_kwargs: Any,
) -> MDResult:
    """Run fixed-cell batch MD with the same calculator used for relaxation."""

    state = calculator.create_state(_normalize_systems(systems))
    if ensemble == "nve":
        return batched_velocity_verlet(state, calculator, **md_kwargs)
    if ensemble in ("nvt", "nvt_langevin"):
        return batched_langevin_baoab(state, calculator, **md_kwargs)
    if ensemble in ("npt", "npt_mtk"):
        raise NotImplementedError(
            "the NPT API slot is reserved; no validated batch barostat is implemented"
        )
    raise ValueError(f"unsupported ensemble {ensemble!r}")
