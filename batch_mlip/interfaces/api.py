"""Public structure-level API shared by relaxation and MD."""

from __future__ import annotations

from collections.abc import Sequence
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
from ..optimization.registry import BatchOptimizer, create_optimizer
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
    **optimizer_kwargs: Any,
) -> RelaxationResult:
    """Relax structures directly or through an automatic execution schedule.

    Supplying ``pilot`` activates optimizer- and workload-aware selection.
    Without it, ``scheduling="auto"`` retains the memory-only planner.
    """

    resolved = create_optimizer(optimizer) if isinstance(optimizer, str) else optimizer
    if not isinstance(resolved, BatchOptimizer):
        raise TypeError(
            "optimizer must be a registered name or implement BatchOptimizer"
        )
    normalized = _normalize_systems(systems)
    if scheduling == "auto":
        if planner is None:
            raise ValueError("scheduling='auto' requires a calibrated BatchPlanner")
        if calculator.cutoff is None:
            raise ValueError("automatic scheduling requires a calculator cutoff")
        if optimizer_kwargs.get("refill_batch_size") is not None:
            raise ValueError(
                "automatic scheduling controls refill_batch_size; do not set it"
            )
        capabilities = resolved.capabilities()
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
            result = _offload_relaxation_result(result)
            torch.cuda.empty_cache()
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
