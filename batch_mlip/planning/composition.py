"""Compose planner mechanisms into one inspectable relaxation decision."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.calculator import BatchCalculator
from .deterministic import DeterministicRelaxationPlan


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    numeric = [float(value) for value in values]
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(numeric) / len(numeric),
        "coefficient_of_variation": _coefficient_of_variation(numeric),
    }


def _type_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _refill_evidence(
    plan: DeterministicRelaxationPlan,
) -> tuple[str, list[str], list[str]]:
    predictions = [
        chunk.refill_prediction
        for chunk in plan.chunks
        if chunk.refill_prediction is not None
    ]
    reasons = sorted(
        {
            str(prediction["reason"])
            for prediction in predictions
            if prediction.get("reason") is not None
        }
    )
    families = sorted(
        {
            str(prediction["matched_family"])
            for prediction in predictions
            if prediction.get("matched_family") is not None
        }
    )
    if families:
        return "exact_offline_evidence", reasons, families
    if predictions:
        return "no_matching_offline_evidence", reasons, families
    return "not_evaluated", reasons, families


def compose_relaxation_policy_manifest(
    plan: DeterministicRelaxationPlan,
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
    *,
    fully_periodic: bool,
    available_devices: Sequence[str],
    active_device_count: int,
    execution_chunk_sizes: Sequence[int],
    execution_resident_capacities: Sequence[int],
    work_stealing: bool,
    outer_assignment: str | None = None,
    refill_fallback_reasons: Sequence[str] = (),
    observed_converged_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Describe how the abstract scheduler was instantiated for one pool."""

    if not available_devices:
        raise ValueError("available_devices must not be empty")
    if outer_assignment is not None and not outer_assignment:
        raise ValueError("outer_assignment must not be empty")
    if not 0 < active_device_count <= len(available_devices):
        raise ValueError(
            "active_device_count must be positive and no larger than "
            "the available device count"
        )
    if not execution_chunk_sizes or any(size <= 0 for size in execution_chunk_sizes):
        raise ValueError("execution_chunk_sizes must contain positive values")
    if (
        len(execution_resident_capacities) != len(execution_chunk_sizes)
        or any(capacity <= 0 for capacity in execution_resident_capacities)
    ):
        raise ValueError(
            "execution_resident_capacities must contain one positive value "
            "per execution chunk"
        )
    if sum(execution_chunk_sizes) != len(plan.workload.profiles):
        raise ValueError(
            "execution_chunk_sizes must cover every profiled system exactly once"
        )
    if any(
        capacity > size
        for capacity, size in zip(
            execution_resident_capacities,
            execution_chunk_sizes,
            strict=True,
        )
    ):
        raise ValueError("resident capacity cannot exceed its execution queue")

    profiles = plan.workload.profiles
    atom_counts = [profile.atom_count for profile in profiles]
    candidate_edge_counts = [profile.edge_count for profile in profiles]
    dof_squared = [profile.dof_squared for profile in profiles]
    layered = all(profile.bound_cost is not None for profile in profiles)
    active_edge_counts = [
        (
            profile.edge_count
            if profile.bound_cost is None
            else profile.bound_cost.mlip_graph.active_edge_count
        )
        for profile in profiles
    ]
    generalized_dimensions = [
        (
            round(math.sqrt(profile.dof_squared))
            if profile.bound_cost is None
            else profile.bound_cost.task_auxiliary.generalized_dimension
        )
        for profile in profiles
    ]
    dense_state_elements = [
        (
            profile.dof_squared
            if profile.bound_cost is None
            else profile.bound_cost.task_auxiliary.dense_state_elements
        )
        for profile in profiles
    ]
    dense_linear_algebra_work = [
        (
            0
            if profile.bound_cost is None
            else profile.bound_cost.task_auxiliary.dense_linear_algebra_work
        )
        for profile in profiles
    ]
    structurally_mixed = (
        len(set(atom_counts)) > 1
        or len(set(active_edge_counts)) > 1
        or len(set(candidate_edge_counts)) > 1
        or len(set(dof_squared)) > 1
    )
    resident_waves = sum(
        math.ceil(
            len(chunk.system_indices)
            / (chunk.resident_capacity or len(chunk.system_indices))
        )
        for chunk in plan.chunks
    )
    active_refill = any(chunk.active_refill for chunk in plan.chunks)
    evidence_source, refill_reasons, matched_families = _refill_evidence(plan)
    refill_reasons = sorted({*refill_reasons, *refill_fallback_reasons})
    if evidence_source == "not_evaluated" and refill_fallback_reasons:
        evidence_source = "policy_exclusion"
    variable_cell = optimizer_kwargs.get("cell_filter") is not None
    active_compaction = bool(optimizer_kwargs.get("active_compaction", False))
    duration_observation = None
    if observed_converged_steps is not None:
        if len(observed_converged_steps) != len(profiles):
            raise ValueError(
                "observed_converged_steps must match the profiled pool size"
            )
        completed = [step for step in observed_converged_steps if step >= 0]
        duration_observation = {
            "converged_systems": len(completed),
            "unconverged_systems": len(observed_converged_steps) - len(completed),
            "converged_step": (
                None if not completed else _distribution(completed)
            ),
        }
    if variable_cell and fully_periodic:
        task_kind = "periodic_variable_cell_relaxation"
    elif variable_cell:
        task_kind = "variable_cell_relaxation"
    else:
        task_kind = "fixed_cell_relaxation"

    return {
        "schema_version": 1,
        "application_domain": "runtime_unspecified",
        "task": {
            "kind": task_kind,
            "optimizer": _type_name(optimizer),
            "cell_filter": (
                None
                if optimizer_kwargs.get("cell_filter") is None
                else _type_name(optimizer_kwargs["cell_filter"])
            ),
            "stress_required": variable_cell,
            "horizon": "variable",
        },
        "profile": {
            "pool_size": len(profiles),
            "atom_count": _distribution(atom_counts),
            "candidate_edge_count": _distribution(candidate_edge_counts),
            "optimizer_dimension_squared": _distribution(dof_squared),
            "structurally_mixed": structurally_mixed,
            "layers": {
                "general_structure": {
                    "features": ["atom_count"],
                    "atom_count": _distribution(atom_counts),
                },
                "mlip_graph": {
                    "features": ["active_edge_count"],
                    "model_id": (
                        None
                        if not layered
                        else profiles[0].bound_cost.mlip_graph.model_id
                    ),
                    "cutoff_A": calculator.cutoff,
                    "force_mode": (
                        None
                        if not layered
                        else profiles[0].bound_cost.mlip_graph.force_mode
                    ),
                    "model_dtype": str(calculator.dtype),
                    "active_edge_count": _distribution(active_edge_counts),
                },
                "task_auxiliary": {
                    "features": [
                        "generalized_dimension",
                        "dense_state_elements",
                        "dense_linear_algebra_work",
                        "stress_required",
                        "variable_cell",
                        "cell_method",
                        "cell_degrees_of_freedom",
                        "state_dtype",
                    ],
                    "generalized_dimension": _distribution(
                        generalized_dimensions
                    ),
                    "dense_state_elements": _distribution(
                        dense_state_elements
                    ),
                    "dense_linear_algebra_work": _distribution(
                        dense_linear_algebra_work
                    ),
                    "stress_required": variable_cell,
                    "variable_cell": variable_cell,
                    "cell_method": (
                        None
                        if not layered
                        else profiles[0].bound_cost.task_auxiliary.cell_method
                    ),
                    "cell_degrees_of_freedom": (
                        9
                        if not layered and variable_cell
                        else (
                            0
                            if not layered
                            else profiles[
                                0
                            ].bound_cost.task_auxiliary.cell_degrees_of_freedom
                        )
                    ),
                    "state_dtype": (
                        None
                        if not layered
                        else profiles[0].bound_cost.task_auxiliary.state_dtype
                    ),
                },
                "graph_execution_policy": {
                    "features": [
                        "skin_A",
                        "candidate_edge_count",
                        "cache_enabled",
                        "neighbor_backend",
                    ],
                    "skin_A": calculator.skin,
                    "candidate_edge_count": _distribution(
                        candidate_edge_counts
                    ),
                    "cache_enabled": calculator.skin > 0.0,
                    "neighbor_backend": calculator.neighbor_backend,
                },
                "hardware_binding": {
                    "available_devices": list(available_devices),
                    "active_device_count": active_device_count,
                    "memory_budget_bytes": plan.probe.memory_budget_bytes,
                    "memory_safety_fraction": plan.memory_fraction,
                },
                "scalar_cost": {
                    "universal": False,
                    "legacy_projection_used": not layered,
                    "rule": (
                        "hardware-calibrated coefficients must bind the "
                        "separate layers"
                    ),
                },
            },
            "duration_variation": {
                "static_prediction": "unavailable",
                "online_handling": (
                    "active_compaction" if active_compaction else "masked_state"
                ),
                "refill_evidence_source": evidence_source,
                "matched_families": matched_families,
                "observed": duration_observation,
            },
        },
        "outer_scheduler": {
            "pool_regime": (
                "single_resident_wave"
                if resident_waves == 1
                else "multiple_resident_waves"
            ),
            "resident_wave_count": resident_waves,
            "system_mix": "mixed" if structurally_mixed else "unmixed",
            "cost_bucket_count": len(plan.workload.buckets),
            "bucketing": (
                "cost_compatible"
                if len(plan.workload.buckets) > 1
                else "single_cost_bucket"
            ),
            "available_devices": list(available_devices),
            "active_device_count": active_device_count,
            "execution_chunk_count": len(execution_chunk_sizes),
            "execution_chunk_sizes": list(execution_chunk_sizes),
            "assignment": (
                outer_assignment
                if outer_assignment is not None
                else (
                    "largest_cost_first_work_stealing"
                    if work_stealing
                    else "sequential_submitted_chunks"
                )
            ),
        },
        "inner_scheduler": {
            "memory_safety_fraction": plan.memory_fraction,
            "resident_capacities": sorted(set(execution_resident_capacities)),
            "graph": {
                "cutoff_A": calculator.cutoff,
                "skin_A": calculator.skin,
                "cache_enabled": calculator.skin > 0.0,
                "neighbor_backend": calculator.neighbor_backend,
                "backend_resolution": (
                    "adaptive_per_rebuild"
                    if calculator.neighbor_backend == "auto"
                    else "fixed_by_user_contract"
                ),
                "selection_source": "calculator_contract",
            },
            "compaction": "active" if active_compaction else "disabled",
            "queue_policy": "refill" if active_refill else "active_drain",
            "refill": {
                "enabled": active_refill,
                "evidence_source": evidence_source,
                "fallback_reasons": refill_reasons,
            },
        },
    }
