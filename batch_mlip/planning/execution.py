"""Memory-aware execution schedules for independent relaxations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ase import Atoms

from .memory import BatchPlan, BatchPlanner, SystemProfile


@dataclass(frozen=True)
class ScheduledRelaxationBatch:
    """One pending queue and its maximum resident system count."""

    system_indices: tuple[int, ...]
    resident_capacity: int
    active_refill: bool
    refill_storage: str = "repack"
    predicted_seconds: float | None = None


@dataclass(frozen=True)
class RelaxationSchedule:
    """Planner decision and executable queues for one input pool."""

    decision: str
    plan: BatchPlan
    batches: tuple[ScheduledRelaxationBatch, ...]
    total_predicted_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


def plan_relaxation_execution(
    planner: BatchPlanner,
    systems: Sequence[Atoms],
    *,
    cutoff: float,
    skin: float = 0.0,
    supports_refill: bool = False,
    system_profiles: Sequence[SystemProfile] | None = None,
) -> RelaxationSchedule:
    """Prefer one safe resident batch, otherwise execute planned queues."""

    if system_profiles is None:
        plan = planner.plan(systems, cutoff=cutoff, skin=skin)
    else:
        if len(system_profiles) != len(systems):
            raise ValueError(
                "system profile count differs from relaxation workload"
            )
        plan = planner.plan_profiles(system_profiles)
    predicted = planner.estimate_profiles_bytes(plan.profiles)
    within_count_limit = (
        planner.max_batch_size is None
        or len(systems) <= planner.max_batch_size
    )
    if within_count_limit and predicted <= planner.memory_budget_bytes:
        batches = (
            ScheduledRelaxationBatch(
                system_indices=tuple(range(len(systems))),
                resident_capacity=len(systems),
                active_refill=False,
            ),
        )
        decision = "whole_batch_predicted_to_fit"
    else:
        scheduled = []
        for bucket in plan.buckets:
            capacity = min(
                bucket.resident_capacity,
                len(bucket.system_indices),
            )
            if len(bucket.system_indices) <= capacity or supports_refill:
                scheduled.append(
                    ScheduledRelaxationBatch(
                        system_indices=bucket.system_indices,
                        resident_capacity=capacity,
                        active_refill=len(bucket.system_indices) > capacity,
                    )
                )
                continue
            for start in range(0, len(bucket.system_indices), capacity):
                indices = bucket.system_indices[start : start + capacity]
                scheduled.append(
                    ScheduledRelaxationBatch(
                        system_indices=indices,
                        resident_capacity=len(indices),
                        active_refill=False,
                    )
                )
        batches = tuple(scheduled)
        decision = "memory_safe_planned_queues"
    return RelaxationSchedule(
        decision=decision,
        plan=plan,
        batches=batches,
        total_predicted_bytes=predicted,
    )
