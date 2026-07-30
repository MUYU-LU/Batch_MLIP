"""User-facing scheduling decisions shared by relaxation execution paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

SchedulingMode = Literal["single_batch", "auto", "autotune"]

_MANUAL_BATCH_OPTIONS = frozenset(
    {
        "refill_batch_size",
        "refill_policy",
        "refill_storage",
        "refill_interval",
        "refill_low_watermark",
        "refill_min_chunk",
        "refill_tail_compaction_threshold",
    }
)


def resolve_scheduling_mode(
    requested: SchedulingMode | None,
    *,
    has_devices: bool,
    has_cutoff: bool,
    has_planning_options: bool,
    optimizer_kwargs: Mapping[str, Any],
) -> SchedulingMode:
    """Choose the production default without overriding manual batch controls."""

    if requested is not None:
        return requested
    if has_devices or has_planning_options:
        return "auto"
    if any(name in optimizer_kwargs for name in _MANUAL_BATCH_OPTIONS):
        return "single_batch"
    return "auto" if has_cutoff else "single_batch"


def scheduling_summary(
    *,
    strategy: str,
    devices: Sequence[str],
    resident_capacities: Sequence[int],
    active_compaction: bool,
    active_refill: Sequence[bool],
    memory_fraction: float | None,
    work_stealing: bool,
    refill_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Return one compact, stable description of an execution schedule."""

    refill_flags = tuple(active_refill)
    if not refill_flags or not any(refill_flags):
        batch_mode = "active_drain"
    elif all(refill_flags):
        batch_mode = "refill"
    else:
        batch_mode = "hybrid"
    return {
        "strategy": strategy,
        "batch_mode": batch_mode,
        "devices": list(devices),
        "device_count": len(devices),
        "resident_capacities": sorted(set(resident_capacities)),
        "memory_fraction": memory_fraction,
        "active_compaction": active_compaction,
        "work_stealing": work_stealing,
        "refill_reasons": sorted(set(refill_reasons)),
    }
