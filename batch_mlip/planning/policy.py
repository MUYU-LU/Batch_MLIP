"""Task-aware policy for independent batched relaxations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ase import Atoms

from .execution import RelaxationSchedule, ScheduledRelaxationBatch
from .memory import BatchPlan, BatchPlanner, PlannedBucket, SystemProfile


@dataclass(frozen=True)
class BatchTimingPoint:
    """Measured wall time for one optimizer model evaluation."""

    batch_size: int
    seconds: float
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.seconds) or self.seconds <= 0.0:
            raise ValueError("seconds must be finite and positive")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes <= 0:
            raise ValueError("peak_memory_bytes must be positive or None")


@dataclass(frozen=True)
class PilotRegime:
    """Pilot observations for structures with similar graph cost."""

    atom_count: int
    edge_count: int
    sampled_steps: tuple[int, ...]
    timing_points: tuple[BatchTimingPoint, ...]
    label: str = ""
    mps_systems_per_second: float | None = None

    def __post_init__(self) -> None:
        if self.atom_count <= 0 or self.edge_count < 0:
            raise ValueError("pilot atom and edge counts must be non-negative")
        if not self.sampled_steps or any(step < 0 for step in self.sampled_steps):
            raise ValueError("sampled_steps must contain non-negative values")
        sizes = [point.batch_size for point in self.timing_points]
        if not sizes or sizes != sorted(set(sizes)):
            raise ValueError(
                "timing_points must have unique, increasing batch sizes"
            )
        if self.mps_systems_per_second is not None and (
            not math.isfinite(self.mps_systems_per_second)
            or self.mps_systems_per_second <= 0.0
        ):
            raise ValueError(
                "regime mps_systems_per_second must be positive or None"
            )


@dataclass(frozen=True)
class OptimizationPilot:
    """Reusable optimizer-specific measurements used by the scheduler."""

    optimizer: str
    regimes: tuple[PilotRegime, ...]
    mps_systems_per_second: float | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if not self.optimizer.strip():
            raise ValueError("optimizer must not be empty")
        if not self.regimes:
            raise ValueError("at least one pilot regime is required")
        if self.mps_systems_per_second is not None and (
            not math.isfinite(self.mps_systems_per_second)
            or self.mps_systems_per_second <= 0.0
        ):
            raise ValueError("mps_systems_per_second must be positive or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "optimizer": self.optimizer,
            "source": self.source,
            "mps_systems_per_second": self.mps_systems_per_second,
            "regimes": [
                {
                    "label": regime.label,
                    "atom_count": regime.atom_count,
                    "edge_count": regime.edge_count,
                    "mps_systems_per_second": (
                        regime.mps_systems_per_second
                    ),
                    "sampled_steps": list(regime.sampled_steps),
                    "timing_points": [
                        {
                            "batch_size": point.batch_size,
                            "seconds": point.seconds,
                            "peak_memory_bytes": point.peak_memory_bytes,
                        }
                        for point in regime.timing_points
                    ],
                }
                for regime in self.regimes
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OptimizationPilot:
        """Construct a validated pilot from serialized data."""

        return cls(
            optimizer=str(data["optimizer"]),
            source=str(data.get("source", "")),
            mps_systems_per_second=(
                None
                if data.get("mps_systems_per_second") is None
                else float(data["mps_systems_per_second"])
            ),
            regimes=tuple(
                PilotRegime(
                    label=str(regime.get("label", "")),
                    atom_count=int(regime["atom_count"]),
                    edge_count=int(regime["edge_count"]),
                    mps_systems_per_second=(
                        None
                        if regime.get("mps_systems_per_second") is None
                        else float(regime["mps_systems_per_second"])
                    ),
                    sampled_steps=tuple(
                        int(step) for step in regime["sampled_steps"]
                    ),
                    timing_points=tuple(
                        BatchTimingPoint(
                            batch_size=int(point["batch_size"]),
                            seconds=float(point["seconds"]),
                            peak_memory_bytes=(
                                None
                                if point.get("peak_memory_bytes") is None
                                else int(point["peak_memory_bytes"])
                            ),
                        )
                        for point in regime["timing_points"]
                    ),
                )
                for regime in data["regimes"]
            ),
        )


@dataclass(frozen=True)
class TaskAwarePolicy:
    """Selection thresholds and explicit external-execution boundary."""

    min_refill_speedup: float = 1.05
    min_tensor_speedup_over_mps: float = 1.05
    allow_mps_recommendation: bool = True
    allow_refill_regime_extrapolation: bool = False
    max_refill_atom_relative_error: float = 0.05
    max_refill_edge_relative_error: float = 0.25

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.min_refill_speedup)
            or self.min_refill_speedup < 1.0
        ):
            raise ValueError("min_refill_speedup must be at least 1")
        if (
            not math.isfinite(self.min_tensor_speedup_over_mps)
            or self.min_tensor_speedup_over_mps < 1.0
        ):
            raise ValueError("min_tensor_speedup_over_mps must be at least 1")
        if (
            not math.isfinite(self.max_refill_atom_relative_error)
            or self.max_refill_atom_relative_error < 0.0
        ):
            raise ValueError(
                "max_refill_atom_relative_error must be non-negative"
            )
        if (
            not math.isfinite(self.max_refill_edge_relative_error)
            or self.max_refill_edge_relative_error < 0.0
        ):
            raise ValueError(
                "max_refill_edge_relative_error must be non-negative"
            )


@dataclass(frozen=True)
class _Candidate:
    mode: Literal["drain", "refill"]
    capacity: int
    predicted_seconds: float


def _nearest_regime(
    profiles: Sequence[SystemProfile],
    regimes: Sequence[PilotRegime],
) -> PilotRegime:
    mean_atoms = sum(profile.atom_count for profile in profiles) / len(profiles)
    mean_edges = sum(profile.edge_count for profile in profiles) / len(profiles)

    def distance(regime: PilotRegime) -> float:
        atom_distance = math.log((mean_atoms + 1.0) / (regime.atom_count + 1.0))
        edge_distance = math.log((mean_edges + 1.0) / (regime.edge_count + 1.0))
        return atom_distance * atom_distance + edge_distance * edge_distance

    return min(regimes, key=distance)


def _canonical_optimizer_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    if normalized.startswith("batched"):
        normalized = normalized[7:]
    if normalized == "quasinewton":
        return "bfgslinesearch"
    return normalized


def _interpolated_seconds(
    timing_points: Sequence[BatchTimingPoint],
    batch_size: int,
) -> float:
    if batch_size <= 0:
        return 0.0
    points = tuple(timing_points)
    if batch_size <= points[0].batch_size:
        return points[0].seconds * batch_size / points[0].batch_size
    for left, right in zip(points, points[1:], strict=False):
        if batch_size <= right.batch_size:
            fraction = (batch_size - left.batch_size) / (
                right.batch_size - left.batch_size
            )
            return left.seconds + fraction * (right.seconds - left.seconds)
    raise ValueError("batch size exceeds the measured timing curve")


def _expanded_durations(sampled_steps: Sequence[int], count: int) -> list[int]:
    # One initial evaluation is required before every optimizer trajectory.
    return [sampled_steps[index % len(sampled_steps)] + 1 for index in range(count)]


def _drain_seconds(
    durations: Sequence[int],
    capacity: int,
    timing_points: Sequence[BatchTimingPoint],
) -> float:
    total = 0.0
    for start in range(0, len(durations), capacity):
        chunk = durations[start : start + capacity]
        for evaluation in range(max(chunk)):
            active = sum(duration > evaluation for duration in chunk)
            total += _interpolated_seconds(timing_points, active)
    return total


def _refill_seconds(
    durations: Sequence[int],
    capacity: int,
    timing_points: Sequence[BatchTimingPoint],
) -> float:
    # Immediate refill is equivalent to assigning the pending queue to the
    # first slot that becomes free, then timing the number of occupied slots.
    slot_loads = [0] * min(capacity, len(durations))
    for duration in durations:
        slot = min(range(len(slot_loads)), key=slot_loads.__getitem__)
        slot_loads[slot] += duration
    return sum(
        _interpolated_seconds(
            timing_points,
            sum(load > evaluation for load in slot_loads),
        )
        for evaluation in range(max(slot_loads))
    )


def _memory_safe_capacities(
    planner: BatchPlanner,
    bucket: PlannedBucket,
    regime: PilotRegime,
) -> tuple[int, ...]:
    maximum = min(bucket.resident_capacity, len(bucket.system_indices))
    capacities = {
        min(point.batch_size, maximum)
        for point in regime.timing_points
        if point.batch_size <= maximum
        and (
            point.peak_memory_bytes is None
            or point.peak_memory_bytes <= planner.memory_budget_bytes
        )
    }
    exact_maximum = next(
        (
            point
            for point in regime.timing_points
            if point.batch_size == maximum
        ),
        None,
    )
    if (
        maximum <= regime.timing_points[-1].batch_size
        and (
            exact_maximum is None
            or exact_maximum.peak_memory_bytes is None
            or exact_maximum.peak_memory_bytes
            <= planner.memory_budget_bytes
        )
    ):
        capacities.add(maximum)
    if not capacities:
        smallest = regime.timing_points[0]
        if smallest.batch_size > maximum:
            capacities.add(maximum)
        else:
            raise MemoryError(
                "no measured pilot batch fits the planner memory budget"
            )
    return tuple(sorted(capacities))


def _select_bucket_candidate(
    planner: BatchPlanner,
    bucket: PlannedBucket,
    regime: PilotRegime,
    policy: TaskAwarePolicy,
    *,
    supports_refill: bool,
    refill_evidence_matched: bool,
) -> tuple[_Candidate, list[dict[str, Any]]]:
    durations = _expanded_durations(
        regime.sampled_steps,
        len(bucket.system_indices),
    )
    candidates: list[_Candidate] = []
    records: list[dict[str, Any]] = []
    for capacity in _memory_safe_capacities(planner, bucket, regime):
        drain = _drain_seconds(durations, capacity, regime.timing_points)
        candidates.append(_Candidate("drain", capacity, drain))
        records.append(
            {
                "mode": "drain",
                "capacity": capacity,
                "predicted_seconds": drain,
                "eligible": True,
            }
        )
        if (
            supports_refill
            and refill_evidence_matched
            and len(durations) > capacity
        ):
            refill = _refill_seconds(durations, capacity, regime.timing_points)
            eligible = drain / refill >= policy.min_refill_speedup
            records.append(
                {
                    "mode": "refill",
                    "capacity": capacity,
                    "predicted_seconds": refill,
                    "eligible": eligible,
                    "speedup_over_same_capacity_drain": drain / refill,
                }
            )
            if eligible:
                candidates.append(_Candidate("refill", capacity, refill))
    return min(candidates, key=lambda candidate: candidate.predicted_seconds), records


def _bucket_profiles(
    plan: BatchPlan,
    bucket: PlannedBucket,
) -> tuple[SystemProfile, ...]:
    by_index = {profile.index: profile for profile in plan.profiles}
    return tuple(by_index[index] for index in bucket.system_indices)


def _refill_evidence_matches(
    profiles: Sequence[SystemProfile],
    regime: PilotRegime,
    policy: TaskAwarePolicy,
) -> bool:
    if policy.allow_refill_regime_extrapolation:
        return True
    mean_atoms = sum(profile.atom_count for profile in profiles) / len(profiles)
    mean_edges = sum(profile.edge_count for profile in profiles) / len(profiles)
    atom_error = abs(mean_atoms - regime.atom_count) / regime.atom_count
    edge_error = abs(mean_edges - regime.edge_count) / max(1, regime.edge_count)
    return (
        atom_error <= policy.max_refill_atom_relative_error
        and edge_error <= policy.max_refill_edge_relative_error
    )


def plan_task_aware_relaxation(
    planner: BatchPlanner,
    systems: Sequence[Atoms],
    *,
    cutoff: float,
    pilot: OptimizationPilot,
    policy: TaskAwarePolicy | None = None,
    skin: float = 0.0,
    supports_refill: bool = False,
    refill_storage: Literal["auto", "repack", "slots"] = "auto",
    optimizer_name: str | None = None,
    system_profiles: Sequence[SystemProfile] | None = None,
) -> RelaxationSchedule:
    """Select measured capacities and drain/refill behavior for each bucket.

    MPS remains an external worker mode. If the pilot predicts that it wins,
    the returned schedule records that recommendation while retaining a
    directly executable tensor schedule as the fallback.
    """

    resolved_policy = policy or TaskAwarePolicy()
    if (
        optimizer_name is not None
        and _canonical_optimizer_name(optimizer_name)
        != _canonical_optimizer_name(pilot.optimizer)
    ):
        raise ValueError(
            f"pilot optimizer {pilot.optimizer!r} does not match "
            f"requested optimizer {optimizer_name!r}"
        )
    if refill_storage not in {"auto", "repack", "slots"}:
        raise ValueError("refill_storage must be auto, repack, or slots")
    if system_profiles is None:
        plan = planner.plan(systems, cutoff=cutoff, skin=skin)
    else:
        if len(system_profiles) != len(systems):
            raise ValueError(
                "system_profiles must contain one entry per input system"
            )
        if sorted(profile.index for profile in system_profiles) != list(
            range(len(systems))
        ):
            raise ValueError(
                "system_profiles indices must cover the input order exactly"
            )
        plan = planner.plan_profiles(system_profiles)
    scheduled: list[ScheduledRelaxationBatch] = []
    bucket_records: list[dict[str, Any]] = []
    tensor_seconds = 0.0
    regime_mps_seconds = 0.0
    all_regimes_have_mps = True
    for bucket in plan.buckets:
        profiles = _bucket_profiles(plan, bucket)
        regime = _nearest_regime(profiles, pilot.regimes)
        refill_evidence_matched = _refill_evidence_matches(
            profiles,
            regime,
            resolved_policy,
        )
        if regime.mps_systems_per_second is None:
            all_regimes_have_mps = False
        else:
            regime_mps_seconds += (
                len(bucket.system_indices) / regime.mps_systems_per_second
            )
        selected, candidate_records = _select_bucket_candidate(
            planner,
            bucket,
            regime,
            resolved_policy,
            supports_refill=supports_refill,
            refill_evidence_matched=refill_evidence_matched,
        )
        tensor_seconds += selected.predicted_seconds
        homogeneous_atoms = len({profile.atom_count for profile in profiles}) == 1
        if refill_storage == "slots" and not homogeneous_atoms:
            raise ValueError(
                "refill_storage='slots' requires equal atom counts within a bucket"
            )
        selected_storage = (
            ("slots" if homogeneous_atoms else "repack")
            if refill_storage == "auto"
            else refill_storage
        )
        if selected.mode == "refill":
            scheduled.append(
                ScheduledRelaxationBatch(
                    system_indices=bucket.system_indices,
                    resident_capacity=selected.capacity,
                    active_refill=True,
                    refill_storage=selected_storage,
                    predicted_seconds=selected.predicted_seconds,
                )
            )
        else:
            for start in range(0, len(bucket.system_indices), selected.capacity):
                indices = bucket.system_indices[start : start + selected.capacity]
                fraction = len(indices) / len(bucket.system_indices)
                scheduled.append(
                    ScheduledRelaxationBatch(
                        system_indices=indices,
                        resident_capacity=len(indices),
                        active_refill=False,
                        refill_storage=selected_storage,
                        predicted_seconds=selected.predicted_seconds * fraction,
                    )
                )
        bucket_records.append(
            {
                "system_count": len(bucket.system_indices),
                "pilot_regime": regime.label,
                "refill_evidence_matched": refill_evidence_matched,
                "selected_mode": selected.mode,
                "selected_capacity": selected.capacity,
                "predicted_seconds": selected.predicted_seconds,
                "candidates": candidate_records,
            }
        )

    if all_regimes_have_mps:
        mps_seconds = regime_mps_seconds
    elif pilot.mps_systems_per_second is not None:
        mps_seconds = len(systems) / pilot.mps_systems_per_second
    else:
        mps_seconds = None
    recommend_mps = (
        resolved_policy.allow_mps_recommendation
        and mps_seconds is not None
        and tensor_seconds / mps_seconds
        >= resolved_policy.min_tensor_speedup_over_mps
    )
    return RelaxationSchedule(
        decision="task_aware_pilot_policy",
        plan=plan,
        batches=tuple(scheduled),
        total_predicted_bytes=planner.estimate_profiles_bytes(plan.profiles),
        metadata={
            "optimizer": pilot.optimizer,
            "pilot_source": pilot.source,
            "predicted_tensor_seconds": tensor_seconds,
            "predicted_mps_seconds": mps_seconds,
            "recommended_worker_mode": "mps" if recommend_mps else "tensor",
            "mps_requires_external_dispatch": recommend_mps,
            "bucket_decisions": bucket_records,
        },
    )
