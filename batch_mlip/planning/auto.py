"""Cold-start and cached online scheduling for production relaxations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from ase import Atoms

try:
    import fcntl
except ImportError:  # pragma: no cover - project execution targets POSIX hosts
    fcntl = None

from ..core.calculator import BatchCalculator
from ..core.neighbors import neighbor_list
from .memory import SystemProfile


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AutoSchedulerConfig:
    """Controls deterministic scheduling and explicit experimental autotuning."""

    cache_path: str | Path | None = None
    cache_enabled: bool = True
    initial_batch_size: int = 1
    growth_factor: int = 4
    max_batch_size: int = 256
    max_cost_ratio: float = 2.0
    min_throughput_improvement: float = 0.05
    memory_safety_fraction: float = 0.85
    memory_growth_margin: float = 1.25
    memory_probe_batch_size: int = 4
    memory_budget_bytes: int | None = None
    dense_optimizer_tensor_multiplier: float = 16.0
    near_frontier_budget_fraction: float = 0.20
    near_frontier_growth_factor: int = 2
    stop_growth_budget_fraction: float = 0.65
    cache_atom_relative_tolerance: float = 0.10
    cache_edge_relative_tolerance: float = 0.25
    refill_occupancy_threshold: float = 0.65
    refill_min_pending_factor: float = 2.0
    refill_min_capacity: int = 8
    multi_gpu_cold_start_jobs: int = 32
    multi_gpu_worker_backend: Literal["auto", "process", "thread"] = "auto"
    multi_gpu_process_cpu_threads: int = 1
    multi_gpu_process_min_chunks_per_device: int = 8
    cuda_allocator_policy: Literal[
        "auto", "native", "expandable_segments"
    ] = "auto"

    def __post_init__(self) -> None:
        _positive_int("initial_batch_size", self.initial_batch_size)
        _positive_int("growth_factor", self.growth_factor)
        _positive_int("max_batch_size", self.max_batch_size)
        _positive_int("memory_probe_batch_size", self.memory_probe_batch_size)
        _positive_int("refill_min_capacity", self.refill_min_capacity)
        _positive_int(
            "multi_gpu_cold_start_jobs",
            self.multi_gpu_cold_start_jobs,
        )
        _positive_int(
            "multi_gpu_process_cpu_threads",
            self.multi_gpu_process_cpu_threads,
        )
        _positive_int(
            "multi_gpu_process_min_chunks_per_device",
            self.multi_gpu_process_min_chunks_per_device,
        )
        _positive_int(
            "near_frontier_growth_factor",
            self.near_frontier_growth_factor,
        )
        if self.growth_factor < 2:
            raise ValueError("growth_factor must be at least 2")
        if self.multi_gpu_worker_backend not in ("auto", "process", "thread"):
            raise ValueError(
                "multi_gpu_worker_backend must be 'auto', 'process', or 'thread'"
            )
        if self.cuda_allocator_policy not in (
            "auto",
            "native",
            "expandable_segments",
        ):
            raise ValueError(
                "cuda_allocator_policy must be 'auto', 'native', or "
                "'expandable_segments'"
            )
        if not math.isfinite(self.max_cost_ratio) or self.max_cost_ratio < 1.0:
            raise ValueError("max_cost_ratio must be at least 1")
        if (
            not math.isfinite(self.min_throughput_improvement)
            or self.min_throughput_improvement < 0.0
        ):
            raise ValueError("min_throughput_improvement must be non-negative")
        if (
            not math.isfinite(self.memory_safety_fraction)
            or not 0.0 < self.memory_safety_fraction < 1.0
        ):
            raise ValueError("memory_safety_fraction must be between zero and one")
        if (
            not math.isfinite(self.memory_growth_margin)
            or self.memory_growth_margin < 1.0
        ):
            raise ValueError("memory_growth_margin must be at least one")
        if self.memory_budget_bytes is not None:
            _positive_int("memory_budget_bytes", self.memory_budget_bytes)
        if (
            not math.isfinite(self.dense_optimizer_tensor_multiplier)
            or self.dense_optimizer_tensor_multiplier < 1.0
        ):
            raise ValueError(
                "dense_optimizer_tensor_multiplier must be at least one"
            )
        if (
            not math.isfinite(self.near_frontier_budget_fraction)
            or not 0.0 < self.near_frontier_budget_fraction < 1.0
        ):
            raise ValueError(
                "near_frontier_budget_fraction must be between zero and one"
            )
        if (
            not math.isfinite(self.stop_growth_budget_fraction)
            or not self.near_frontier_budget_fraction
            < self.stop_growth_budget_fraction
            < 1.0
        ):
            raise ValueError(
                "stop_growth_budget_fraction must be between "
                "near_frontier_budget_fraction and one"
            )
        for name, value in (
            ("cache_atom_relative_tolerance", self.cache_atom_relative_tolerance),
            ("cache_edge_relative_tolerance", self.cache_edge_relative_tolerance),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if (
            not math.isfinite(self.refill_occupancy_threshold)
            or not 0.0 < self.refill_occupancy_threshold <= 1.0
        ):
            raise ValueError(
                "refill_occupancy_threshold must be in the interval (0, 1]"
            )
        if (
            not math.isfinite(self.refill_min_pending_factor)
            or self.refill_min_pending_factor < 1.0
        ):
            raise ValueError("refill_min_pending_factor must be at least one")

    def resolved_cache_path(self) -> Path:
        if self.cache_path is not None:
            return Path(self.cache_path).expanduser()
        override = os.environ.get("BATCH_MLIP_AUTOSCHEDULER_CACHE")
        if override:
            return Path(override).expanduser()
        root = Path(
            os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
        ).expanduser()
        return root / "batch_mlip" / "autoscheduler-v1.json"


@dataclass(frozen=True)
class AutoWorkloadBucket:
    """Cost-compatible pending systems handled by one online controller."""

    system_indices: tuple[int, ...]
    mean_atom_count: float
    mean_edge_count: float
    mean_dof_squared: float
    homogeneous_atom_count: bool


@dataclass(frozen=True)
class AutoWorkloadPlan:
    """Profiled workload and stable execution fingerprint."""

    profiles: tuple[SystemProfile, ...]
    buckets: tuple[AutoWorkloadBucket, ...]
    profiling_seconds: float
    fingerprint: str
    fingerprint_fields: dict[str, Any]


@dataclass(frozen=True)
class AutoBatchAction:
    """The next production queue selected by an online controller."""

    system_count: int
    resident_capacity: int
    active_refill: bool = False
    refill_storage: str = "repack"


@dataclass(frozen=True)
class AutoBatchObservation:
    """Measured outcome from one completed production queue."""

    system_count: int
    resident_capacity: int
    active_refill: bool
    wall_seconds: float
    system_evaluations: int
    model_evaluations: int
    mean_active_occupancy: float
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    baseline_allocated_bytes: int | None = None
    baseline_reserved_bytes: int | None = None
    memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        _positive_int("system_count", self.system_count)
        _positive_int("resident_capacity", self.resident_capacity)
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0.0:
            raise ValueError("wall_seconds must be finite and positive")
        if self.system_evaluations <= 0 or self.model_evaluations <= 0:
            raise ValueError("evaluation counts must be positive")
        if not 0.0 < self.mean_active_occupancy <= 1.0:
            raise ValueError("mean_active_occupancy must be in (0, 1]")

    @property
    def throughput(self) -> float:
        """Return useful system-model evaluations per wall second."""

        return self.system_evaluations / self.wall_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_count": self.system_count,
            "resident_capacity": self.resident_capacity,
            "active_refill": self.active_refill,
            "wall_seconds": self.wall_seconds,
            "system_evaluations": self.system_evaluations,
            "model_evaluations": self.model_evaluations,
            "mean_active_occupancy": self.mean_active_occupancy,
            "throughput": self.throughput,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "baseline_reserved_bytes": self.baseline_reserved_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
        }


@dataclass(frozen=True)
class CachedAutoPolicy:
    """Reusable capacity decision for one workload regime."""

    fingerprint: str
    mean_atom_count: float
    mean_edge_count: float
    mean_dof_squared: float
    homogeneous_atom_count: bool
    resident_capacity: int
    capacity_source: str
    active_refill: bool
    refill_storage: str
    throughput: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    updated_unix_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "mean_atom_count": self.mean_atom_count,
            "mean_edge_count": self.mean_edge_count,
            "mean_dof_squared": self.mean_dof_squared,
            "homogeneous_atom_count": self.homogeneous_atom_count,
            "resident_capacity": self.resident_capacity,
            "capacity_source": self.capacity_source,
            "active_refill": self.active_refill,
            "refill_storage": self.refill_storage,
            "throughput": self.throughput,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "updated_unix_seconds": self.updated_unix_seconds,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CachedAutoPolicy:
        return cls(
            fingerprint=str(values["fingerprint"]),
            mean_atom_count=float(values["mean_atom_count"]),
            mean_edge_count=float(values["mean_edge_count"]),
            mean_dof_squared=float(values["mean_dof_squared"]),
            homogeneous_atom_count=bool(values["homogeneous_atom_count"]),
            resident_capacity=int(values["resident_capacity"]),
            capacity_source=str(
                values.get("capacity_source", "measured")
            ),
            active_refill=bool(values["active_refill"]),
            refill_storage=str(values["refill_storage"]),
            throughput=float(values["throughput"]),
            peak_allocated_bytes=(
                None
                if values.get("peak_allocated_bytes") is None
                else int(values["peak_allocated_bytes"])
            ),
            peak_reserved_bytes=(
                None
                if values.get("peak_reserved_bytes") is None
                else int(values["peak_reserved_bytes"])
            ),
            updated_unix_seconds=float(values["updated_unix_seconds"]),
        )


class AutoPolicyCache:
    """Small atomic JSON store for reusable scheduling decisions."""

    schema_version = 1

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.enabled = bool(enabled)

    def load(self) -> tuple[CachedAutoPolicy, ...]:
        if not self.enabled or not self.path.exists():
            return ()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.schema_version:
            raise ValueError(
                f"unsupported autoscheduler cache schema in {self.path}"
            )
        return tuple(
            CachedAutoPolicy.from_dict(item)
            for item in payload.get("policies", [])
        )

    @contextmanager
    def _update_lock(self):
        if fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def find(
        self,
        fingerprint: str,
        bucket: AutoWorkloadBucket,
        config: AutoSchedulerConfig,
    ) -> CachedAutoPolicy | None:
        candidates = []
        for policy in self.load():
            if (
                policy.fingerprint != fingerprint
                or policy.homogeneous_atom_count
                != bucket.homogeneous_atom_count
            ):
                continue
            atom_error = abs(
                policy.mean_atom_count - bucket.mean_atom_count
            ) / max(1.0, bucket.mean_atom_count)
            edge_error = abs(
                policy.mean_edge_count - bucket.mean_edge_count
            ) / max(1.0, bucket.mean_edge_count)
            if (
                atom_error <= config.cache_atom_relative_tolerance
                and edge_error <= config.cache_edge_relative_tolerance
            ):
                candidates.append((atom_error * atom_error + edge_error * edge_error, policy))
        return None if not candidates else min(candidates, key=lambda item: item[0])[1]

    def update(self, policy: CachedAutoPolicy) -> None:
        if not self.enabled:
            return
        with self._update_lock():
            policies = list(self.load())
            retained = [
                item
                for item in policies
                if not (
                    item.fingerprint == policy.fingerprint
                    and item.homogeneous_atom_count
                    == policy.homogeneous_atom_count
                    and abs(item.mean_atom_count - policy.mean_atom_count)
                    / max(1.0, policy.mean_atom_count)
                    <= 0.02
                    and abs(item.mean_edge_count - policy.mean_edge_count)
                    / max(1.0, policy.mean_edge_count)
                    <= 0.05
                )
            ]
            retained.append(policy)
            payload = {
                "schema_version": self.schema_version,
                "policies": [
                    item.to_dict()
                    for item in sorted(
                        retained,
                        key=lambda item: (
                            item.fingerprint,
                            item.mean_atom_count,
                            item.mean_edge_count,
                        ),
                    )
                ],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(self.path)


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _optimizer_name(optimizer: object) -> str:
    return _type_name(optimizer)


def _model_fields(calculator: BatchCalculator) -> dict[str, Any]:
    model = getattr(calculator, "model", None)
    if model is None:
        return {"model_type": None, "parameter_count": 0}
    return {
        "model_type": _type_name(model),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def execution_fingerprint(
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Fingerprint performance-relevant model, optimizer, and hardware fields."""

    device = calculator.device
    hardware: dict[str, Any] = {"device_type": device.type}
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        hardware.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory": properties.total_memory,
                "gpu_compute_capability": (
                    properties.major,
                    properties.minor,
                ),
            }
        )
    fields = {
        "schema_version": 1,
        "calculator_type": _type_name(calculator),
        **_model_fields(calculator),
        "dtype": str(calculator.dtype),
        "cutoff": calculator.cutoff,
        "skin": calculator.skin,
        "neighbor_backend": calculator.neighbor_backend,
        "force_mode": getattr(calculator, "force_mode", None),
        "graph_mode": getattr(calculator, "graph_mode", None),
        "optimizer_type": _optimizer_name(optimizer),
        "variable_cell": optimizer_kwargs.get("cell_filter") is not None,
        "cell_filter_type": (
            None
            if optimizer_kwargs.get("cell_filter") is None
            else _type_name(optimizer_kwargs["cell_filter"])
        ),
        "torch_version": torch.__version__,
        "hardware": hardware,
    }
    serialized = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), fields


def _profile_cost(
    profile: SystemProfile,
    *,
    optimizer: object,
    dtype: torch.dtype,
) -> float:
    itemsize = torch.empty((), dtype=dtype).element_size()
    name = _optimizer_name(optimizer).lower()
    dense_state = 4 * itemsize * profile.dof_squared if "bfgs" in name else 0
    return max(
        1.0,
        dense_state
        + 256.0 * profile.atom_count
        + 64.0 * profile.edge_count,
    )


def profile_auto_workload(
    systems: Sequence[Atoms],
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
    config: AutoSchedulerConfig,
) -> AutoWorkloadPlan:
    """Profile topology once and group structures by relative execution cost."""

    if calculator.cutoff is None:
        raise ValueError("automatic scheduling requires a calculator cutoff")
    started = time.perf_counter()
    variable_cell = optimizer_kwargs.get("cell_filter") is not None
    profiles = []
    for index, atoms in enumerate(systems):
        centers = neighbor_list(
            "i",
            atoms,
            calculator.cutoff + calculator.skin,
        )
        dof = 3 * len(atoms) + (9 if variable_cell else 0)
        profiles.append(
            SystemProfile(
                index=index,
                atom_count=len(atoms),
                edge_count=len(centers),
                dof_squared=dof * dof,
            )
        )
    ordered = sorted(
        profiles,
        key=lambda profile: _profile_cost(
            profile,
            optimizer=optimizer,
            dtype=calculator.dtype,
        ),
        reverse=True,
    )
    groups: list[list[SystemProfile]] = []
    current: list[SystemProfile] = []
    largest = 0.0
    for profile in ordered:
        cost = _profile_cost(
            profile,
            optimizer=optimizer,
            dtype=calculator.dtype,
        )
        if current and largest / cost > config.max_cost_ratio:
            groups.append(current)
            current = []
        if not current:
            largest = cost
        current.append(profile)
    groups.append(current)
    buckets = tuple(
        AutoWorkloadBucket(
            system_indices=tuple(sorted(profile.index for profile in group)),
            mean_atom_count=sum(profile.atom_count for profile in group)
            / len(group),
            mean_edge_count=sum(profile.edge_count for profile in group)
            / len(group),
            mean_dof_squared=sum(profile.dof_squared for profile in group)
            / len(group),
            homogeneous_atom_count=(
                len({profile.atom_count for profile in group}) == 1
            ),
        )
        for group in groups
    )
    fingerprint, fields = execution_fingerprint(
        calculator,
        optimizer,
        optimizer_kwargs,
    )
    return AutoWorkloadPlan(
        profiles=tuple(sorted(profiles, key=lambda profile: profile.index)),
        buckets=buckets,
        profiling_seconds=time.perf_counter() - started,
        fingerprint=fingerprint,
        fingerprint_fields=fields,
    )


@dataclass
class OnlineCapacityController:
    """Select future production queues from completed-batch observations."""

    bucket: AutoWorkloadBucket
    fingerprint: str
    config: AutoSchedulerConfig
    cached_policy: CachedAutoPolicy | None
    supports_refill: bool
    observations: list[AutoBatchObservation] = field(default_factory=list)
    _next_capacity: int = field(init=False)
    _best_capacity: int = field(init=False)
    _best_throughput: float = field(init=False, default=0.0)
    _exploring: bool = field(init=False)
    _request_refill: bool = field(init=False, default=False)
    _refill_won: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.cached_policy is None:
            self._next_capacity = min(
                self.config.initial_batch_size,
                self.config.max_batch_size,
            )
            self._best_capacity = self._next_capacity
            self._exploring = True
        else:
            self._next_capacity = min(
                self.cached_policy.resident_capacity,
                self.config.max_batch_size,
            )
            self._best_capacity = self._next_capacity
            self._best_throughput = self.cached_policy.throughput
            self._exploring = (
                self.cached_policy.capacity_source == "memory_extrapolated"
            )
            self._request_refill = (
                self.cached_policy.active_refill and self.supports_refill
            )

    @property
    def cache_hit(self) -> bool:
        return self.cached_policy is not None

    def next_action(self, remaining: int) -> AutoBatchAction:
        _positive_int("remaining", remaining)
        capacity = min(self._next_capacity, remaining)
        refill = self._request_refill and remaining > capacity
        return AutoBatchAction(
            system_count=remaining if refill else capacity,
            resident_capacity=capacity,
            active_refill=refill,
            refill_storage=(
                "slots" if self.bucket.homogeneous_atom_count else "repack"
            ),
        )

    def _safe_capacity(self) -> int:
        limits = [self.config.max_batch_size]
        for observation in self.observations:
            if (
                observation.memory_budget_bytes is None
                or observation.peak_allocated_bytes is None
                or observation.peak_reserved_bytes is None
                or observation.baseline_allocated_bytes is None
                or observation.baseline_reserved_bytes is None
            ):
                continue
            allocated_increment = max(
                1,
                observation.peak_allocated_bytes
                - observation.baseline_allocated_bytes,
            )
            reserved_increment = max(
                1,
                observation.peak_reserved_bytes
                - observation.baseline_reserved_bytes,
            )
            per_system = max(
                allocated_increment,
                reserved_increment,
            ) / observation.system_count
            available = max(
                0,
                observation.memory_budget_bytes
                - max(
                    observation.baseline_allocated_bytes,
                    observation.baseline_reserved_bytes,
                ),
            )
            limits.append(
                max(
                    1,
                    math.floor(
                        available
                        / (per_system * self.config.memory_growth_margin)
                    ),
                )
            )
        return max(1, min(limits))

    def _growth_factor(self, observation: AutoBatchObservation) -> int:
        if (
            observation.memory_budget_bytes is not None
            and observation.peak_reserved_bytes is not None
            and observation.peak_reserved_bytes
            / observation.memory_budget_bytes
            >= self.config.near_frontier_budget_fraction
        ):
            return min(
                self.config.growth_factor,
                self.config.near_frontier_growth_factor,
            )
        return self.config.growth_factor

    def _at_memory_stop(self, observation: AutoBatchObservation) -> bool:
        return (
            observation.memory_budget_bytes is not None
            and observation.peak_reserved_bytes is not None
            and observation.peak_reserved_bytes
            / observation.memory_budget_bytes
            >= self.config.stop_growth_budget_fraction
        )

    def observe(self, observation: AutoBatchObservation, *, remaining: int) -> None:
        if remaining < 0:
            raise ValueError("remaining must be non-negative")
        self.observations.append(observation)
        if observation.active_refill:
            self._refill_won = (
                observation.throughput
                >= self._best_throughput
                * (1.0 + self.config.min_throughput_improvement)
            )
            if self._refill_won:
                self._best_throughput = observation.throughput
            return
        underfilled_target = (
            self._exploring
            and observation.resident_capacity < self._next_capacity
        )
        previous_best = self._best_throughput
        improvement = (
            math.inf
            if previous_best == 0.0
            else observation.throughput / previous_best - 1.0
        )
        if previous_best == 0.0 or (
            improvement >= self.config.min_throughput_improvement
        ):
            if observation.resident_capacity >= self._best_capacity:
                self._best_capacity = observation.resident_capacity
                self._best_throughput = observation.throughput
        elif self._exploring and not underfilled_target:
            self._exploring = False

        safe_capacity = self._safe_capacity()
        if self._at_memory_stop(observation):
            self._exploring = False
        grown = min(
            observation.resident_capacity
            * self._growth_factor(observation),
            safe_capacity,
            self.config.max_batch_size,
        )
        if remaining == 0:
            return
        if self._exploring and grown > observation.resident_capacity:
            self._next_capacity = grown
            return

        self._exploring = False
        self._next_capacity = min(self._best_capacity, safe_capacity)
        enough_pending = (
            remaining
            >= self._next_capacity
            * self.config.refill_min_pending_factor
        )
        self._request_refill = (
            self.supports_refill
            and self._next_capacity >= self.config.refill_min_capacity
            and enough_pending
            and observation.mean_active_occupancy
            < self.config.refill_occupancy_threshold
        )

    def cached_result(self) -> CachedAutoPolicy:
        if not self.observations:
            raise RuntimeError("cannot cache a controller without observations")
        best_observation = max(
            (
                observation
                for observation in self.observations
                if not observation.active_refill
            ),
            key=lambda observation: observation.throughput,
            default=self.observations[-1],
        )
        use_refill = self._refill_won
        extrapolate_capacity = self._exploring and not use_refill
        selected_capacity = (
            min(
                self._next_capacity,
                self._safe_capacity(),
                self.config.max_batch_size,
            )
            if extrapolate_capacity
            else self._best_capacity
        )
        selected = (
            self.observations[-1]
            if use_refill
            else best_observation
        )
        capacity_was_measured = any(
            observation.resident_capacity == selected_capacity
            for observation in self.observations
        )
        return CachedAutoPolicy(
            fingerprint=self.fingerprint,
            mean_atom_count=self.bucket.mean_atom_count,
            mean_edge_count=self.bucket.mean_edge_count,
            mean_dof_squared=self.bucket.mean_dof_squared,
            homogeneous_atom_count=self.bucket.homogeneous_atom_count,
            resident_capacity=selected_capacity,
            capacity_source=(
                "measured"
                if capacity_was_measured
                else "memory_extrapolated"
            ),
            active_refill=use_refill,
            refill_storage=(
                "slots" if self.bucket.homogeneous_atom_count else "repack"
            ),
            throughput=max(self._best_throughput, selected.throughput),
            peak_allocated_bytes=selected.peak_allocated_bytes,
            peak_reserved_bytes=selected.peak_reserved_bytes,
            updated_unix_seconds=time.time(),
        )


class AutoScheduler:
    """Own workload profiling, cache matching, and bucket controllers."""

    def __init__(
        self,
        calculator: BatchCalculator,
        optimizer: object,
        optimizer_kwargs: Mapping[str, Any],
        *,
        config: AutoSchedulerConfig | None = None,
        supports_refill: bool = False,
    ) -> None:
        self.calculator = calculator
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.config = config or AutoSchedulerConfig()
        self.supports_refill = supports_refill
        self.cache = AutoPolicyCache(
            self.config.resolved_cache_path(),
            enabled=self.config.cache_enabled,
        )

    def plan(self, systems: Sequence[Atoms]) -> AutoWorkloadPlan:
        return profile_auto_workload(
            systems,
            self.calculator,
            self.optimizer,
            self.optimizer_kwargs,
            self.config,
        )

    def controller(
        self,
        plan: AutoWorkloadPlan,
        bucket: AutoWorkloadBucket,
    ) -> OnlineCapacityController:
        cached = self.cache.find(
            plan.fingerprint,
            bucket,
            self.config,
        )
        return OnlineCapacityController(
            bucket=bucket,
            fingerprint=plan.fingerprint,
            config=self.config,
            cached_policy=cached,
            supports_refill=self.supports_refill,
        )

    def save(self, controller: OnlineCapacityController) -> None:
        self.cache.update(controller.cached_result())
