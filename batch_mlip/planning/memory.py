"""Memory calibration and workload planning for batched simulations."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..core.neighbors import neighbor_list


@dataclass(frozen=True)
class CalibrationObservation:
    """Measured peak memory and aggregate work features for one batch."""

    atom_count: int
    edge_count: int
    dof_squared: int
    peak_memory_bytes: int

    @classmethod
    def homogeneous(
        cls,
        *,
        atoms_per_system: int,
        edges_per_system: int,
        batch_size: int,
        peak_memory_bytes: int,
        variable_cell: bool = True,
    ) -> CalibrationObservation:
        if atoms_per_system <= 0 or edges_per_system < 0 or batch_size <= 0:
            raise ValueError("calibration counts must be positive")
        dof = 3 * atoms_per_system + (9 if variable_cell else 0)
        return cls(
            atom_count=atoms_per_system * batch_size,
            edge_count=edges_per_system * batch_size,
            dof_squared=dof * dof * batch_size,
            peak_memory_bytes=peak_memory_bytes,
        )


@dataclass(frozen=True)
class MemoryCoefficients:
    """Non-negative byte coefficients for a batch peak-memory model."""

    fixed_bytes: float
    bytes_per_atom: float
    bytes_per_edge: float
    bytes_per_dof_squared: float

    def estimate(
        self, *, atom_count: int, edge_count: int, dof_squared: int
    ) -> int:
        predicted = (
            self.fixed_bytes
            + self.bytes_per_atom * atom_count
            + self.bytes_per_edge * edge_count
            + self.bytes_per_dof_squared * dof_squared
        )
        return max(0, math.ceil(predicted))


def _nonnegative_least_squares(
    features: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Solve a small non-negative least-squares problem by coordinate descent."""

    scales = np.linalg.norm(features, axis=0)
    if np.any(scales == 0.0):
        raise ValueError("calibration features must vary and be non-zero")
    normalized = features / scales
    coefficients = np.zeros(normalized.shape[1], dtype=np.float64)
    prediction = normalized @ coefficients
    for _ in range(20_000):
        previous = coefficients.copy()
        for column in range(normalized.shape[1]):
            values = normalized[:, column]
            residual = targets - prediction + values * coefficients[column]
            updated = max(0.0, float(values @ residual) / float(values @ values))
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        if np.max(np.abs(coefficients - previous)) <= 1e-10 * max(
            1.0, float(np.max(np.abs(coefficients)))
        ):
            break
    return coefficients / scales


def fit_memory_coefficients(
    observations: Sequence[CalibrationObservation],
    *,
    optimizer_itemsize: int = 8,
) -> MemoryCoefficients:
    """Fit a conservative non-negative model from measured CUDA peaks.

    One optimizer-state byte term per generalized Hessian element is reserved
    before fitting. The remaining non-negative coefficient absorbs additional
    eigensolver, model, graph, and framework storage.
    """

    rows = list(observations)
    if len(rows) < 4:
        raise ValueError("at least four calibration observations are required")
    if optimizer_itemsize <= 0:
        raise ValueError("optimizer_itemsize must be positive")
    features = np.asarray(
        [
            [1.0, row.atom_count, row.edge_count, row.dof_squared]
            for row in rows
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            row.peak_memory_bytes - optimizer_itemsize * row.dof_squared
            for row in rows
        ],
        dtype=np.float64,
    )
    if np.any(targets < 0.0):
        raise ValueError("measured peak is smaller than mandatory optimizer state")
    fitted = _nonnegative_least_squares(features, targets)
    return MemoryCoefficients(
        fixed_bytes=float(fitted[0]),
        bytes_per_atom=float(fitted[1]),
        bytes_per_edge=float(fitted[2]),
        bytes_per_dof_squared=float(fitted[3] + optimizer_itemsize),
    )


@dataclass(frozen=True)
class SystemProfile:
    """Static graph and optimizer work estimate for one input structure."""

    index: int
    atom_count: int
    edge_count: int
    dof_squared: int


@dataclass(frozen=True)
class PlannedBucket:
    """Compatible pending queue with a memory-safe resident capacity."""

    system_indices: tuple[int, ...]
    resident_capacity: int
    predicted_peak_bytes: int
    max_system_bytes: int


@dataclass(frozen=True)
class BatchPlan:
    """Planner output preserving every original input index exactly once."""

    profiles: tuple[SystemProfile, ...]
    buckets: tuple[PlannedBucket, ...]
    memory_budget_bytes: int
    profiling_seconds: float


class BatchPlanner:
    """Profile, bucket, and size independent simulation workloads."""

    def __init__(
        self,
        coefficients: MemoryCoefficients,
        *,
        memory_budget_bytes: int,
        max_batch_size: int | None = None,
        max_cost_ratio: float = 2.0,
        variable_cell: bool = True,
    ) -> None:
        if memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive or None")
        if max_cost_ratio < 1.0:
            raise ValueError("max_cost_ratio must be at least 1")
        self.coefficients = coefficients
        self.memory_budget_bytes = int(memory_budget_bytes)
        self.max_batch_size = max_batch_size
        self.max_cost_ratio = float(max_cost_ratio)
        self.variable_cell = variable_cell

    def profile_systems(
        self,
        systems: Sequence[Atoms],
        *,
        cutoff: float,
        skin: float = 0.0,
    ) -> tuple[SystemProfile, ...]:
        """Count candidate edges and generalized BFGS dimensions on CPU."""

        if cutoff <= 0.0 or skin < 0.0:
            raise ValueError("cutoff must be positive and skin non-negative")
        profiles = []
        for index, atoms in enumerate(systems):
            if not isinstance(atoms, Atoms) or len(atoms) == 0:
                raise TypeError("systems must contain non-empty ASE Atoms objects")
            centers = neighbor_list("i", atoms, cutoff + skin)
            dof = 3 * len(atoms) + (9 if self.variable_cell else 0)
            profiles.append(
                SystemProfile(
                    index=index,
                    atom_count=len(atoms),
                    edge_count=len(centers),
                    dof_squared=dof * dof,
                )
            )
        if not profiles:
            raise ValueError("systems must not be empty")
        return tuple(profiles)

    def _incremental_bytes(self, profile: SystemProfile) -> int:
        coefficients = self.coefficients
        return coefficients.estimate(
            atom_count=profile.atom_count,
            edge_count=profile.edge_count,
            dof_squared=profile.dof_squared,
        ) - math.ceil(coefficients.fixed_bytes)

    def _resident_capacity(
        self, profiles: Sequence[SystemProfile]
    ) -> tuple[int, int]:
        fixed = math.ceil(self.coefficients.fixed_bytes)
        running = fixed
        capacity = 0
        for profile in sorted(
            profiles, key=self._incremental_bytes, reverse=True
        ):
            if self.max_batch_size is not None and capacity >= self.max_batch_size:
                break
            candidate = running + self._incremental_bytes(profile)
            if candidate > self.memory_budget_bytes:
                break
            running = candidate
            capacity += 1
        if capacity == 0:
            raise MemoryError("one system exceeds the configured memory budget")
        return capacity, running

    def plan_profiles(
        self,
        profiles: Sequence[SystemProfile],
        *,
        profiling_seconds: float = 0.0,
    ) -> BatchPlan:
        """Group similar costs and assign each queue a resident capacity."""

        ordered = sorted(profiles, key=self._incremental_bytes, reverse=True)
        if not ordered:
            raise ValueError("profiles must not be empty")
        if len({profile.index for profile in ordered}) != len(ordered):
            raise ValueError("profile indices must be unique")

        groups: list[list[SystemProfile]] = []
        current: list[SystemProfile] = []
        largest = 0
        for profile in ordered:
            cost = max(1, self._incremental_bytes(profile))
            if current and largest / cost > self.max_cost_ratio:
                groups.append(current)
                current = []
            if not current:
                largest = cost
            current.append(profile)
        groups.append(current)

        buckets = []
        for group in groups:
            capacity, predicted = self._resident_capacity(group)
            buckets.append(
                PlannedBucket(
                    system_indices=tuple(sorted(item.index for item in group)),
                    resident_capacity=capacity,
                    predicted_peak_bytes=predicted,
                    max_system_bytes=max(
                        self._incremental_bytes(item) for item in group
                    ),
                )
            )
        return BatchPlan(
            profiles=tuple(sorted(ordered, key=lambda item: item.index)),
            buckets=tuple(buckets),
            memory_budget_bytes=self.memory_budget_bytes,
            profiling_seconds=profiling_seconds,
        )

    def plan(
        self,
        systems: Sequence[Atoms],
        *,
        cutoff: float,
        skin: float = 0.0,
    ) -> BatchPlan:
        """Profile and plan structures in one call."""

        started = time.perf_counter()
        profiles = self.profile_systems(systems, cutoff=cutoff, skin=skin)
        elapsed = time.perf_counter() - started
        return self.plan_profiles(profiles, profiling_seconds=elapsed)
