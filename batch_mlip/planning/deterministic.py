"""Deterministic memory-bounded planning for production relaxations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .auto import AutoSchedulerConfig, AutoWorkloadPlan
from .memory import SystemProfile


@dataclass(frozen=True)
class DeterministicMemoryProbe:
    """One representative forward used to scale model and graph memory."""

    memory_budget_bytes: int | None
    baseline_allocated_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    probe_indices: tuple[int, ...]
    probe_model_work: int
    model_bytes_per_work: float


@dataclass(frozen=True)
class DeterministicRelaxationChunk:
    """A memory-safe production batch."""

    system_indices: tuple[int, ...]
    bucket_index: int
    predicted_peak_bytes: int | None
    estimated_cost: float


@dataclass(frozen=True)
class DeterministicRelaxationPlan:
    """Complete deterministic schedule for one homogeneous device class."""

    workload: AutoWorkloadPlan
    probe: DeterministicMemoryProbe
    chunks: tuple[DeterministicRelaxationChunk, ...]
    memory_fraction: float
    memory_growth_margin: float


def profile_model_work(profile: SystemProfile) -> int:
    """Approximate persistent model/graph work without optimizer state."""

    return max(1, 256 * profile.atom_count + 64 * profile.edge_count)


def optimizer_dof_bytes(
    profile: SystemProfile,
    optimizer: object,
    optimizer_kwargs: dict[str, Any],
    calculator_dtype: torch.dtype,
    *,
    dense_tensor_multiplier: float,
) -> int:
    """Reserve dense quasi-Newton matrices and linear-algebra workspace."""

    optimizer_name = type(optimizer).__name__.lower()
    if "bfgs" not in optimizer_name and "quasinewton" not in optimizer_name:
        return 0
    requested_dtype = optimizer_kwargs.get("optimizer_dtype")
    if isinstance(requested_dtype, str):
        dtype = getattr(torch, requested_dtype.removeprefix("torch."))
    elif isinstance(requested_dtype, torch.dtype):
        dtype = requested_dtype
    else:
        dtype = calculator_dtype
    itemsize = torch.empty((), dtype=dtype).element_size()
    return math.ceil(
        dense_tensor_multiplier * itemsize * profile.dof_squared
    )


def select_probe_indices(
    workload: AutoWorkloadPlan,
    *,
    probe_batch_size: int,
) -> tuple[int, ...]:
    """Select the largest model/graph systems for one conservative probe."""

    ordered = sorted(
        workload.profiles,
        key=lambda profile: (
            profile_model_work(profile),
            profile.dof_squared,
            -profile.index,
        ),
        reverse=True,
    )
    return tuple(
        profile.index for profile in ordered[: min(probe_batch_size, len(ordered))]
    )


def _incremental_bytes(
    profile: SystemProfile,
    *,
    probe: DeterministicMemoryProbe,
    optimizer: object,
    optimizer_kwargs: dict[str, Any],
    calculator_dtype: torch.dtype,
    config: AutoSchedulerConfig,
) -> int:
    model_bytes = probe.model_bytes_per_work * profile_model_work(profile)
    dense_bytes = optimizer_dof_bytes(
        profile,
        optimizer,
        optimizer_kwargs,
        calculator_dtype,
        dense_tensor_multiplier=config.dense_optimizer_tensor_multiplier,
    )
    return max(1, math.ceil(config.memory_growth_margin * (model_bytes + dense_bytes)))


def plan_deterministic_relaxation(
    workload: AutoWorkloadPlan,
    probe: DeterministicMemoryProbe,
    optimizer: object,
    optimizer_kwargs: dict[str, Any],
    calculator_dtype: torch.dtype,
    config: AutoSchedulerConfig,
) -> DeterministicRelaxationPlan:
    """Pack each cost bucket without timing or trial optimization."""

    profiles = {profile.index: profile for profile in workload.profiles}
    chunks: list[DeterministicRelaxationChunk] = []
    baseline = probe.baseline_allocated_bytes or 0
    for bucket_index, bucket in enumerate(workload.buckets):
        ordered = sorted(
            bucket.system_indices,
            key=lambda index: (
                _incremental_bytes(
                    profiles[index],
                    probe=probe,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    calculator_dtype=calculator_dtype,
                    config=config,
                ),
                -index,
            ),
            reverse=True,
        )
        pending: list[int] = []
        predicted = baseline
        for index in ordered:
            incremental = _incremental_bytes(
                profiles[index],
                probe=probe,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                calculator_dtype=calculator_dtype,
                config=config,
            )
            exceeds_memory = (
                probe.memory_budget_bytes is not None
                and pending
                and predicted + incremental > probe.memory_budget_bytes
            )
            exceeds_count = len(pending) >= config.max_batch_size
            if exceeds_memory or exceeds_count:
                chunks.append(
                    _make_chunk(
                        pending,
                        bucket_index=bucket_index,
                        predicted_peak_bytes=predicted,
                        profiles=profiles,
                    )
                )
                pending = []
                predicted = baseline
            if (
                probe.memory_budget_bytes is not None
                and predicted + incremental > probe.memory_budget_bytes
            ):
                raise MemoryError(
                    f"system {index} is predicted to require "
                    f"{predicted + incremental} bytes, exceeding the "
                    f"{probe.memory_budget_bytes}-byte device budget"
                )
            pending.append(index)
            predicted += incremental
        if pending:
            chunks.append(
                _make_chunk(
                    pending,
                    bucket_index=bucket_index,
                    predicted_peak_bytes=(
                        predicted if probe.memory_budget_bytes is not None else None
                    ),
                    profiles=profiles,
                )
            )
    executed = [index for chunk in chunks for index in chunk.system_indices]
    if sorted(executed) != list(range(len(workload.profiles))):
        raise RuntimeError("deterministic planning duplicated or omitted systems")
    return DeterministicRelaxationPlan(
        workload=workload,
        probe=probe,
        chunks=tuple(chunks),
        memory_fraction=config.memory_safety_fraction,
        memory_growth_margin=config.memory_growth_margin,
    )


def _make_chunk(
    indices: Sequence[int],
    *,
    bucket_index: int,
    predicted_peak_bytes: int | None,
    profiles: dict[int, SystemProfile],
) -> DeterministicRelaxationChunk:
    return DeterministicRelaxationChunk(
        system_indices=tuple(indices),
        bucket_index=bucket_index,
        predicted_peak_bytes=predicted_peak_bytes,
        estimated_cost=sum(
            profile_model_work(profiles[index])
            + math.sqrt(profiles[index].dof_squared)
            for index in indices
        ),
    )
