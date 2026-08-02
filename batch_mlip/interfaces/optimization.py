"""One-call production interface for automatically planned relaxation pools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch
from ase import Atoms

from ..core.calculator import BatchCalculator
from ..core.types import RelaxationResult
from ..optimization.cell_filters import FrechetCellFilter
from ..optimization.registry import BatchedBFGS, BatchOptimizer
from ..planning.auto import AutoSchedulerConfig
from ..planning.capacity_policy import HardwareCapacityPolicy
from .executor import BatchExecutor

PoolPolicy = Literal["auto", "probe"] | HardwareCapacityPolicy | str | Path
CellFilterInput = Literal["frechet"] | FrechetCellFilter | None


def _resolve_cell_filter(cell_filter: CellFilterInput) -> FrechetCellFilter | None:
    if cell_filter is None or isinstance(cell_filter, FrechetCellFilter):
        return cell_filter
    if isinstance(cell_filter, str) and cell_filter.strip().lower() == "frechet":
        return FrechetCellFilter()
    raise ValueError("cell_filter must be None, 'frechet', or FrechetCellFilter")


def _policy_label(policy: PoolPolicy) -> str:
    if isinstance(policy, HardwareCapacityPolicy):
        return f"policy:{policy.policy_id}"
    return str(policy)


def optimize_pool(
    systems: Atoms | Sequence[Atoms],
    calculator: BatchCalculator,
    *,
    devices: Sequence[str | torch.device] | None = None,
    optimizer: str | BatchOptimizer = "bfgs",
    cell_filter: CellFilterInput = None,
    policy: PoolPolicy = "auto",
    auto_config: AutoSchedulerConfig | None = None,
    fmax: float = 0.01,
    max_steps: int = 1000,
    **optimizer_kwargs: Any,
) -> RelaxationResult:
    """Optimize an in-memory structure pool with automatic safe scheduling.

    ``policy='auto'`` uses a packaged signed capacity model only when its full
    model, numerical, software, and hardware contract matches. Otherwise the
    same call falls back to a representative memory probe. ``policy='probe'``
    explicitly requests that conservative fallback.
    """

    config = auto_config or AutoSchedulerConfig()
    if isinstance(policy, str) and policy == "probe":
        config = replace(config, offline_hardware_capacity_enabled=False)
        capacity_policy: HardwareCapacityPolicy | str | Path | None = None
    elif isinstance(policy, str) and policy == "auto":
        capacity_policy = None
    else:
        capacity_policy = policy

    resolved_filter = _resolve_cell_filter(cell_filter)
    options = dict(optimizer_kwargs)
    if (
        (isinstance(optimizer, str) and optimizer.strip().lower() == "bfgs")
        or isinstance(optimizer, BatchedBFGS)
    ):
        options.setdefault("optimizer_dtype", "float64")

    resolved_devices = (
        (calculator.device,) if devices is None else tuple(devices)
    )
    executor = BatchExecutor(
        calculator,
        devices=resolved_devices,
        auto_config=config,
        shutdown_timeout_seconds=30.0,
    )
    try:
        result = executor.relax(
            systems,
            optimizer=optimizer,
            hardware_capacity_policy=capacity_policy,
            cell_filter=resolved_filter,
            fmax=fmax,
            max_steps=max_steps,
            **options,
        )
    finally:
        executor.close()

    scheduling = result.metadata.get("scheduling", {})
    result.metadata["optimize_pool"] = {
        "schema_version": 1,
        "requested_policy": _policy_label(policy),
        "devices": [str(device) for device in resolved_devices],
        "capacity_planning": scheduling.get("capacity_planning"),
        "allocator": scheduling.get("allocator"),
        "executor_shutdown": executor.shutdown_metadata,
    }
    return result
