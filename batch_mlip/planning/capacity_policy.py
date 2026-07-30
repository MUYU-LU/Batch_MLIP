"""Signed, exact-contract hardware-capacity selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..core.calculator import BatchCalculator
from .auto import AutoSchedulerConfig
from .profiles import (
    HardwareBoundCostModel,
    HardwareCostProfile,
    LayeredCostCoefficients,
    PlanningProfileBundle,
)
from .refill_policy import model_state_sha256

_POLICY_PATH = Path(__file__).with_name("data") / "capacity_policy_v1.json"
_VALIDATED_MINIMUM_MEMORY_GROWTH_MARGIN = 1.10


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _qualified_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _optimizer_options(
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = getattr(optimizer, "options", {})
    return {**dict(defaults), **dict(optimizer_kwargs)}


def _normalized_dtype(value: object) -> str:
    label = str(value)
    return label if label.startswith("torch.") else f"torch.{label}"


@dataclass(frozen=True)
class HardwareCapacityPolicy:
    """One signed memory model plus its exact execution contract."""

    policy_id: str
    source_calibration_sha256: str
    model_name: str
    contract: dict[str, Any]
    model: HardwareBoundCostModel
    policy_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported hardware-capacity policy schema")
        if not self.policy_id or not self.model_name:
            raise ValueError("hardware-capacity policy identity is required")
        if len(self.source_calibration_sha256) != 64:
            raise ValueError("hardware-capacity policy requires a calibration hash")
        if self.model.metric != "bytes":
            raise ValueError("hardware-capacity policy requires a byte model")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "source_calibration_sha256": self.source_calibration_sha256,
            "model_name": self.model_name,
            "contract": dict(self.contract),
            "model": asdict(self.model),
        }

    def calculate_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def verify(self) -> None:
        if len(self.policy_sha256) != 64:
            raise ValueError("hardware-capacity policy is not sealed")
        if self.policy_sha256 != self.calculate_sha256():
            raise ValueError("hardware-capacity policy content hash does not match")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HardwareCapacityPolicy:
        values = dict(payload)
        model_values = values.pop("model")
        model = HardwareBoundCostModel(
            contract_id=model_values["contract_id"],
            metric=model_values["metric"],
            hardware=HardwareCostProfile(**model_values["hardware"]),
            coefficients=LayeredCostCoefficients(
                **model_values["coefficients"]
            ),
            safety_factor=float(model_values["safety_factor"]),
        )
        policy = cls(model=model, **values)
        policy.verify()
        return policy


@dataclass(frozen=True)
class HardwareCapacityDecision:
    """Result of matching one runtime against an offline capacity policy."""

    mode: str
    reason: str
    policy: HardwareCapacityPolicy | None = None

    @property
    def use_offline_model(self) -> bool:
        return self.mode == "offline_hardware_model" and self.policy is not None

    def to_dict(self) -> dict[str, Any]:
        policy = self.policy
        return {
            "mode": self.mode,
            "reason": self.reason,
            "policy_id": None if policy is None else policy.policy_id,
            "policy_sha256": None if policy is None else policy.policy_sha256,
            "source_calibration_sha256": (
                None if policy is None else policy.source_calibration_sha256
            ),
            "cost_model_contract_id": (
                None if policy is None else policy.model.contract_id
            ),
            "memory_model": None if policy is None else policy.model_name,
        }


@lru_cache(maxsize=1)
def _load_packaged_hardware_capacity_policy() -> HardwareCapacityPolicy:
    return HardwareCapacityPolicy.from_dict(
        json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    )


def load_hardware_capacity_policy(
    path: str | Path | None = None,
) -> HardwareCapacityPolicy:
    """Load and verify a packaged or explicitly supplied capacity policy."""

    if path is None:
        return _load_packaged_hardware_capacity_policy()
    return HardwareCapacityPolicy.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _fallback(
    reason: str,
    policy: HardwareCapacityPolicy | None,
) -> HardwareCapacityDecision:
    return HardwareCapacityDecision(
        mode="representative_probe_fallback",
        reason=reason,
        policy=policy,
    )


def select_hardware_capacity_policy(
    policy: HardwareCapacityPolicy,
    planning_profile: PlanningProfileBundle,
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
    devices: Sequence[torch.device],
    config: AutoSchedulerConfig,
    *,
    allocator_policy: str,
) -> HardwareCapacityDecision:
    """Use an offline model only when its full execution contract matches."""

    contract = policy.contract
    if (
        config.memory_growth_margin
        < _VALIDATED_MINIMUM_MEMORY_GROWTH_MARGIN
    ):
        return _fallback(
            "memory growth margin is below the validated minimum",
            policy,
        )
    systems = planning_profile.systems
    model_ids = {system.mlip_graph.model_id for system in systems}
    if model_ids != {contract["model_id"]}:
        return _fallback("planning profile model identity differs", policy)
    if {
        system.mlip_graph.model_dtype for system in systems
    } != {contract["model_dtype"]}:
        return _fallback("planning profile model dtype differs", policy)
    if any(
        not math.isclose(system.mlip_graph.cutoff_A, contract["cutoff_A"])
        or system.mlip_graph.force_mode != contract["force_mode"]
        or not math.isclose(
            system.graph_execution.skin_A,
            contract["skin_A"],
        )
        or system.graph_execution.neighbor_backend
        != contract["neighbor_backend"]
        for system in systems
    ):
        return _fallback("planning profile graph contract differs", policy)
    if {
        system.task_auxiliary.state_dtype for system in systems
    } != {contract["optimizer_dtype"]}:
        return _fallback("planning profile optimizer dtype differs", policy)
    if any(
        not system.task_auxiliary.variable_cell
        or not system.task_auxiliary.stress_required
        or system.task_auxiliary.optimizer_state_kind != "dense"
        or not system.task_auxiliary.algorithm.endswith(
            f".{contract['optimizer_type']}"
        )
        or system.task_auxiliary.cell_method is None
        or not system.task_auxiliary.cell_method.endswith(
            f".{contract['cell_filter_type']}"
        )
        for system in systems
    ):
        return _fallback("planning profile numerical task differs", policy)

    if _qualified_name(calculator) != contract["calculator_type"]:
        return _fallback("calculator type differs", policy)
    model = getattr(calculator, "model", None)
    if not isinstance(model, torch.nn.Module):
        return _fallback("calculator has no hashable torch model", policy)
    if _qualified_name(model) != contract["model_type"]:
        return _fallback("model type differs", policy)
    if sum(parameter.numel() for parameter in model.parameters()) != contract[
        "model_parameter_count"
    ]:
        return _fallback("model parameter count differs", policy)
    if model_state_sha256(model) != contract["model_state_sha256"]:
        return _fallback("model state differs", policy)
    if model.training:
        return _fallback("model is not in evaluation mode", policy)
    if getattr(calculator, "e0_dict", None) != {}:
        return _fallback("E0 offsets differ", policy)
    if getattr(calculator, "model_call_kwargs", None) != {}:
        return _fallback("model call options differ", policy)

    if type(optimizer).__name__ != contract["optimizer_type"]:
        return _fallback("optimizer type differs", policy)
    options = _optimizer_options(optimizer, optimizer_kwargs)
    if _normalized_dtype(options.get("optimizer_dtype")) != contract[
        "optimizer_dtype"
    ]:
        return _fallback("optimizer dtype differs", policy)
    if options.get("linear_algebra_backend", "auto") != contract[
        "linear_algebra_backend"
    ]:
        return _fallback("BFGS linear-algebra backend differs", policy)
    cell_filter = options.get("cell_filter")
    if (
        cell_filter is None
        or type(cell_filter).__name__ != contract["cell_filter_type"]
        or float(getattr(cell_filter, "pressure_GPa", float("nan"))) != 0.0
        or getattr(cell_filter, "mask", None) is not None
        or getattr(cell_filter, "cell_factor", None) is not None
        or getattr(cell_filter, "hydrostatic_strain", None) is not False
    ):
        return _fallback("cell-filter contract differs", policy)

    if allocator_policy != contract["cuda_allocator"]:
        return _fallback("CUDA allocator policy differs", policy)
    if not torch.are_deterministic_algorithms_enabled():
        return _fallback("deterministic algorithms are not enabled", policy)
    if torch.get_num_threads() != contract["cpu_threads"]:
        return _fallback("CPU thread count differs", policy)
    if torch.__version__ != contract["torch"] or torch.version.cuda != contract[
        "cuda"
    ]:
        return _fallback("PyTorch or CUDA version differs", policy)
    if config.max_batch_size > contract["maximum_validated_batch_size"]:
        return _fallback("requested maximum batch exceeds calibration", policy)
    if not math.isclose(
        config.memory_safety_fraction,
        policy.model.hardware.memory_safety_fraction,
    ):
        return _fallback("memory safety fraction differs", policy)
    if not devices or any(device.type != "cuda" for device in devices):
        return _fallback("capacity policy requires CUDA devices", policy)
    for device in devices:
        index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
        properties = torch.cuda.get_device_properties(index)
        if (
            properties.name != policy.model.hardware.device_name
            or properties.total_memory
            != policy.model.hardware.total_memory_bytes
        ):
            return _fallback("GPU model or memory differs", policy)
    calibrated_budget = math.floor(
        policy.model.hardware.total_memory_bytes
        * policy.model.hardware.memory_safety_fraction
    )
    if (
        config.memory_budget_bytes is not None
        and config.memory_budget_bytes > calibrated_budget
    ):
        return _fallback("explicit memory budget exceeds calibration", policy)
    return HardwareCapacityDecision(
        mode="offline_hardware_model",
        reason="signed execution and hardware contracts matched",
        policy=policy,
    )
