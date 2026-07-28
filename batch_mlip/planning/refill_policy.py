"""Offline, evidence-matched refill selection for deterministic planning."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..core.calculator import BatchCalculator

_POLICY_PATH = Path(__file__).with_name("data") / "refill_policy_v2.json"


@dataclass(frozen=True)
class RefillPrediction:
    """One conservative offline refill decision."""

    mode: str
    reason: str
    policy_id: str | None = None
    matched_family: str | None = None
    predicted_speedup: float | None = None
    evidence_split: str | None = None

    @property
    def use_refill(self) -> bool:
        return self.mode == "refill"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "policy_id": self.policy_id,
            "matched_family": self.matched_family,
            "predicted_speedup": self.predicted_speedup,
            "evidence_split": self.evidence_split,
        }


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash tensor names, metadata, and values independent of serialization."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_refill_policy() -> dict[str, Any]:
    """Load the versioned package policy once."""

    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported refill policy schema")
    return payload


def _optimizer_options(
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = getattr(optimizer, "options", {})
    return {**dict(defaults), **dict(optimizer_kwargs)}


def _active(reason: str, policy_id: str | None = None) -> RefillPrediction:
    return RefillPrediction(
        mode="active",
        reason=reason,
        policy_id=policy_id,
    )


def _contract_matches(
    policy: dict[str, Any],
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
) -> tuple[bool, str]:
    contract = policy["contract"]
    model = getattr(calculator, "model", None)
    if not isinstance(model, torch.nn.Module):
        return False, "calculator has no hashable torch model"
    calculator_type = (
        f"{type(calculator).__module__}.{type(calculator).__qualname__}"
    )
    model_type = f"{type(model).__module__}.{type(model).__qualname__}"
    if calculator_type != contract["calculator_type"]:
        return False, "calculator type has no matched refill evidence"
    if model_type != contract["model_type"]:
        return False, "model type has no matched refill evidence"
    if sum(parameter.numel() for parameter in model.parameters()) != contract[
        "model_parameter_count"
    ]:
        return False, "model parameter count differs from refill evidence"
    if str(calculator.dtype) != contract["model_dtype"]:
        return False, "model dtype differs from refill evidence"
    if calculator.cutoff != contract["cutoff_A"] or calculator.skin != contract["skin_A"]:
        return False, "cutoff or skin differs from refill evidence"
    if getattr(calculator, "force_mode", None) != contract["force_mode"]:
        return False, "force mode differs from refill evidence"
    if getattr(calculator, "neighbor_backend", None) != contract["neighbor_backend"]:
        return False, "neighbor backend differs from refill evidence"
    if getattr(calculator, "e0_dict", None) != {}:
        return False, "E0 offsets differ from refill evidence"
    if getattr(calculator, "model_call_kwargs", None) != {}:
        return False, "model call options differ from refill evidence"
    if model.training:
        return False, "model is not in evaluation mode"
    if type(optimizer).__name__ != contract["optimizer_type"]:
        return False, "optimizer type differs from refill evidence"

    options = _optimizer_options(optimizer, optimizer_kwargs)
    cell_filter = options.get("cell_filter")
    if cell_filter is None or type(cell_filter).__name__ != contract["cell_filter_type"]:
        return False, "cell filter differs from refill evidence"
    pressure = getattr(cell_filter, "pressure_GPa", None)
    if (
        not isinstance(pressure, int | float)
        or float(pressure) != 0.0
        or getattr(cell_filter, "mask", None) is not None
        or getattr(cell_filter, "cell_factor", None) is not None
        or getattr(cell_filter, "hydrostatic_strain", None) is not False
    ):
        return False, "cell filter parameters differ from refill evidence"
    optimizer_dtype = options.get("optimizer_dtype")
    if isinstance(optimizer_dtype, torch.dtype):
        optimizer_dtype = str(optimizer_dtype)
    elif isinstance(optimizer_dtype, str):
        optimizer_dtype = (
            optimizer_dtype
            if optimizer_dtype.startswith("torch.")
            else f"torch.{optimizer_dtype}"
        )
    if optimizer_dtype != contract["optimizer_dtype"]:
        return False, "optimizer dtype differs from refill evidence"
    if options.get("linear_algebra_backend", "auto") != contract[
        "linear_algebra_backend"
    ]:
        return False, "linear algebra backend differs from refill evidence"
    if float(options.get("fmax", 0.05)) != contract["fmax_eV_per_A"]:
        return False, "force threshold differs from refill evidence"
    if int(options.get("max_steps", 1000)) != contract["max_steps"]:
        return False, "maximum steps differ from refill evidence"
    if float(options.get("alpha", 70.0)) != 70.0:
        return False, "BFGS alpha differs from refill evidence"
    if float(options.get("max_step", 0.2)) != 0.2:
        return False, "BFGS maximum step differs from refill evidence"
    if options.get("smax", 0.005) is not None:
        return False, "stress threshold differs from refill evidence"
    if options.get("active_compaction") is not True:
        return False, "active compaction differs from refill evidence"
    if not torch.are_deterministic_algorithms_enabled():
        return False, "deterministic algorithms are not enabled"
    if torch.get_num_threads() != contract["cpu_threads"]:
        return False, "CPU thread count differs from refill evidence"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract[
        "cublas_workspace_config"
    ]:
        return False, "cuBLAS deterministic workspace differs from refill evidence"

    device = calculator.device
    if device.type != "cuda":
        return False, "refill evidence is CUDA-only"
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    if properties.name != contract["gpu_name"]:
        return False, "GPU model differs from refill evidence"
    if properties.total_memory != contract["gpu_total_memory_bytes"]:
        return False, "GPU memory differs from refill evidence"
    if torch.__version__ != contract["torch"] or torch.version.cuda != contract["cuda"]:
        return False, "torch or CUDA version differs from refill evidence"
    allocator = contract["allocator"]
    if (
        os.environ.get("PYTORCH_ALLOC_CONF") != allocator
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != allocator
    ):
        return False, "CUDA allocator environment differs from refill evidence"
    if model_state_sha256(model) != contract["model_state_sha256"]:
        return False, "model state differs from refill evidence"
    return True, "execution contract matched"


def predict_refill(
    calculator: BatchCalculator,
    optimizer: object,
    optimizer_kwargs: Mapping[str, Any],
    *,
    pool_size: int,
    resident_capacity: int,
    mean_atom_count: float,
    atom_count_cv: float,
    mean_edge_count: float,
    edge_count_cv: float,
    homogeneous_atom_count: bool,
    predicted_peak_bytes: int | None,
    memory_budget_bytes: int | None,
) -> RefillPrediction:
    """Select refill only from scientifically valid matching evidence."""

    policy = load_refill_policy()
    policy_id = policy["policy_id"]
    if pool_size <= resident_capacity:
        return _active("pool fits in one resident batch", policy_id)
    if not homogeneous_atom_count:
        return _active("repack refill has no validated positive policy", policy_id)
    if (
        predicted_peak_bytes is not None
        and memory_budget_bytes is not None
        and predicted_peak_bytes > memory_budget_bytes
    ):
        return _active("predicted peak exceeds the memory budget", policy_id)
    matched, reason = _contract_matches(
        policy,
        calculator,
        optimizer,
        optimizer_kwargs,
    )
    if not matched:
        return _active(reason, policy_id)

    matching = policy["matching"]
    candidates = []
    for record in policy["records"]:
        if (
            record["pool_size"] != pool_size
            or record["resident_capacity"] != resident_capacity
            or not record["homogeneous_atom_count"]
            or record["storage"] != "slots"
        ):
            continue
        atom_error = abs(record["mean_atom_count"] - mean_atom_count) / max(
            1.0, mean_atom_count
        )
        edge_error = abs(record["mean_edge_count"] - mean_edge_count) / max(
            1.0, mean_edge_count
        )
        atom_cv_error = abs(record["atom_count_cv"] - atom_count_cv)
        edge_cv_error = abs(record["edge_count_cv"] - edge_count_cv)
        if (
            atom_error <= matching["atom_relative_tolerance"]
            and edge_error <= matching["edge_relative_tolerance"]
        ):
            distance = (
                atom_error * atom_error
                + edge_error * edge_error
                + atom_cv_error * atom_cv_error
                + edge_cv_error * edge_cv_error
            )
            candidates.append((distance, record))
    if not candidates:
        return _active("no workload descriptor evidence matched", policy_id)

    record = min(candidates, key=lambda item: item[0])[1]
    if record["selected_mode"] != "refill":
        return RefillPrediction(
            mode="active",
            reason="matched evidence did not pass every refill gate",
            policy_id=policy_id,
            matched_family=record["family"],
            predicted_speedup=record["measured_refill_speedup"],
            evidence_split=record["split"],
        )
    return RefillPrediction(
        mode="refill",
        reason="matched evidence passed speed, memory, convergence, and endpoint gates",
        policy_id=policy_id,
        matched_family=record["family"],
        predicted_speedup=record["measured_refill_speedup"],
        evidence_split=record["split"],
    )
