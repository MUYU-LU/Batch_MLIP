"""Model factory and checkpoint-loading utilities."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


def resolve_callable(spec: str) -> Callable[..., Any]:
    """Resolve ``'package.module:function'`` to a callable."""

    if ":" not in spec:
        raise ValueError("factory must use 'package.module:function' syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"{spec!r} does not resolve to a callable")
    return function


def build_model(factory: str, kwargs: Mapping[str, Any] | None = None) -> torch.nn.Module:
    model = resolve_callable(factory)(**dict(kwargs or {}))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model factory {factory!r} returned {type(model).__name__}, not nn.Module")
    return model


def load_full_torch_model(
    checkpoint: str | Path,
    *,
    key: str | None = None,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a checkpoint that contains a serialized ``nn.Module``.

    State-dict-only checkpoints require a project-specific factory because the
    architecture/configuration cannot be reconstructed generically.
    """

    payload = torch.load(
        Path(checkpoint),
        map_location=map_location,
        weights_only=False,
    )
    if key is not None:
        if not isinstance(payload, Mapping) or key not in payload:
            raise KeyError(f"checkpoint does not contain key {key!r}")
        payload = payload[key]
    elif isinstance(payload, Mapping):
        for candidate in ("model", "ema_model", "module"):
            if isinstance(payload.get(candidate), torch.nn.Module):
                payload = payload[candidate]
                break

    if not isinstance(payload, torch.nn.Module):
        raise TypeError(
            "checkpoint does not contain a serialized nn.Module. Supply a custom "
            "factory that constructs the architecture and loads its state_dict."
        )
    return payload


def load_e0(source: str | Path | Mapping[int, float] | None) -> dict[int, float]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return {int(k): float(v) for k, v in source.items()}

    path = Path(source)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, Mapping) and "e0_dict" in payload:
            payload = payload["e0_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError("E0 source must contain a mapping of atomic number to energy")
    return {int(k): float(v) for k, v in payload.items()}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    normalized = str(name).lower().replace("torch.", "")
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}; use float32 or float64") from exc


def infer_cutoff(model: torch.nn.Module, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    cfg = getattr(model, "cfg", None)
    cutoff = getattr(cfg, "cutoff", None)
    if cutoff is None:
        cutoff = getattr(model, "cutoff", None)
    if isinstance(cutoff, torch.Tensor):
        cutoff = cutoff.detach().cpu().item()
    if cutoff is None:
        raise ValueError("cutoff was not provided and could not be inferred from the model")
    return float(cutoff)
