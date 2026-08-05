"""Model factory and checkpoint-loading utilities."""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch


def _install_legacy_atombit_config_alias() -> None:
    """Make checkpoints pickled with the original nested module importable."""

    from src.utils import AtomBitConfig

    module = types.ModuleType("src.utils.Utils")
    module.AtomBitConfig = AtomBitConfig
    sys.modules.setdefault("src.utils.Utils", module)


def load_atombit_training_checkpoint(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct AtomBit from its training checkpoint without benchmark code."""

    from src.model import AtomBitModel
    from src.utils import AtomBitConfig

    _install_legacy_atombit_config_alias()
    payload = torch.load(
        Path(checkpoint),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("AtomBit checkpoint must contain a mapping")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise KeyError(
            "AtomBit checkpoint must contain model_config and model_state_dict"
        )

    raw_config = payload["model_config"]
    config = (
        AtomBitConfig.from_dict(raw_config)
        if isinstance(raw_config, Mapping)
        else raw_config
    )
    if not isinstance(config, AtomBitConfig):
        raise TypeError("model_config must be an AtomBitConfig or mapping")

    raw_state = payload["model_state_dict"]
    if not isinstance(raw_state, Mapping):
        raise TypeError("model_state_dict must be a tensor mapping")
    state_dict = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in raw_state.items()
    }
    floating_dtypes = {
        value.dtype
        for value in state_dict.values()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    if len(floating_dtypes) != 1:
        raise ValueError(
            "checkpoint floating state tensors must use one consistent dtype; "
            f"found {sorted(map(str, floating_dtypes))}"
        )
    checkpoint_dtype = next(iter(floating_dtypes))
    model = AtomBitModel(config).to(dtype=checkpoint_dtype)
    model.load_state_dict(state_dict, strict=strict)
    metadata = {
        "epoch": payload.get("epoch"),
        "label_mode": payload.get("label_mode"),
        "precision_dtype": payload.get("precision_dtype"),
        "state_dtype": str(checkpoint_dtype),
        "state_tensor_count": len(state_dict),
    }
    return model, metadata


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
