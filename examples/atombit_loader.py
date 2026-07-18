"""Example factories for full-model and state-dict AtomBit checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from atombit_batch.loaders import load_full_torch_model
from src.model import AtomBitModel
from src.utils import AtomBitConfig


def load_pickled_model(checkpoint: str, key: str | None = None):
    """Load a checkpoint that serialized the complete ``nn.Module``."""

    return load_full_torch_model(checkpoint, key=key, map_location="cpu")


def load_atombit_state_dict(
    checkpoint: str,
    model_config: str,
    state_dict_key: str | None = None,
    strict: bool = True,
):
    """Construct ``AtomBitModel`` from YAML and load a state dictionary."""

    config_payload = yaml.safe_load(Path(model_config).read_text(encoding="utf-8"))
    config = AtomBitConfig.from_dict(config_payload)
    model = AtomBitModel(config)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state_dict_key is not None:
        payload = payload[state_dict_key]
    elif isinstance(payload, dict):
        for candidate in ("state_dict", "model_state_dict"):
            if candidate in payload:
                payload = payload[candidate]
                break
    if not isinstance(payload, dict):
        raise TypeError("checkpoint does not contain a state dictionary")

    # Common DDP prefix cleanup.
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }
    model.load_state_dict(cleaned, strict=strict)
    return model
