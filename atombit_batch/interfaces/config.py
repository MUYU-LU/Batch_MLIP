"""YAML configuration helpers for public interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("configuration root must be a mapping")
    config = dict(payload)
    version = int(config.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"unsupported schema_version={version}; expected 1")
    return config


def required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing required key {context}.{key}")
    return mapping[key]
