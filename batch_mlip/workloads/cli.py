"""YAML entry point for the controlled workload runner."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.calculator import BatchCalculator
from ..interfaces.config import load_yaml, required
from ..models.loaders import resolve_callable
from .runner import WorkloadExecutionResult, WorkloadRunSpec, execute_workload
from .schema import read_workload_manifest


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return dict(value)


def run_workload_config(path: str | Path) -> WorkloadExecutionResult:
    """Build a calculator and execute one signed workload from YAML."""

    config = load_yaml(path)
    workload = _mapping(required(config, "workload", "config"), "workload")
    calculator_config = _mapping(required(config, "calculator", "config"), "calculator")
    run_config = _mapping(required(config, "run", "config"), "run")
    output_config = _mapping(required(config, "output", "config"), "output")

    manifest = read_workload_manifest(required(workload, "manifest", "workload"))
    factory_name = str(required(calculator_config, "factory", "calculator"))
    kwargs = _mapping(calculator_config.get("kwargs", {}), "calculator.kwargs")
    calculator = resolve_callable(factory_name)(**kwargs)
    if not isinstance(calculator, BatchCalculator):
        raise TypeError(
            f"calculator factory {factory_name!r} returned "
            f"{type(calculator).__name__}, not BatchCalculator"
        )
    spec = WorkloadRunSpec(**run_config)
    return execute_workload(
        manifest,
        required(workload, "dataset_dir", "workload"),
        calculator,
        spec,
        output_dir=required(output_config, "directory", "output"),
        registry_path=output_config.get("registry"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    result = run_workload_config(args.config)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
