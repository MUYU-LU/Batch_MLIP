#!/usr/bin/env python3
"""Validate named and object optimizer dispatch on the production model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import load_manifest, load_production_model  # noqa: E402

from atombit_batch import (  # noqa: E402
    BatchedFIRE,
    BatchedFrechetCellFilter,
    BatchedPotential,
    relax,
)

TOLERANCES = {
    "max_position_error_A": 1e-6,
    "max_cell_error_A": 1e-6,
    "max_energy_error_eV": 1e-5,
    "max_force_error_eV_per_A": 5e-5,
    "max_stress_error_eV_per_A3": 5e-8,
}


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def compare(named, object_result) -> dict[str, Any]:
    position_error = max_abs(named.state.positions, object_result.state.positions)
    cell_error = max_abs(named.state.cells, object_result.state.cells)
    energy_error = max_abs(named.evaluation.energy, object_result.evaluation.energy)
    force_error = max_abs(named.evaluation.forces, object_result.evaluation.forces)
    if named.evaluation.stress is None:
        stress_error = None
    else:
        stress_error = max_abs(
            named.evaluation.stress, object_result.evaluation.stress
        )
    same = bool(
        torch.equal(named.converged, object_result.converged)
        and torch.equal(named.converged_step, object_result.converged_step)
    )
    active_sizes_equal = (
        named.active_batch_sizes == object_result.active_batch_sizes
    )
    errors = {
        "max_position_error_A": position_error,
        "max_cell_error_A": cell_error,
        "max_energy_error_eV": energy_error,
        "max_force_error_eV_per_A": force_error,
        "max_stress_error_eV_per_A3": stress_error,
    }
    passed = bool(
        same
        and active_sizes_equal
        and all(
            value is None or value <= TOLERANCES[name]
            for name, value in errors.items()
        )
    )
    return {
        "passed": passed,
        "convergence_tensors_equal": same,
        "steps": [named.steps, object_result.steps],
        "active_batch_sizes_equal": active_sizes_equal,
        "tolerances": TOLERANCES,
        **errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atom-count", type=int, default=46)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, args.pool_size)
    names = manifest["samples"][str(args.atom_count)][: args.pool_size]
    systems = [read(args.dataset_dir / name) for name in names]
    device = torch.device(args.device)
    model, _ = load_production_model(args.checkpoint)
    calculator = BatchedPotential(
        model,
        cutoff=6.0,
        skin=0.0,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )

    common = {
        "fmax": 1e-30,
        "max_steps": args.steps,
        "active_compaction": True,
    }
    fixed_named = relax(systems, calculator, optimizer="fire", **common)
    fixed_control = relax(systems, calculator, optimizer="fire", **common)
    fixed_object = relax(systems, calculator, optimizer=BatchedFIRE(), **common)

    variable_options = {
        **common,
        "cell_filter": BatchedFrechetCellFilter(),
        "smax": None,
    }
    variable_named = relax(
        systems, calculator, optimizer="fire", **variable_options
    )
    variable_control = relax(
        systems, calculator, optimizer="fire", **variable_options
    )
    variable_object = relax(
        systems,
        calculator,
        optimizer=BatchedFIRE(),
        **variable_options,
    )

    result = {
        "schema_version": 1,
        "status": "complete",
        "device": str(device),
        "dtype": "float32",
        "sample_files": names,
        "fixed_cell_same_path_control": compare(fixed_named, fixed_control),
        "fixed_cell": compare(fixed_named, fixed_object),
        "variable_cell_same_path_control": compare(
            variable_named, variable_control
        ),
        "variable_cell": compare(variable_named, variable_object),
    }
    result["passed"] = bool(
        result["fixed_cell"]["passed"]
        and result["variable_cell"]["passed"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
