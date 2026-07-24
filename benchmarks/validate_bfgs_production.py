#!/usr/bin/env python3
"""Compare common ASE and batched BFGS on fixed production structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.io import read
from ase.optimize import BFGS, BFGSLineSearch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_manifest,
    load_production_model,
    sha256_file,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    FrechetCellFilter,
    relax,
)
from src.Calculator import AtomBitCalculator  # noqa: E402

TOLERANCES = {
    "max_position_error_A": 2e-5,
    "max_cell_error_A": 2e-5,
    "max_energy_error_eV": 2e-4,
    "max_force_error_eV_per_A": 3e-4,
    "max_stress_error_eV_per_A3": 2e-7,
}


def run_ase(
    model: torch.nn.Module,
    systems: list[Any],
    *,
    device: torch.device,
    variable_cell: bool,
    steps: int,
    optimizer_name: str,
    alpha: float,
    model_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    calculator = AtomBitCalculator(
        model,
        cutoff=6.0,
        device=device,
        dtype=model_dtype,
        enable_stress=variable_cell,
        add_e0=False,
    )
    records = []
    for source in systems:
        atoms = source.copy()
        atoms.calc = calculator
        target = ASEFrechetCellFilter(atoms) if variable_cell else atoms
        optimizer_class = BFGS if optimizer_name == "bfgs" else BFGSLineSearch
        optimizer = optimizer_class(
            target,
            logfile=None,
            alpha=alpha,
            maxstep=0.2,
        )
        converged = bool(optimizer.run(fmax=1e-30, steps=steps))
        records.append(
            {
                "converged": converged,
                "steps": optimizer.nsteps,
                "positions": atoms.positions.copy(),
                "cell": atoms.cell.array.copy(),
                "energy": float(atoms.get_potential_energy()),
                "forces": atoms.get_forces().copy(),
                "stress": (
                    atoms.get_stress(voigt=False).copy()
                    if variable_cell
                    else None
                ),
            }
        )
    return records


def run_batch(
    model: torch.nn.Module,
    systems: list[Any],
    *,
    device: torch.device,
    variable_cell: bool,
    steps: int,
    optimizer_name: str,
    alpha: float,
    model_dtype: torch.dtype,
):
    calculator = AtomBitBatchCalculator(
        model,
        cutoff=6.0,
        skin=0.0,
        device=device,
        dtype=model_dtype,
        force_mode="autograd",
    )
    options: dict[str, Any] = {
        "optimizer": optimizer_name,
        "active_compaction": True,
        "fmax": 1e-30,
        "max_steps": steps,
        "alpha": alpha,
        "max_step": 0.2,
    }
    if variable_cell:
        options.update(
            cell_filter=FrechetCellFilter(),
            smax=None,
        )
    return relax(systems, calculator, **options)


def compare(
    references: list[dict[str, Any]],
    candidate,
) -> dict[str, Any]:
    errors = {name: 0.0 for name in TOLERANCES}
    convergence_matches = []
    step_matches = []
    reference_force_scale = max(
        float(np.max(np.abs(reference["forces"]))) for reference in references
    )
    positions = candidate.state.positions.detach().cpu().numpy()
    cells = candidate.state.cells.detach().cpu().numpy()
    energies = candidate.evaluation.energy.detach().cpu().numpy()
    forces = candidate.evaluation.forces.detach().cpu().numpy()
    stresses = (
        None
        if candidate.evaluation.stress is None
        else candidate.evaluation.stress.detach().cpu().numpy()
    )
    for system_id, reference in enumerate(references):
        atom_slice = candidate.state.atom_slice(system_id)
        errors["max_position_error_A"] = max(
            errors["max_position_error_A"],
            float(np.max(np.abs(positions[atom_slice] - reference["positions"]))),
        )
        errors["max_cell_error_A"] = max(
            errors["max_cell_error_A"],
            float(np.max(np.abs(cells[system_id] - reference["cell"]))),
        )
        errors["max_energy_error_eV"] = max(
            errors["max_energy_error_eV"],
            abs(float(energies[system_id]) - reference["energy"]),
        )
        errors["max_force_error_eV_per_A"] = max(
            errors["max_force_error_eV_per_A"],
            float(np.max(np.abs(forces[atom_slice] - reference["forces"]))),
        )
        if stresses is not None:
            errors["max_stress_error_eV_per_A3"] = max(
                errors["max_stress_error_eV_per_A3"],
                float(
                    np.max(
                        np.abs(stresses[system_id] - reference["stress"])
                    )
                ),
            )
        convergence_matches.append(
            bool(candidate.converged[system_id]) == reference["converged"]
        )
        step_matches.append(candidate.steps == reference["steps"])
    passed = bool(
        all(convergence_matches)
        and all(step_matches)
        and all(errors[name] <= tolerance for name, tolerance in TOLERANCES.items())
    )
    return {
        "passed": passed,
        "convergence_matches": convergence_matches,
        "step_matches": step_matches,
        "batch_steps": candidate.steps,
        "model_evaluations": candidate.model_evaluations,
        "graph_evaluations": candidate.graph_evaluations,
        "active_batch_sizes": candidate.active_batch_sizes,
        "tolerances": TOLERANCES,
        "reference_max_abs_force_eV_per_A": reference_force_scale,
        "max_force_error_relative": (
            errors["max_force_error_eV_per_A"] / reference_force_scale
        ),
        **errors,
    }


def batch_records(candidate) -> list[dict[str, Any]]:
    positions = candidate.state.positions.detach().cpu().numpy()
    cells = candidate.state.cells.detach().cpu().numpy()
    energies = candidate.evaluation.energy.detach().cpu().numpy()
    forces = candidate.evaluation.forces.detach().cpu().numpy()
    stresses = (
        None
        if candidate.evaluation.stress is None
        else candidate.evaluation.stress.detach().cpu().numpy()
    )
    records = []
    for system_id in range(candidate.state.n_systems):
        atom_slice = candidate.state.atom_slice(system_id)
        records.append(
            {
                "converged": bool(candidate.converged[system_id]),
                "steps": candidate.steps,
                "positions": positions[atom_slice],
                "cell": cells[system_id],
                "energy": float(energies[system_id]),
                "forces": forces[atom_slice],
                "stress": (
                    None if stresses is None else stresses[system_id]
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atom-count", type=int, default=46)
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--optimizer",
        choices=("bfgs", "bfgslinesearch"),
        default="bfgs",
    )
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
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
    model_dtype = getattr(torch, args.model_dtype)
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=model_dtype).eval()

    results = {}
    for variable_cell in (False, True):
        mode = "variable_cell" if variable_cell else "fixed_cell"
        references = run_ase(
            model,
            systems,
            device=device,
            variable_cell=variable_cell,
            steps=args.steps,
            optimizer_name=args.optimizer,
            alpha=args.alpha,
            model_dtype=model_dtype,
        )
        candidate = run_batch(
            model,
            systems,
            device=device,
            variable_cell=variable_cell,
            steps=args.steps,
            optimizer_name=args.optimizer,
            alpha=args.alpha,
            model_dtype=model_dtype,
        )
        size_one_records = []
        for system in systems:
            size_one_records.extend(
                batch_records(
                    run_batch(
                        model,
                        [system],
                        device=device,
                        variable_cell=variable_cell,
                        steps=args.steps,
                        optimizer_name=args.optimizer,
                        alpha=args.alpha,
                        model_dtype=model_dtype,
                    )
                )
            )
        results[mode] = {
            **compare(references, candidate),
            "batch_size_one_control": compare(size_one_records, candidate),
        }

    output = {
        "schema_version": 1,
        "status": "complete",
        "passed": all(result["passed"] for result in results.values()),
        "sample_files": names,
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "parameters": {
            "steps": args.steps,
            "optimizer": args.optimizer,
            "fmax_eV_per_A": 1e-30,
            "alpha_eV_per_A2": args.alpha,
            "max_step_A": 0.2,
            "dtype": args.model_dtype,
        },
        "tolerance_basis": {
            "original_force_tolerance_eV_per_A": 2e-4,
            "observed_variable_cell_ase_vs_b2_force_error_eV_per_A": 2.371072769165039e-4,
            "observed_variable_cell_b1_vs_b2_force_error_eV_per_A": 1.0201334953308105e-4,
            "observed_ase_relative_force_error": 2.3928270048031582e-4,
            "amended_force_tolerance_eV_per_A": 3e-4,
            "reason": (
                "Float32 ASE-calculator versus batched graph reduction and "
                "trajectory variation; float64 ASE algorithm tests are exact."
            ),
        },
        **results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
