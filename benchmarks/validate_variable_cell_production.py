#!/usr/bin/env python3
"""Validate native AtomBit stress and short variable-cell FIRE on fixed T2 samples."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import load_manifest, load_production_model  # noqa: E402

from atombit_batch import (  # noqa: E402
    BatchedFrechetCellFilter,
    BatchedPotential,
    batched_fire_relax,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument("--atom-count", type=int, default=46)
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/variable_cell_production_smoke.json")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = load_manifest(args.manifest, args.sample_count)
    names = manifest["samples"][str(args.atom_count)][: args.sample_count]
    systems = [read(args.dataset_dir / name) for name in names]
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    calculator = BatchedPotential(
        model,
        cutoff=6.0,
        skin=0.0,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )

    batch_state = calculator.create_state(systems)
    batch_evaluation = calculator(batch_state, compute_stress=True)
    single_energy = []
    single_forces = []
    single_stress = []
    for atoms in systems:
        evaluation = calculator(
            calculator.create_state([atoms]), compute_stress=True
        )
        single_energy.append(evaluation.energy[0])
        single_forces.append(evaluation.forces)
        single_stress.append(evaluation.stress[0])

    reference_energy = torch.stack(single_energy)
    reference_forces = torch.cat(single_forces)
    reference_stress = torch.stack(single_stress)
    errors = {
        "max_abs_energy_eV": float(
            (batch_evaluation.energy - reference_energy).abs().max().cpu()
        ),
        "max_abs_force_eV_per_A": float(
            (batch_evaluation.forces - reference_forces).abs().max().cpu()
        ),
        "max_abs_stress_eV_per_A3": float(
            (batch_evaluation.stress - reference_stress).abs().max().cpu()
        ),
    }

    initial_cells = batch_state.cells.clone()
    relaxation = batched_fire_relax(
        batch_state,
        calculator,
        cell_filter=BatchedFrechetCellFilter(hydrostatic_strain=True),
        active_compaction=True,
        # Fixed thresholds make the first manifest sample converge at step 0
        # while the second remains active, exercising real state compaction.
        fmax=7.0,
        smax=0.015,
        max_steps=2,
        dt_start=0.01,
        dt_max=0.01,
        max_step=0.01,
    )
    relaxation.state.assert_graph_integrity()
    determinants = torch.linalg.det(relaxation.state.cells)
    cell_change = torch.linalg.vector_norm(
        relaxation.state.cells - initial_cells, dim=(-2, -1)
    )
    passed = (
        errors["max_abs_energy_eV"] < 5e-5
        and errors["max_abs_force_eV_per_A"] < 5e-4
        and errors["max_abs_stress_eV_per_A3"] < 5e-5
        and bool(torch.isfinite(relaxation.state.cells).all())
        and bool((determinants > 0.0).all())
        and relaxation.converged_step.detach().cpu().tolist() == [0, -1]
        and relaxation.active_batch_sizes == (2, 1, 1)
        and relaxation.graph_evaluations == 4
        and float(cell_change[0].cpu()) == 0.0
        and float(cell_change[1].cpu()) > 0.0
    )

    result = {
        "passed": passed,
        "samples": names,
        "atom_count": args.atom_count,
        "sample_count": args.sample_count,
        "device": str(device),
        "dtype": "float32",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "errors": errors,
        "variable_cell": {
            "steps": relaxation.steps,
            "converged_step": relaxation.converged_step.detach().cpu().tolist(),
            "model_evaluations": relaxation.model_evaluations,
            "graph_evaluations": relaxation.graph_evaluations,
            "active_batch_sizes": list(relaxation.active_batch_sizes),
            "cell_change_frobenius_A": cell_change.detach().cpu().tolist(),
            "final_cell_determinant_A3": determinants.detach().cpu().tolist(),
            "max_force_eV_per_A": relaxation.max_force.detach().cpu().tolist(),
            "max_stress_eV_per_A3": relaxation.max_stress.detach().cpu().tolist(),
            "cross_system_edges": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
