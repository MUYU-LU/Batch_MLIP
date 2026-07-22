#!/usr/bin/env python3
"""Check AtomBit energy derivatives along atomic and cell-filter directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    write_result,
)

from batch_mlip import AtomBitBatchCalculator, FrechetCellFilter  # noqa: E402


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _load_geometry(structure: Path, record_json: Path | None) -> tuple[Any, str]:
    atoms = read(structure)
    label = "initial"
    if record_json is None:
        return atoms, label

    payload = json.loads(record_json.read_text(encoding="utf-8"))
    records = payload["points"][0]["records"]
    matching = [record for record in records if record["source"] == structure.name]
    if len(matching) != 1:
        raise ValueError(
            f"expected one record for {structure.name} in {record_json}, "
            f"found {len(matching)}"
        )
    record = matching[0]
    atoms.positions[:] = record["positions_A"]
    atoms.set_cell(record["cell_A"], scale_atoms=False)
    return atoms, record_json.stem


def _normalized_directions(
    atomic_forces: torch.Tensor,
    cell_forces: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    zero_atoms = torch.zeros_like(atomic_forces)
    zero_cell = torch.zeros_like(cell_forces)

    def normalize(
        atoms: torch.Tensor, cell: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.sqrt(torch.sum(atoms * atoms) + torch.sum(cell * cell))
        if not bool(torch.isfinite(norm)) or bool(norm <= 0.0):
            raise ValueError("cannot normalize a zero or non-finite direction")
        return atoms / norm, cell / norm

    return {
        "atomic": normalize(atomic_forces, zero_cell),
        "cell": normalize(zero_atoms, cell_forces),
        "combined": normalize(atomic_forces, cell_forces),
    }


def _evaluate_displacement(
    calculator: AtomBitBatchCalculator,
    atoms: Any,
    atomic_direction: torch.Tensor,
    cell_direction: torch.Tensor,
    displacement: float,
    *,
    rebuild_neighbors: bool,
) -> tuple[float, int, int]:
    state = calculator.create_state([atoms])
    cell_filter = FrechetCellFilter().bind(state, dtype=calculator.dtype)
    cell_filter.apply_displacement(
        state,
        atomic_direction * displacement,
        cell_direction.unsqueeze(0) * displacement,
    )
    evaluation = calculator(
        state,
        neighbor_policy="always" if rebuild_neighbors else "never",
        compute_stress=True,
    )
    physical_edges = int(state.as_model_data().edge_index.shape[1])
    return (
        float(evaluation.energy[0].item()),
        int(state.edge_index.shape[1]),
        physical_edges,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--record-json", type=Path)
    parser.add_argument(
        "--steps",
        type=_parse_float_list,
        default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(step <= 0.0 for step in args.steps):
        raise ValueError("finite-difference steps must be positive")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    atoms, geometry_label = _load_geometry(args.structure, args.record_json)
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=dtype).eval()
    calculator = AtomBitBatchCalculator(
        model,
        cutoff=args.cutoff,
        skin=args.skin,
        device=device,
        dtype=dtype,
        force_mode="autograd",
    )

    base_state = calculator.create_state([atoms])
    base_filter = FrechetCellFilter().bind(base_state, dtype=dtype)
    base_evaluation = calculator(base_state, compute_stress=True)
    atomic_forces, cell_forces = base_filter.generalized_forces(
        base_state, base_evaluation
    )
    directions = _normalized_directions(atomic_forces, cell_forces[0])

    result = {
        "schema_version": 1,
        "status": "running",
        "structure": args.structure.name,
        "geometry": geometry_label,
        "record_json": None if args.record_json is None else str(args.record_json),
        "dtype": args.dtype,
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "parameters": {
            "cutoff_A": args.cutoff,
            "skin_A": args.skin,
            "steps": args.steps,
            "direction_normalization": (
                "unit Euclidean norm in Frechet optimizer coordinates"
            ),
        },
        "base": {
            "energy_eV": float(base_evaluation.energy[0].item()),
            "max_force_eV_per_A": float(
                torch.linalg.vector_norm(base_evaluation.forces, dim=1).max().item()
            ),
            "max_abs_stress_eV_per_A3": float(
                base_evaluation.stress[0].abs().max().item()
            ),
            "candidate_edges": int(base_state.edge_index.shape[1]),
            "physical_edges": int(base_state.as_model_data().edge_index.shape[1]),
        },
        "directions": {},
    }
    write_result(args.output, result)

    for direction_name, (atom_direction, cell_direction) in directions.items():
        analytic = -float(
            (
                torch.sum(atomic_forces * atom_direction)
                + torch.sum(cell_forces[0] * cell_direction)
            ).item()
        )
        modes = {}
        for mode, rebuild_neighbors in (("cached", False), ("rebuilt", True)):
            points = []
            for step in args.steps:
                plus = _evaluate_displacement(
                    calculator,
                    atoms,
                    atom_direction,
                    cell_direction,
                    step,
                    rebuild_neighbors=rebuild_neighbors,
                )
                minus = _evaluate_displacement(
                    calculator,
                    atoms,
                    atom_direction,
                    cell_direction,
                    -step,
                    rebuild_neighbors=rebuild_neighbors,
                )
                finite_difference = (plus[0] - minus[0]) / (2.0 * step)
                absolute_error = abs(finite_difference - analytic)
                points.append(
                    {
                        "step": step,
                        "analytic_dE_dt": analytic,
                        "finite_difference_dE_dt": finite_difference,
                        "absolute_error": absolute_error,
                        "relative_error": absolute_error / max(abs(analytic), 1e-30),
                        "energy_plus_eV": plus[0],
                        "energy_minus_eV": minus[0],
                        "candidate_edges_plus": plus[1],
                        "candidate_edges_minus": minus[1],
                        "physical_edges_plus": plus[2],
                        "physical_edges_minus": minus[2],
                    }
                )
            modes[mode] = points
        result["directions"][direction_name] = modes
        write_result(args.output, result)

    result["status"] = "complete"
    write_result(args.output, result)


if __name__ == "__main__":
    main()
