#!/usr/bin/env python3
"""Merge variable-cell ASE/masked/active raw benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

VALIDATION_TOLERANCES = {
    "max_energy_error_eV_per_atom": 1e-4,
    "max_final_fmax_error_eV_per_A": 0.03,
    "max_stress_tensor_error_eV_per_A3": 0.01,
    "max_position_rmsd_A": 0.02,
    "max_cell_rmsd_A": 0.02,
    "max_step_difference": 25,
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(reference: list[dict], candidate: list[dict], atom_count: int) -> dict:
    if [item["source"] for item in reference] != [
        item["source"] for item in candidate
    ]:
        raise ValueError("sample order differs")

    energy_per_atom = []
    force_errors = []
    stress_errors = []
    position_rmsd = []
    cell_rmsd = []
    step_errors = []
    convergence_matches = []
    for expected, actual in zip(reference, candidate, strict=True):
        energy_per_atom.append(
            abs(actual["energy_eV"] - expected["energy_eV"]) / atom_count
        )
        force_errors.append(
            abs(
                actual["max_force_eV_per_A"]
                - expected["max_force_eV_per_A"]
            )
        )
        stress_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(actual["stress_eV_per_A3"])
                        - np.asarray(expected["stress_eV_per_A3"])
                    )
                )
            )
        )
        position_rmsd.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(actual["positions_A"])
                            - np.asarray(expected["positions_A"])
                        )
                        ** 2
                    )
                )
            )
        )
        cell_rmsd.append(
            float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(actual["cell_A"])
                            - np.asarray(expected["cell_A"])
                        )
                        ** 2
                    )
                )
            )
        )
        step_errors.append(abs(actual["steps"] - expected["steps"]))
        convergence_matches.append(actual["converged"] == expected["converged"])

    result = {
        "reference_converged_count": sum(item["converged"] for item in reference),
        "candidate_converged_count": sum(item["converged"] for item in candidate),
        "convergence_flags_match": all(convergence_matches),
        "max_energy_error_eV_per_atom": max(energy_per_atom),
        "max_final_fmax_error_eV_per_A": max(force_errors),
        "max_stress_tensor_error_eV_per_A3": max(stress_errors),
        "max_position_rmsd_A": max(position_rmsd),
        "max_cell_rmsd_A": max(cell_rmsd),
        "max_step_difference": max(step_errors),
    }
    failed_checks = []
    if not result["convergence_flags_match"]:
        failed_checks.append("convergence_flags_match")
    failed_checks.extend(
        name
        for name, tolerance in VALIDATION_TOLERANCES.items()
        if result[name] > tolerance
    )
    result["tolerances"] = VALIDATION_TOLERANCES
    result["failed_checks"] = failed_checks
    result["passed"] = not failed_checks
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atom-counts", default="46,92,184,276")
    args = parser.parse_args()
    atom_counts = [int(value) for value in args.atom_counts.split(",")]

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "raw_dir": str(args.raw_dir),
        "groups": {},
    }
    for atom_count in atom_counts:
        raw = {
            method: load(args.raw_dir / f"{method}_atoms{atom_count}.json")
            for method in ("ase", "masked", "active")
        }
        if not (
            raw["ase"]["sample_files"]
            == raw["masked"]["sample_files"]
            == raw["active"]["sample_files"]
        ):
            raise ValueError(f"sample pools differ for atom count {atom_count}")

        ase_point = raw["ase"]["points"][0]
        ase_seconds = ase_point["timing"]["median_seconds"]
        masked_points = {point["batch_size"]: point for point in raw["masked"]["points"]}
        active_points = {point["batch_size"]: point for point in raw["active"]["points"]}
        points = []
        for batch_size in sorted(masked_points):
            masked = masked_points[batch_size]
            active = active_points[batch_size]
            item: dict[str, Any] = {"batch_size": batch_size}
            for method, point in (("masked", masked), ("active", active)):
                if point["status"] != "passed":
                    item[method] = {"status": point["status"], "error": point.get("error")}
                    continue
                seconds = point["timing"]["median_seconds"]
                item[method] = {
                    "status": "passed",
                    "median_seconds": seconds,
                    "timing_samples_seconds": point["timing"]["samples_seconds"],
                    "speedup_vs_ase": ase_seconds / seconds,
                    "systems_per_second": point["systems_per_second"],
                    "atoms_per_second": point["atoms_per_second"],
                    "peak_memory_bytes": point["peak_memory_bytes"],
                    "model_evaluations": point["model_evaluations"],
                    "graph_evaluations": point["graph_evaluations"],
                    "uncompacted_graph_evaluations": point[
                        "uncompacted_graph_evaluations"
                    ],
                    "avoided_graph_evaluations": point.get(
                        "avoided_graph_evaluations", 0
                    ),
                    "neighbor_rebuilds": point["neighbor_rebuilds"],
                    "converged_count": sum(
                        record["converged"] for record in point["records"]
                    ),
                    "validation": validate(
                        ase_point["records"], point["records"], atom_count
                    ),
                }
            if masked["status"] == "passed" and active["status"] == "passed":
                item["active_speedup_vs_masked"] = (
                    masked["timing"]["median_seconds"]
                    / active["timing"]["median_seconds"]
                )
            points.append(item)

        summary["groups"][str(atom_count)] = {
            "pool_size": raw["ase"]["pool_size"],
            "sample_files": raw["ase"]["sample_files"],
            "ase": {
                "median_seconds": ase_seconds,
                "timing_samples_seconds": ase_point["timing"]["samples_seconds"],
                "systems_per_second": ase_point["systems_per_second"],
                "atoms_per_second": ase_point["atoms_per_second"],
                "peak_memory_bytes": ase_point["peak_memory_bytes"],
                "model_evaluations": ase_point["model_evaluations"],
                "graph_evaluations": ase_point["graph_evaluations"],
                "neighbor_rebuilds": ase_point["neighbor_rebuilds"],
                "converged_count": sum(
                    record["converged"] for record in ase_point["records"]
                ),
            },
            "points": points,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
