#!/usr/bin/env python3
"""Merge deterministic mixed-precision ASE/native BFGS scaling artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_variable_cell_scaling import load, validate


def summarize_point(point: dict[str, Any]) -> dict[str, Any]:
    timing = point["timing"]
    return {
        "status": point["status"],
        "median_seconds": timing["median_seconds"],
        "timing_samples_seconds": timing["samples_seconds"],
        "systems_per_second": point["systems_per_second"],
        "atoms_per_second": point["atoms_per_second"],
        "peak_memory_bytes": point["peak_memory_bytes"],
        "model_evaluations": point["model_evaluations"],
        "graph_evaluations": point["graph_evaluations"],
        "avoided_graph_evaluations": point.get("avoided_graph_evaluations", 0),
        "neighbor_rebuilds": point["neighbor_rebuilds"],
        "optimizer_steps_total": point["optimizer_steps_total"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atom-counts", default="46,92,184,276")
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "ratio_convention": "ase_seconds / active_seconds; above 1 is faster",
        "groups": {},
    }
    for atom_count in [int(value) for value in args.atom_counts.split(",")]:
        ase = load(args.raw_dir / f"bfgs_ase_atoms{atom_count}.json")
        active = load(args.raw_dir / f"bfgs_active_atoms{atom_count}.json")
        if ase["sample_files"] != active["sample_files"]:
            raise ValueError(f"sample pools differ for atom count {atom_count}")
        for artifact in (ase, active):
            parameters = artifact["parameters"]
            if not parameters["deterministic_algorithms"]:
                raise ValueError("artifact is not deterministic")
            if parameters["optimizer_dtype"] != "float64":
                raise ValueError("artifact does not use float64 optimizer state")

        ase_point = ase["points"][0]
        if ase_point["status"] != "passed":
            raise ValueError(f"ASE point failed for atom count {atom_count}")
        ase_seconds = ase_point["timing"]["median_seconds"]
        points = []
        for point in active["points"]:
            item = summarize_point(point)
            item["batch_size"] = point["batch_size"]
            item["speedup_vs_ase"] = ase_seconds / item["median_seconds"]
            item["validation_vs_ase"] = validate(
                ase_point["records"], point["records"], atom_count
            )
            points.append(item)

        result["groups"][str(atom_count)] = {
            "sample_files": ase["sample_files"],
            "ase": summarize_point(ase_point),
            "points": points,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
