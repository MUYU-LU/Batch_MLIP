#!/usr/bin/env python3
"""Merge float64-model ASE/native BFGS scaling artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_bfgs_mixed_scaling import summarize_point
from summarize_variable_cell_scaling import load, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--mixed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atom-counts", default="46,92,184,276")
    args = parser.parse_args()

    mixed = load(args.mixed_results)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "ratio_convention": "ase_seconds / active_seconds; above 1 is faster",
        "groups": {},
    }
    for atom_count in [int(value) for value in args.atom_counts.split(",")]:
        ase = load(args.raw_dir / f"ase_atoms{atom_count}.json")
        active = load(args.raw_dir / f"active_atoms{atom_count}.json")
        if ase["sample_files"] != active["sample_files"]:
            raise ValueError(f"sample pools differ for atom count {atom_count}")
        for artifact in (ase, active):
            parameters = artifact["parameters"]
            if not parameters["deterministic_algorithms"]:
                raise ValueError("artifact is not deterministic")
            if parameters["optimizer_dtype"] != "float64":
                raise ValueError("artifact does not use float64 optimizer state")
            if parameters["model_dtype"] != "float64":
                raise ValueError("artifact does not use a float64 model")

        ase_point = ase["points"][0]
        ase_summary = summarize_point(ase_point)
        mixed_group = mixed["groups"][str(atom_count)]
        mixed_points = {point["batch_size"]: point for point in mixed_group["points"]}
        ase_summary["float64_vs_mixed_time_ratio"] = (
            ase_summary["median_seconds"] / mixed_group["ase"]["median_seconds"]
        )
        ase_summary["float64_vs_mixed_memory_ratio"] = (
            ase_summary["peak_memory_bytes"] / mixed_group["ase"]["peak_memory_bytes"]
        )

        points = []
        for point in active["points"]:
            item = summarize_point(point)
            batch_size = point["batch_size"]
            mixed_point = mixed_points[batch_size]
            item["batch_size"] = batch_size
            item["speedup_vs_ase"] = (
                ase_summary["median_seconds"] / item["median_seconds"]
            )
            item["float64_vs_mixed_time_ratio"] = (
                item["median_seconds"] / mixed_point["median_seconds"]
            )
            item["float64_vs_mixed_memory_ratio"] = (
                item["peak_memory_bytes"] / mixed_point["peak_memory_bytes"]
            )
            item["float64_vs_mixed_time_per_graph_ratio"] = (
                item["median_seconds"]
                / item["graph_evaluations"]
                / (mixed_point["median_seconds"] / mixed_point["graph_evaluations"])
            )
            item["validation_vs_ase"] = validate(
                ase_point["records"], point["records"], atom_count
            )
            points.append(item)

        result["groups"][str(atom_count)] = {
            "sample_files": ase["sample_files"],
            "ase": ase_summary,
            "points": points,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
