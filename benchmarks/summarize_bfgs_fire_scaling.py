#!/usr/bin/env python3
"""Merge common ASE and active-batch BFGS/FIRE scaling artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_variable_cell_scaling import load, validate


def point_summary(point: dict[str, Any]) -> dict[str, Any]:
    if point["status"] != "passed":
        return {"status": point["status"], "error": point.get("error")}
    timing = point["timing"]
    return {
        "status": "passed",
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
    atom_counts = [int(value) for value in args.atom_counts.split(",")]

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "raw_dir": str(args.raw_dir),
        "ratio_convention": "reference_seconds / candidate_seconds; above 1 is faster",
        "groups": {},
    }
    for atom_count in atom_counts:
        raw = {
            f"{optimizer}_{method}": load(
                args.raw_dir / f"{optimizer}_{method}_atoms{atom_count}.json"
            )
            for optimizer in ("fire", "bfgs")
            for method in ("ase", "active")
        }
        sample_pools = {tuple(item["sample_files"]) for item in raw.values()}
        if len(sample_pools) != 1:
            raise ValueError(f"sample pools differ for atom count {atom_count}")

        ase = {
            optimizer: raw[f"{optimizer}_ase"]["points"][0]
            for optimizer in ("fire", "bfgs")
        }
        active = {
            optimizer: {
                point["batch_size"]: point
                for point in raw[f"{optimizer}_active"]["points"]
            }
            for optimizer in ("fire", "bfgs")
        }
        if active["fire"].keys() != active["bfgs"].keys():
            raise ValueError(f"batch sizes differ for atom count {atom_count}")

        points = []
        for batch_size in sorted(active["fire"]):
            fire_point = active["fire"][batch_size]
            bfgs_point = active["bfgs"][batch_size]
            item: dict[str, Any] = {
                "batch_size": batch_size,
                "active_fire": point_summary(fire_point),
                "active_bfgs": point_summary(bfgs_point),
            }
            if fire_point["status"] == "passed":
                fire_seconds = fire_point["timing"]["median_seconds"]
                item["active_fire"]["speedup_vs_ase_fire"] = (
                    ase["fire"]["timing"]["median_seconds"] / fire_seconds
                )
                item["active_fire"]["validation_vs_ase_fire"] = validate(
                    ase["fire"]["records"], fire_point["records"], atom_count
                )
            if bfgs_point["status"] == "passed":
                bfgs_seconds = bfgs_point["timing"]["median_seconds"]
                item["active_bfgs"]["speedup_vs_ase_bfgs"] = (
                    ase["bfgs"]["timing"]["median_seconds"] / bfgs_seconds
                )
                item["active_bfgs"]["speedup_vs_ase_fire"] = (
                    ase["fire"]["timing"]["median_seconds"] / bfgs_seconds
                )
                item["active_bfgs"]["validation_vs_ase_bfgs"] = validate(
                    ase["bfgs"]["records"], bfgs_point["records"], atom_count
                )
            if fire_point["status"] == bfgs_point["status"] == "passed":
                item["active_bfgs_speedup_vs_active_fire"] = (
                    fire_point["timing"]["median_seconds"]
                    / bfgs_point["timing"]["median_seconds"]
                )
            points.append(item)

        ase_fire_seconds = ase["fire"]["timing"]["median_seconds"]
        ase_bfgs_seconds = ase["bfgs"]["timing"]["median_seconds"]
        result["groups"][str(atom_count)] = {
            "sample_files": raw["fire_ase"]["sample_files"],
            "ase_fire": point_summary(ase["fire"]),
            "ase_bfgs": point_summary(ase["bfgs"]),
            "ase_bfgs_speedup_vs_ase_fire": ase_fire_seconds / ase_bfgs_seconds,
            "points": points,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
