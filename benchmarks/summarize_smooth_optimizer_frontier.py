#!/usr/bin/env python3
"""Summarize the smooth-RMS AtomBit optimizer frontier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ASE_FILES = {
    "fire": {
        atoms: f"ase64_fire_H{atoms}.json" for atoms in (46, 92, 184, 276)
    },
    "bfgs": {atoms: f"ase64_bfgs_H{atoms}.json" for atoms in (46, 92, 184, 276)},
    "bfgslinesearch": {
        atoms: f"ase64_bfgslinesearch_H{atoms}.json"
        for atoms in (46, 92, 184, 276)
    },
}

BATCH_FILES = {
    "fire": {
        46: [("active", "final_fire1000_H46.json")],
        92: [("active", "final_fire2000_H92.json")],
        184: [("active", "batch_fire1000_H184.json")],
        276: [("active", "final_fire1000_H276.json")],
    },
    "bfgs": {
        46: [("refill", "confirm_bfgs_H46.json")],
        92: [
            ("refill", "confirm_bfgs_H92.json"),
            ("active", "fallback_active_bfgs_H92.json"),
        ],
        184: [("active", "fallback_active_bfgs_H184.json")],
        276: [
            ("refill", "confirm_bfgs_H276.json"),
            ("active", "fallback_active_bfgs_H276.json"),
        ],
    },
    "bfgslinesearch": {
        46: [("active", "final_bfgslinesearch_H46.json")],
        92: [
            ("active", "final_bfgslinesearch_H92.json"),
            ("active", "fallback_bfgslinesearch_H92_B64.json"),
        ],
        184: [
            ("active", "final_bfgslinesearch_H184.json"),
            ("active", "fallback_bfgslinesearch_H184_B64.json"),
            ("active", "fallback_bfgslinesearch_H184_B32.json"),
            ("active", "fallback_bfgslinesearch_H184_B16.json"),
        ],
        276: [("active", "batch_bfgslinesearch1000_H276.json")],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def convergence_count(point: dict[str, Any]) -> int:
    return sum(bool(record["converged"]) for record in point["records"])


def point_is_valid(
    point: dict[str, Any],
    *,
    pool_size: int,
    memory_limit_bytes: float,
) -> bool:
    reserved = point.get("peak_reserved_memory_bytes")
    return bool(
        point.get("status") == "passed"
        and convergence_count(point) == pool_size
        and reserved is not None
        and reserved <= memory_limit_bytes
    )


def select_frontier(
    points: list[dict[str, Any]],
    *,
    pool_size: int,
    memory_limit_bytes: float,
    relative_band: float = 0.02,
) -> dict[str, Any] | None:
    valid = [
        point
        for point in points
        if point_is_valid(
            point,
            pool_size=pool_size,
            memory_limit_bytes=memory_limit_bytes,
        )
    ]
    if not valid:
        return None
    maximum = max(float(point["systems_per_second"]) for point in valid)
    threshold = maximum * (1.0 - relative_band)
    return min(
        (
            point
            for point in valid
            if float(point["systems_per_second"]) >= threshold
        ),
        key=lambda point: int(point["batch_size"]),
    )


def endpoint_diagnostics(
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    atom_count: int,
) -> dict[str, float]:
    references = {record["source"]: record for record in reference_records}
    metrics = {
        "max_energy_difference_eV_per_atom": 0.0,
        "max_force_maximum_difference_eV_per_A": 0.0,
        "max_stress_element_difference_eV_per_A3": 0.0,
        "max_position_rmsd_A": 0.0,
        "max_cell_rmsd_A": 0.0,
    }
    for candidate in candidate_records:
        reference = references[candidate["source"]]
        metrics["max_energy_difference_eV_per_atom"] = max(
            metrics["max_energy_difference_eV_per_atom"],
            abs(candidate["energy_eV"] - reference["energy_eV"]) / atom_count,
        )
        metrics["max_force_maximum_difference_eV_per_A"] = max(
            metrics["max_force_maximum_difference_eV_per_A"],
            abs(
                candidate["max_force_eV_per_A"]
                - reference["max_force_eV_per_A"]
            ),
        )
        stress_difference = np.asarray(candidate["stress_eV_per_A3"]) - np.asarray(
            reference["stress_eV_per_A3"]
        )
        metrics["max_stress_element_difference_eV_per_A3"] = max(
            metrics["max_stress_element_difference_eV_per_A3"],
            float(np.abs(stress_difference).max()),
        )
        position_difference = np.asarray(candidate["positions_A"]) - np.asarray(
            reference["positions_A"]
        )
        metrics["max_position_rmsd_A"] = max(
            metrics["max_position_rmsd_A"],
            float(np.sqrt(np.mean(np.square(position_difference)))),
        )
        cell_difference = np.asarray(candidate["cell_A"]) - np.asarray(
            reference["cell_A"]
        )
        metrics["max_cell_rmsd_A"] = max(
            metrics["max_cell_rmsd_A"],
            float(np.sqrt(np.mean(np.square(cell_difference)))),
        )
    return metrics


def summarize(raw_dir: Path) -> dict[str, Any]:
    rows = []
    raw_sha256 = {}
    gpu_total_memory_bytes = None
    for optimizer, atom_files in ASE_FILES.items():
        for atom_count, ase_name in atom_files.items():
            ase_path = raw_dir / ase_name
            ase = load(ase_path)
            raw_sha256[ase_name] = sha256_file(ase_path)
            batch_points = []
            batch_names = []
            for batch_mode, batch_name in BATCH_FILES[optimizer][atom_count]:
                batch_path = raw_dir / batch_name
                batch = load(batch_path)
                batch_names.append(batch_name)
                raw_sha256[batch_name] = sha256_file(batch_path)
                batch_points.extend(
                    {
                        **point,
                        "_batch_mode": batch_mode,
                        "_batch_file": batch_name,
                    }
                    for point in batch["points"]
                )
            if gpu_total_memory_bytes is None:
                gpu_total_memory_bytes = int(
                    batch["environment"]["gpu_total_memory_bytes"]
                )
            memory_limit = 0.85 * gpu_total_memory_bytes
            ase_point = ase["points"][0]
            selected = select_frontier(
                batch_points,
                pool_size=256,
                memory_limit_bytes=memory_limit,
            )
            row: dict[str, Any] = {
                "optimizer": optimizer,
                "atom_count": atom_count,
                "ase_file": ase_name,
                "batch_files": batch_names,
                "ase_converged": convergence_count(ase_point),
                "ase_wall_seconds_R64": ase_point["timing"]["median_seconds"],
                "ase_model_evaluations_R64": ase_point["model_evaluations"],
                "candidate_points": [],
                "selected": None,
            }
            for point in batch_points:
                row["candidate_points"].append(
                    {
                        "batch_file": point["_batch_file"],
                        "batch_mode": point["_batch_mode"],
                        "batch_size": point["batch_size"],
                        "wall_seconds": point.get("timing", {}).get("median_seconds"),
                        "systems_per_second": point.get("systems_per_second"),
                        "converged": (
                            convergence_count(point) if "records" in point else 0
                        ),
                        "peak_allocated_bytes": point.get("peak_memory_bytes"),
                        "peak_reserved_bytes": point.get(
                            "peak_reserved_memory_bytes"
                        ),
                        "valid": point_is_valid(
                            point,
                            pool_size=256,
                            memory_limit_bytes=memory_limit,
                        ),
                    }
                )
            if convergence_count(ase_point) == 64 and selected is not None:
                selected_seconds = float(selected["timing"]["median_seconds"])
                row["selected"] = {
                    "batch_file": selected["_batch_file"],
                    "batch_mode": selected["_batch_mode"],
                    "batch_size": selected["batch_size"],
                    "wall_seconds_R256": selected_seconds,
                    "systems_per_second": selected["systems_per_second"],
                    "speedup_vs_ase": (
                        4.0
                        * float(ase_point["timing"]["median_seconds"])
                        / selected_seconds
                    ),
                    "peak_allocated_bytes": selected["peak_memory_bytes"],
                    "peak_reserved_bytes": selected[
                        "peak_reserved_memory_bytes"
                    ],
                    "model_evaluations": selected["model_evaluations"],
                    "graph_evaluations": selected["graph_evaluations"],
                    "neighbor_rebuilds": selected["neighbor_rebuilds"],
                    "optimizer_steps_total": selected["optimizer_steps_total"],
                    "endpoint": endpoint_diagnostics(
                        ase_point["records"],
                        selected["records"],
                        atom_count,
                    ),
                }
            rows.append(row)

    best_by_atom_count = {}
    for atom_count in (46, 92, 184, 276):
        eligible = [
            row
            for row in rows
            if row["atom_count"] == atom_count and row["selected"] is not None
        ]
        best_by_atom_count[str(atom_count)] = (
            min(
                eligible,
                key=lambda row: float(row["selected"]["wall_seconds_R256"]),
            )["optimizer"]
            if eligible
            else None
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "selection": {
            "pool_size": 256,
            "source_pool_size": 64,
            "memory_fraction_limit": 0.85,
            "throughput_relative_band": 0.02,
            "rule": "smallest valid batch within 2% of maximum throughput",
        },
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "rows": rows,
        "best_optimizer_by_atom_count": best_by_atom_count,
        "raw_sha256": raw_sha256,
    }


def write_csv(path: Path, result: dict[str, Any]) -> None:
    fields = (
        "optimizer",
        "atom_count",
        "batch_mode",
        "ase_converged",
        "selected_batch_size",
        "wall_seconds_R256",
        "systems_per_second",
        "speedup_vs_ase",
        "peak_allocated_GB",
        "peak_reserved_GB",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            selected = row["selected"] or {}
            writer.writerow(
                {
                    "optimizer": row["optimizer"],
                    "atom_count": row["atom_count"],
                    "batch_mode": selected.get("batch_mode"),
                    "ase_converged": row["ase_converged"],
                    "selected_batch_size": selected.get("batch_size"),
                    "wall_seconds_R256": selected.get("wall_seconds_R256"),
                    "systems_per_second": selected.get("systems_per_second"),
                    "speedup_vs_ase": selected.get("speedup_vs_ase"),
                    "peak_allocated_GB": (
                        selected.get("peak_allocated_bytes", 0) / 1e9
                        if selected
                        else None
                    ),
                    "peak_reserved_GB": (
                        selected.get("peak_reserved_bytes", 0) / 1e9
                        if selected
                        else None
                    ),
                }
            )


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "| Optimizer | Size | Mode | Selected B | R256 s | systems/s | vs ASE | Allocated GB | Reserved GB |",
        "|:--|--:|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for row in result["rows"]:
        selected = row["selected"]
        if selected is None:
            values = (
                row["optimizer"],
                row["atom_count"],
                row["candidate_points"][0]["batch_mode"],
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
        else:
            values = (
                row["optimizer"],
                row["atom_count"],
                selected["batch_mode"],
                selected["batch_size"],
                f'{selected["wall_seconds_R256"]:.3f}',
                f'{selected["systems_per_second"]:.3f}',
                f'{selected["speedup_vs_ase"]:.3f}x',
                f'{selected["peak_allocated_bytes"] / 1e9:.3f}',
                f'{selected["peak_reserved_bytes"] / 1e9:.3f}',
            )
        lines.append("| " + " | ".join(map(str, values)) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "results.csv", result)
    write_markdown(args.output_dir / "results.md", result)


if __name__ == "__main__":
    main()
