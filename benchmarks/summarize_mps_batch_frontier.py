#!/usr/bin/env python3
"""Summarize the CUDA MPS versus tensor-batch optimization frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

WORKLOADS = {
    46: {
        "optimizer": "bfgs",
        "batch_file": "confirm_bfgs_H46.json",
        "batch_size": 256,
        "ase_file": "ase64_bfgs_H46.json",
    },
    92: {
        "optimizer": "bfgs",
        "batch_file": "confirm_bfgs_H92.json",
        "batch_size": 256,
        "ase_file": "ase64_bfgs_H92.json",
    },
    184: {
        "optimizer": "bfgs",
        "batch_file": "fallback_active_bfgs_H184.json",
        "batch_size": 64,
        "ase_file": "ase64_bfgs_H184.json",
    },
    276: {
        "optimizer": "fire",
        "batch_file": "final_fire1000_H276.json",
        "batch_size": 128,
        "ase_file": "ase64_fire_H276.json",
    },
}
MPS_WORKERS = (4, 8, 16, 32)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def convergence_count(records: list[dict[str, Any]]) -> int:
    return sum(bool(record["converged"]) for record in records)


def flatten_mps_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for worker in result["worker_results"]
        for record in worker["records"]
    ]


def endpoint_diagnostics(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    atom_count: int,
) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("endpoint record counts differ")
    metrics = {
        "max_energy_difference_eV_per_atom": 0.0,
        "max_force_maximum_difference_eV_per_A": 0.0,
        "max_stress_element_difference_eV_per_A3": 0.0,
        "max_position_rmsd_A": 0.0,
        "max_cell_rmsd_A": 0.0,
    }
    for expected, actual in zip(reference, candidate, strict=True):
        if expected["source"] != actual["source"]:
            raise ValueError(
                f"source order differs: {expected['source']} != {actual['source']}"
            )
        metrics["max_energy_difference_eV_per_atom"] = max(
            metrics["max_energy_difference_eV_per_atom"],
            abs(expected["energy_eV"] - actual["energy_eV"]) / atom_count,
        )
        metrics["max_force_maximum_difference_eV_per_A"] = max(
            metrics["max_force_maximum_difference_eV_per_A"],
            abs(
                expected["max_force_eV_per_A"]
                - actual["max_force_eV_per_A"]
            ),
        )
        stress_difference = np.asarray(expected["stress_eV_per_A3"]) - np.asarray(
            actual["stress_eV_per_A3"]
        )
        metrics["max_stress_element_difference_eV_per_A3"] = max(
            metrics["max_stress_element_difference_eV_per_A3"],
            float(np.abs(stress_difference).max()),
        )
        position_difference = np.asarray(expected["positions_A"]) - np.asarray(
            actual["positions_A"]
        )
        metrics["max_position_rmsd_A"] = max(
            metrics["max_position_rmsd_A"],
            float(np.sqrt(np.mean(np.square(position_difference)))),
        )
        cell_difference = np.asarray(expected["cell_A"]) - np.asarray(
            actual["cell_A"]
        )
        metrics["max_cell_rmsd_A"] = max(
            metrics["max_cell_rmsd_A"],
            float(np.sqrt(np.mean(np.square(cell_difference)))),
        )
    return metrics


def summarize(mps_dir: Path, batch_dir: Path) -> dict[str, Any]:
    rows = []
    checkpoint_sha256 = None
    for atom_count, config in WORKLOADS.items():
        optimizer = config["optimizer"]
        batch_result = load(batch_dir / config["batch_file"])
        batch_point = next(
            point
            for point in batch_result["points"]
            if point["batch_size"] == config["batch_size"]
        )
        batch_records = batch_point["records"]
        if convergence_count(batch_records) != batch_result["pool_size"]:
            raise ValueError(f"invalid batched convergence for H{atom_count}")

        current_sha256 = batch_result["checkpoint"]["sha256"]
        if checkpoint_sha256 is None:
            checkpoint_sha256 = current_sha256
        elif checkpoint_sha256 != current_sha256:
            raise ValueError("batched checkpoint hashes differ")

        mps_points = []
        mps_results = {}
        for workers in MPS_WORKERS:
            path = mps_dir / f"mps{workers}_{optimizer}_H{atom_count}_R256.json"
            result = load(path)
            if result["checkpoint"]["sha256"] != checkpoint_sha256:
                raise ValueError(f"checkpoint mismatch in {path}")
            if result["pool_size"] != batch_result["pool_size"]:
                raise ValueError(f"pool-size mismatch in {path}")
            records = flatten_mps_records(result)
            if convergence_count(records) != result["pool_size"]:
                raise ValueError(f"invalid MPS convergence in {path}")
            mps_results[workers] = result
            mps_points.append(
                {
                    "workers": workers,
                    "wall_seconds": result["timing"]["wall_seconds"],
                    "systems_per_second": result["timing"]["systems_per_second"],
                    "peak_device_memory_bytes": result[
                        "peak_gpu_memory_bytes_nvidia_smi"
                    ],
                    "worker_seconds_min": min(result["timing"]["worker_seconds"]),
                    "worker_seconds_max": max(result["timing"]["worker_seconds"]),
                    "optimizer_steps_total": result["optimizer_steps_total"],
                    "cpu_threads_per_worker": result["parameters"][
                        "cpu_threads_per_worker"
                    ],
                }
            )
        best_mps_point = max(
            mps_points,
            key=lambda point: point["systems_per_second"],
        )
        best_mps_result = mps_results[best_mps_point["workers"]]
        best_mps_records = flatten_mps_records(best_mps_result)

        ase_result = load(batch_dir / config["ase_file"])
        ase64_point = ase_result["points"][0]
        sequential_seconds = (
            ase64_point["timing"]["median_seconds"]
            * batch_result["pool_size"]
            / ase_result["pool_size"]
        )
        batch_seconds = batch_point["timing"]["median_seconds"]
        mps_seconds = best_mps_point["wall_seconds"]
        winner = "mps" if mps_seconds < batch_seconds else "tensor_batch"
        rows.append(
            {
                "atom_count": atom_count,
                "optimizer": optimizer,
                "pool_size": batch_result["pool_size"],
                "best_mps_workers": best_mps_point["workers"],
                "best_mps_seconds": mps_seconds,
                "best_mps_systems_per_second": best_mps_point[
                    "systems_per_second"
                ],
                "best_mps_peak_device_memory_bytes": best_mps_point[
                    "peak_device_memory_bytes"
                ],
                "best_mps_optimizer_steps_total": best_mps_point[
                    "optimizer_steps_total"
                ],
                "batch_method": batch_result["method"],
                "batch_size": batch_point["batch_size"],
                "batch_seconds": batch_seconds,
                "batch_systems_per_second": batch_point["systems_per_second"],
                "batch_peak_allocated_bytes": batch_point["peak_memory_bytes"],
                "batch_peak_reserved_bytes": batch_point[
                    "peak_reserved_memory_bytes"
                ],
                "batch_optimizer_steps_total": batch_point[
                    "optimizer_steps_total"
                ],
                "sequential_ase_seconds": sequential_seconds,
                "mps_speedup_vs_sequential_ase": sequential_seconds / mps_seconds,
                "batch_speedup_vs_sequential_ase": sequential_seconds
                / batch_seconds,
                "mps_speedup_vs_batch": batch_seconds / mps_seconds,
                "winner": winner,
                "winner_speed_ratio": max(mps_seconds, batch_seconds)
                / min(mps_seconds, batch_seconds),
                "converged_mps": convergence_count(best_mps_records),
                "converged_batch": convergence_count(batch_records),
                "endpoint_diagnostics_batch_vs_mps": endpoint_diagnostics(
                    best_mps_records,
                    batch_records,
                    atom_count,
                ),
                "mps_frontier": mps_points,
            }
        )
    return {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "pool_size": 256,
        "mps_workers_tested": list(MPS_WORKERS),
        "selection_rule": "fastest fully converged measured point",
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "atom_count",
        "optimizer",
        "best_mps_workers",
        "best_mps_seconds",
        "best_mps_systems_per_second",
        "best_mps_peak_device_memory_bytes",
        "batch_method",
        "batch_size",
        "batch_seconds",
        "batch_systems_per_second",
        "batch_peak_allocated_bytes",
        "batch_peak_reserved_bytes",
        "mps_speedup_vs_sequential_ase",
        "batch_speedup_vs_sequential_ase",
        "mps_speedup_vs_batch",
        "winner",
        "winner_speed_ratio",
        "converged_mps",
        "converged_batch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mps-dir",
        type=Path,
        default=Path("runs/mps_vs_batch"),
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("runs/smooth_rms_optimizer_frontier"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(args.mps_dir, args.batch_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, result["rows"])
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()
