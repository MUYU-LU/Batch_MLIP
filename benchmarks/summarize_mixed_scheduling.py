#!/usr/bin/env python3
"""Summarize heterogeneous tensor scheduling against an ASE/MPS reference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

MODE_ORDER = (
    "fifo_B64",
    "fifo_B128",
    "fifo_B192",
    "refill_B128",
    "bucketed_B64",
    "planned_B192",
    "auto_B192",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_mps_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for worker in result["worker_results"]
        for record in worker["records"]
    ]


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def endpoint_diagnostics(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("endpoint record counts differ")
    energy_per_atom = []
    force_maximum = []
    stress_maximum = []
    position_rmsd = []
    cell_rmsd = []
    convergence_mismatches = 0
    step_differences = []
    for expected, actual in zip(reference, candidate, strict=True):
        if expected["source"] != actual["source"]:
            raise ValueError(
                f"source order differs: {expected['source']} != {actual['source']}"
            )
        atom_count = len(expected["positions_A"])
        energy_per_atom.append(
            abs(expected["energy_eV"] - actual["energy_eV"]) / atom_count
        )
        force_maximum.append(
            abs(
                expected["max_force_eV_per_A"]
                - actual["max_force_eV_per_A"]
            )
        )
        stress_maximum.append(
            float(
                np.abs(
                    np.asarray(expected["stress_eV_per_A3"])
                    - np.asarray(actual["stress_eV_per_A3"])
                ).max()
            )
        )
        position_rmsd.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(expected["positions_A"])
                            - np.asarray(actual["positions_A"])
                        )
                    )
                )
            )
        )
        cell_rmsd.append(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(expected["cell_A"])
                            - np.asarray(actual["cell_A"])
                        )
                    )
                )
            )
        )
        convergence_mismatches += (
            bool(expected["converged"]) != bool(actual["converged"])
        )
        step_differences.append(abs(expected["steps"] - actual["steps"]))
    return {
        "convergence_mismatches": convergence_mismatches,
        "absolute_step_difference": distribution(step_differences),
        "absolute_energy_difference_eV_per_atom": distribution(energy_per_atom),
        "absolute_force_maximum_difference_eV_per_A": distribution(
            force_maximum
        ),
        "maximum_stress_element_difference_eV_per_A3": distribution(
            stress_maximum
        ),
        "position_rmsd_A": distribution(position_rmsd),
        "cell_rmsd_A": distribution(cell_rmsd),
        "interpretation": (
            "descriptive basin diagnostic; convergence is the full-relaxation gate"
        ),
    }


def summarize_model(
    model: str,
    *,
    production_dir: Path,
    capacity_dir: Path,
) -> dict[str, Any]:
    mps = load(production_dir / f"{model}_mps32.json")
    mps_seconds = mps["timing"]["wall_seconds"]
    mps_records = flatten_mps_records(mps)
    rows = []
    tensor_results = {}
    for mode in MODE_ORDER:
        result = load(production_dir / f"{model}_{mode}.json")
        tensor_results[mode] = result
        if result["converged"] != result["jobs"]:
            raise ValueError(f"{model} {mode} did not fully converge")
        sources = [record["source"] for record in result["records"]]
        if len(set(sources)) != len(sources):
            raise ValueError(f"{model} {mode} contains duplicate job IDs")
        rows.append(
            {
                "mode": mode,
                "decision": result["decision"],
                "optimization_seconds": result["optimization_seconds"],
                "end_to_end_seconds": result["end_to_end_seconds"],
                "systems_per_second": result["systems_per_second"],
                "end_to_end_systems_per_second": result[
                    "end_to_end_systems_per_second"
                ],
                "speedup_vs_mps_optimization": (
                    mps_seconds / result["optimization_seconds"]
                ),
                "speedup_vs_mps_end_to_end": (
                    mps_seconds / result["end_to_end_seconds"]
                ),
                "peak_allocated_bytes": result["peak_allocated_bytes"],
                "peak_reserved_bytes": result["peak_reserved_bytes"],
                "converged": result["converged"],
                "optimizer_steps_total": result["optimizer_steps_total"],
                "model_evaluations": result["model_evaluations"],
                "neighbor_rebuilds": result["neighbor_rebuilds"],
                "execution_buckets": result["execution_buckets"],
            }
        )
    fastest = min(rows, key=lambda row: row["end_to_end_seconds"])
    capacity = []
    for batch_size in (64, 128, 192):
        result = load(capacity_dir / f"{model}_B{batch_size}.json")
        capacity.append(
            {
                "batch_size": batch_size,
                "status": result["status"],
                "peak_allocated_bytes": result["peak_allocated_bytes"],
                "peak_reserved_bytes": result["peak_reserved_bytes"],
            }
        )
    auto = tensor_results["auto_B192"]
    fifo64 = tensor_results["fifo_B64"]
    auto_row = next(row for row in rows if row["mode"] == "auto_B192")
    fastest_gain_over_auto = (
        auto_row["end_to_end_seconds"] / fastest["end_to_end_seconds"] - 1.0
    )
    recommended = fastest if fastest_gain_over_auto >= 0.05 else auto_row
    return {
        "model": model,
        "capacity_probes": capacity,
        "mps32": {
            "wall_seconds": mps_seconds,
            "systems_per_second": mps["timing"]["systems_per_second"],
            "peak_device_memory_bytes": mps[
                "peak_gpu_memory_bytes_nvidia_smi"
            ],
            "converged": mps["converged"],
            "optimizer_steps_total": mps["optimizer_steps_total"],
            "model_evaluations_total": mps["model_evaluations_total"],
        },
        "tensor_policies": rows,
        "measured_fastest_policy": fastest["mode"],
        "measured_fastest_gain_over_auto": fastest_gain_over_auto,
        "recommended_policy": recommended["mode"],
        "recommended_decision": recommended["decision"],
        "recommended_speedup_vs_mps_end_to_end": recommended[
            "speedup_vs_mps_end_to_end"
        ],
        "recommended_speedup_vs_fifo_B64_end_to_end": (
            fifo64["end_to_end_seconds"] / recommended["end_to_end_seconds"]
        ),
        "predicted_whole_pool_bytes": auto["plan"][
            "total_workload_predicted_bytes"
        ],
        "endpoint_diagnostics_recommended_vs_mps": endpoint_diagnostics(
            mps_records,
            auto["records"],
        ),
    }


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [
        {"model": summary["model"], **point}
        for summary in summaries
        for point in summary["tensor_policies"]
    ]
    columns = [
        "model",
        "mode",
        "decision",
        "optimization_seconds",
        "end_to_end_seconds",
        "systems_per_second",
        "end_to_end_systems_per_second",
        "speedup_vs_mps_optimization",
        "speedup_vs_mps_end_to_end",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "converged",
        "optimizer_steps_total",
        "model_evaluations",
        "neighbor_rebuilds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-dir",
        type=Path,
        default=Path("runs/robustness/cross_mix/production"),
    )
    parser.add_argument(
        "--capacity-dir",
        type=Path,
        default=Path("runs/robustness/cross_mix/capacity"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    summaries = [
        summarize_model(
            model,
            production_dir=args.production_dir,
            capacity_dir=args.capacity_dir,
        )
        for model in ("atombit", "mace")
    ]
    result = {
        "schema_version": 1,
        "workload_id": "OPT-RB-CROSS-MIX-R192-v1",
        "jobs": 192,
        "timing_repeats": 1,
        "selection_rule": (
            "retain safe whole-pool auto when it is within 5% of the measured "
            "fastest fully converged point; one timing is insufficient to "
            "trade substantial memory headroom for a smaller difference"
        ),
        "models": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, summaries)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
