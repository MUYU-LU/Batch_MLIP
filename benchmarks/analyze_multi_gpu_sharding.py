#!/usr/bin/env python3
"""Summarize multi-GPU sharding timings and compare records to W1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FILE_STEMS = {
    ("atombit", "homogeneous-276"): "atombit_homogeneous276",
    ("atombit", "mixed-46-276"): "atombit_mixed46_276",
    ("mace", "homogeneous-276"): "mace_homogeneous276",
    ("mace", "mixed-46-276"): "mace_mixed46_276",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "percentile_90": float(np.percentile(array, 90)),
        "percentile_95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "nonzero_count": int(np.count_nonzero(array)),
        "count_above_1e-3": int(np.count_nonzero(array > 1e-3)),
        "count_above_1e-2": int(np.count_nonzero(array > 1e-2)),
        "count_above_1e-1": int(np.count_nonzero(array > 1e-1)),
    }


def compare_records(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("record counts differ")
    energy = []
    position = []
    cell = []
    max_force = []
    max_stress = []
    step = []
    sources_match = True
    convergence_match = True
    for expected, actual in zip(reference, candidate, strict=True):
        sources_match &= expected["source"] == actual["source"]
        convergence_match &= expected["converged"] == actual["converged"]
        energy.append(abs(expected["energy_eV"] - actual["energy_eV"]))
        position.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(expected["positions_A"])
                        - np.asarray(actual["positions_A"])
                    )
                )
            )
        )
        cell.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(expected["cell_A"])
                        - np.asarray(actual["cell_A"])
                    )
                )
            )
        )
        max_force.append(
            abs(
                expected["max_force_eV_per_A"]
                - actual["max_force_eV_per_A"]
            )
        )
        max_stress.append(
            abs(
                expected["max_abs_stress_eV_per_A3"]
                - actual["max_abs_stress_eV_per_A3"]
            )
        )
        step.append(abs(expected["steps"] - actual["steps"]))
    return {
        "sources_match": bool(sources_match),
        "convergence_flags_match": bool(convergence_match),
        "step_mismatch_count": sum(value != 0 for value in step),
        "absolute_step_difference": distribution(step),
        "absolute_energy_difference_eV": distribution(energy),
        "max_abs_position_difference_A": distribution(position),
        "max_abs_cell_difference_A": distribution(cell),
        "absolute_max_force_difference_eV_per_A": distribution(max_force),
        "absolute_max_stress_difference_eV_per_A3": distribution(max_stress),
    }


def phase_totals(data: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for worker in data["workers"]:
        for name, phase in worker["runtime_profile"]["phases"].items():
            totals[name] = totals.get(name, 0.0) + float(phase["total_seconds"])
    return dict(sorted(totals.items()))


def point_summary(
    data: dict[str, Any], baseline: dict[str, Any], path: Path
) -> dict[str, Any]:
    workers = int(data["worker_count"])
    optimization_speedup = (
        baseline["optimization_wall_seconds"] / data["optimization_wall_seconds"]
    )
    end_to_end_speedup = (
        baseline["end_to_end_wall_seconds"] / data["end_to_end_wall_seconds"]
    )
    worker_seconds = [worker["run_seconds"] for worker in data["workers"]]
    return {
        "raw_file": str(path),
        "raw_sha256": sha256(path),
        "all_converged": data["all_converged"],
        "optimization_wall_seconds": data["optimization_wall_seconds"],
        "startup_wall_seconds": data["startup_wall_seconds"],
        "end_to_end_wall_seconds": data["end_to_end_wall_seconds"],
        "systems_per_second": data["systems_per_second"],
        "atoms_per_second": data["atoms_per_second"],
        "optimization_speedup_vs_w1": optimization_speedup,
        "parallel_efficiency": optimization_speedup / workers,
        "end_to_end_speedup_vs_w1": end_to_end_speedup,
        "worker_run_seconds_min": min(worker_seconds),
        "worker_run_seconds_max": max(worker_seconds),
        "worker_imbalance_ratio": max(worker_seconds) / min(worker_seconds),
        "peak_memory_bytes_max_per_worker": max(
            worker["peak_memory_bytes"] for worker in data["workers"]
        ),
        "model_evaluations": data["model_evaluations"],
        "neighbor_rebuilds": data["neighbor_rebuilds"],
        "optimizer_steps_total": data["optimizer_steps_total"],
        "phase_device_seconds_sum": phase_totals(data),
        "comparison_to_w1": compare_records(
            baseline["records"], data["records"]
        ),
    }


def read_complete(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise RuntimeError(f"incomplete result: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/multi_gpu_sharding")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/multi-gpu-sharding/results.json"),
    )
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema_version": 1,
        "main_matrix": {},
        "small_pool_control": {},
    }
    for (model, workload), stem in FILE_STEMS.items():
        paths = {
            workers: args.run_dir / f"{stem}_w{workers}.json"
            for workers in (1, 4, 7)
        }
        data = {workers: read_complete(path) for workers, path in paths.items()}
        model_result = result["main_matrix"].setdefault(model, {})
        model_result[workload] = {
            str(workers): point_summary(
                data[workers], data[1], paths[workers]
            )
            for workers in (1, 4, 7)
        }

    for model in ("atombit", "mace"):
        paths = {
            workers: args.run_dir / f"control32_{model}_mixed_w{workers}.json"
            for workers in (1, 7)
        }
        data = {workers: read_complete(path) for workers, path in paths.items()}
        result["small_pool_control"][model] = {
            str(workers): point_summary(
                data[workers], data[1], paths[workers]
            )
            for workers in (1, 7)
        }

    result["correctness"] = {
        "all_points_converged": all(
            point["all_converged"]
            for model in result["main_matrix"].values()
            for workload in model.values()
            for point in workload.values()
        )
        and all(
            point["all_converged"]
            for model in result["small_pool_control"].values()
            for point in model.values()
        ),
        "all_sources_match_w1": all(
            point["comparison_to_w1"]["sources_match"]
            for model in result["main_matrix"].values()
            for workload in model.values()
            for point in workload.values()
        ),
        "all_convergence_flags_match_w1": all(
            point["comparison_to_w1"]["convergence_flags_match"]
            for model in result["main_matrix"].values()
            for workload in model.values()
            for point in workload.values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
