#!/usr/bin/env python3
"""Compare OMC-CSP outer-dispatch runs against one reference sequence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.planning import read_planning_profile  # noqa: E402


def _distribution(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "minimum": min(values),
        "mean": mean,
        "maximum": max(values),
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if mean else 0.0
        ),
    }


def _maximum_nested_delta(left: Any, right: Any) -> float:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("endpoint arrays have different lengths")
        return max(
            (
                _maximum_nested_delta(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            ),
            default=0.0,
        )
    return abs(float(left) - float(right))


def _static_profile(profile_path: Path) -> dict[str, Any]:
    profile = read_planning_profile(profile_path)
    atoms = [float(item.structure.atom_count) for item in profile.systems]
    active_edges = [
        float(item.mlip_graph.active_edge_count) for item in profile.systems
    ]
    candidate_edges = [
        float(item.graph_execution.candidate_edge_count)
        for item in profile.systems
    ]
    generalized_dimensions = [
        float(item.task_auxiliary.generalized_dimension)
        for item in profile.systems
    ]
    dense_work = [
        float(item.task_auxiliary.dense_linear_algebra_work)
        for item in profile.systems
    ]
    model_work = [
        256.0 * atom_count + 64.0 * edge_count
        for atom_count, edge_count in zip(atoms, active_edges, strict=True)
    ]
    return {
        "profile_sha256": profile.profile_sha256,
        "system_count": len(profile.systems),
        "atom_count": _distribution(atoms),
        "active_edge_count": _distribution(active_edges),
        "candidate_edge_count": _distribution(candidate_edges),
        "generalized_dimension": _distribution(generalized_dimensions),
        "dense_linear_algebra_work": _distribution(dense_work),
        "model_work": _distribution(model_work),
    }


def _execution_record(call: dict[str, Any]) -> dict[str, Any]:
    scheduling = call["scheduling"]
    materialization = scheduling["structure_materialization"]
    return {
        "seconds": float(call["seconds"]),
        "production_seconds": float(scheduling["production_run_seconds"]),
        "active_device_wall_seconds": float(
            call["active_device_wall_seconds"]
        ),
        "worker_startup_seconds": float(
            scheduling["worker_startup_seconds_this_call"]
        ),
        "materialization_dispatch_wait_seconds": float(
            materialization["dispatch_wait_seconds"]
        ),
        "active_device_count": int(scheduling["active_gpu_count"]),
        "resident_plan_chunk_count": int(
            scheduling["resident_plan_chunk_count"]
        ),
        "execution_chunk_count": int(scheduling["execution_chunk_count"]),
        "execution_chunk_sizes": [
            int(chunk["system_count"])
            for chunk in scheduling["planned_chunks"]
        ],
        "peak_reserved_fraction": float(
            call["peak_memory"]["maximum_reserved_fraction"]
        ),
    }


def _correctness(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_records = baseline["records"]
    candidate_records = candidate["records"]
    if len(baseline_records) != len(candidate_records):
        raise ValueError("candidate and baseline record counts differ")
    pairs = zip(baseline_records, candidate_records, strict=True)
    energy_delta = 0.0
    position_delta = 0.0
    cell_delta = 0.0
    stress_delta = 0.0
    step_difference_count = 0
    source_order_equal = True
    convergence_flags_equal = True
    candidate_all_converged = True
    baseline_max_force = 0.0
    candidate_max_force = 0.0
    for baseline_record, candidate_record in pairs:
        source_order_equal &= (
            baseline_record["source"] == candidate_record["source"]
        )
        convergence_flags_equal &= (
            baseline_record["converged"] == candidate_record["converged"]
        )
        candidate_all_converged &= bool(candidate_record["converged"])
        step_difference_count += (
            baseline_record["steps"] != candidate_record["steps"]
        )
        energy_delta = max(
            energy_delta,
            abs(
                float(baseline_record["energy_eV"])
                - float(candidate_record["energy_eV"])
            ),
        )
        position_delta = max(
            position_delta,
            _maximum_nested_delta(
                baseline_record["positions_A"],
                candidate_record["positions_A"],
            ),
        )
        cell_delta = max(
            cell_delta,
            _maximum_nested_delta(
                baseline_record["cell_A"],
                candidate_record["cell_A"],
            ),
        )
        stress_delta = max(
            stress_delta,
            _maximum_nested_delta(
                baseline_record["stress_eV_per_A3"],
                candidate_record["stress_eV_per_A3"],
            ),
        )
        baseline_max_force = max(
            baseline_max_force,
            float(baseline_record["max_force_eV_per_A"]),
        )
        candidate_max_force = max(
            candidate_max_force,
            float(candidate_record["max_force_eV_per_A"]),
        )
    return {
        "source_order_equal": source_order_equal,
        "convergence_flags_equal": convergence_flags_equal,
        "candidate_all_converged": candidate_all_converged,
        "step_difference_count": step_difference_count,
        "maximum_energy_delta_eV": energy_delta,
        "maximum_position_delta_A": position_delta,
        "maximum_cell_delta_A": cell_delta,
        "maximum_stress_delta_eV_per_A3": stress_delta,
        "baseline_maximum_final_force_eV_per_A": baseline_max_force,
        "candidate_maximum_final_force_eV_per_A": candidate_max_force,
    }


def _candidate_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=_candidate_argument,
        action="append",
        required=True,
    )
    parser.add_argument("--planning-profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    candidates = {
        label: json.loads(path.read_text())
        for label, path in args.candidate
    }
    workload_ids = baseline["workload_ids"]
    if any(result["workload_ids"] != workload_ids for result in candidates.values()):
        raise ValueError("all sequences must contain the same workloads in order")

    workloads = []
    aggregate: dict[str, dict[str, Any]] = {}
    for call_index, workload_id in enumerate(workload_ids):
        baseline_call = baseline["calls"][call_index]
        candidate_records = {}
        for label, candidate in candidates.items():
            candidate_call = candidate["calls"][call_index]
            baseline_seconds = float(baseline_call["seconds"])
            candidate_seconds = float(candidate_call["seconds"])
            candidate_records[label] = {
                "execution": _execution_record(candidate_call),
                "speedup_over_baseline": baseline_seconds / candidate_seconds,
                "wall_time_ratio": candidate_seconds / baseline_seconds,
                "correctness": _correctness(
                    baseline_call,
                    candidate_call,
                ),
            }
        workloads.append(
            {
                "workload_id": workload_id,
                "pool_size": int(baseline_call["pool_size"]),
                "static_profile": _static_profile(
                    args.planning_profile_dir / f"{workload_id}.json"
                ),
                "baseline": _execution_record(baseline_call),
                "candidates": candidate_records,
            }
        )

    baseline_sequence_seconds = float(baseline["timing"]["sequence_seconds"])
    for label, candidate in candidates.items():
        candidate_sequence_seconds = float(
            candidate["timing"]["sequence_seconds"]
        )
        correctness = [
            workload["candidates"][label]["correctness"]
            for workload in workloads
        ]
        aggregate[label] = {
            "sequence_seconds": candidate_sequence_seconds,
            "sequence_speedup_over_baseline": (
                baseline_sequence_seconds / candidate_sequence_seconds
            ),
            "sequence_wall_time_ratio": (
                candidate_sequence_seconds / baseline_sequence_seconds
            ),
            "active_device_wall_seconds": float(
                candidate["timing"]["active_device_wall_seconds"]
            ),
            "all_source_orders_equal": all(
                item["source_order_equal"] for item in correctness
            ),
            "all_convergence_flags_equal": all(
                item["convergence_flags_equal"] for item in correctness
            ),
            "all_candidate_jobs_converged": all(
                item["candidate_all_converged"] for item in correctness
            ),
            "maximum_energy_delta_eV": max(
                item["maximum_energy_delta_eV"] for item in correctness
            ),
            "maximum_position_delta_A": max(
                item["maximum_position_delta_A"] for item in correctness
            ),
            "maximum_cell_delta_A": max(
                item["maximum_cell_delta_A"] for item in correctness
            ),
            "maximum_stress_delta_eV_per_A3": max(
                item["maximum_stress_delta_eV_per_A3"]
                for item in correctness
            ),
            "step_difference_count": sum(
                item["step_difference_count"] for item in correctness
            ),
        }

    output = {
        "schema_version": 1,
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sequence_seconds": baseline_sequence_seconds,
            "active_device_wall_seconds": float(
                baseline["timing"]["active_device_wall_seconds"]
            ),
        },
        "aggregate": aggregate,
        "workloads": workloads,
    }
    if not math.isfinite(baseline_sequence_seconds):
        raise ValueError("baseline sequence timing is not finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}))


if __name__ == "__main__":
    main()
