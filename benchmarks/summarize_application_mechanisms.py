#!/usr/bin/env python
"""Summarize the application-shaped tensor batching versus CUDA MPS atlas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MODELS = ("atombit", "mace")
PRACTICAL_GATE = 0.05


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _model_key(value: str) -> str:
    name = value.lower()
    if name.startswith("mace"):
        return "mace"
    if name.startswith("atombit"):
        return "atombit"
    raise ValueError(f"unsupported model identity: {value}")


def _phase_seconds(point: dict[str, Any], prefix: str) -> float:
    phases = point["runtime_profile"]["phases"]
    return sum(
        float(values["total_seconds"])
        for name, values in phases.items()
        if name.startswith(prefix)
    )


def _tensor_row(path: Path, task: str) -> dict[str, Any]:
    payload = _load(path)
    point = payload["points"][0]
    summary = point["summary"]
    telemetry = point["telemetry"]
    wall = float(summary["wall_time_s"])
    return {
        "task": task,
        "workload_id": point["workload_id"],
        "model": _model_key(str(telemetry["model_name"])),
        "method": "tensor",
        "batch_size": int(point["batch_size"]),
        "wall_time_s": wall,
        "throughput_per_s": float(summary["throughput_per_s"]),
        "throughput_unit": str(summary["useful_unit"]) + "_per_second",
        "peak_allocated_GB": float(summary["peak_allocated_GB"]),
        "peak_reserved_GB": float(summary["peak_reserved_GB"]),
        "peak_device_GB": float(summary["peak_reserved_GB"]),
        "model_evaluations": int(telemetry["total_model_calls"]),
        "neighbor_rebuilds": int(telemetry["total_neighbor_rebuilds"]),
        "model_time_s": _phase_seconds(point, "model."),
        "neighbor_time_s": _phase_seconds(point, "graph.neighbor_search"),
        "cache_hit_rate": telemetry["cache_hit_rate"],
        "converged": telemetry["converged_jobs"],
        "finite": int(summary["jobs"]) if telemetry["validation_pass"] else 0,
        "status": point["status"],
    }


def _mps_row(path: Path, task: str) -> dict[str, Any]:
    payload = _load(path)
    timing = payload["timing"]
    workers = payload["worker_results"]
    return {
        "task": task,
        "workload_id": Path(payload["manifest"]["path"]).stem,
        "model": _model_key(str(payload["mlip"])),
        "method": "mps32",
        "batch_size": 1,
        "wall_time_s": float(timing["wall_seconds"]),
        "throughput_per_s": float(timing["throughput_per_second"]),
        "throughput_unit": str(timing["throughput_unit"]),
        "peak_allocated_GB": None,
        "peak_reserved_GB": None,
        "peak_device_GB": float(payload["peak_gpu_memory_bytes_nvidia_smi"]) / 1e9,
        "model_evaluations": sum(int(worker["model_evaluations"]) for worker in workers),
        "neighbor_rebuilds": sum(int(worker["neighbor_rebuilds"]) for worker in workers),
        "model_time_s": None,
        "neighbor_time_s": None,
        "cache_hit_rate": None,
        "converged": payload.get("converged"),
        "finite": payload.get("finite"),
        "status": payload["status"],
    }


def _optimization_row(path: Path) -> dict[str, Any]:
    payload = _load(path)
    return {
        "task": "variable_horizon_optimization",
        "workload_id": payload["workload_id"],
        "model": _model_key(str(payload["mlip"])),
        "method": "active_drain" if payload["method"] == "active" else "active_refill",
        "batch_size": int(payload["batch_size"]),
        "wall_time_s": float(payload["timing_seconds"]),
        "throughput_per_s": float(payload["systems_per_second"]),
        "throughput_unit": "systems_per_second",
        "peak_allocated_GB": float(payload["peak_allocated_bytes"]) / 1e9,
        "peak_reserved_GB": float(payload["peak_reserved_bytes"]) / 1e9,
        "peak_device_GB": float(payload["peak_reserved_bytes"]) / 1e9,
        "model_evaluations": int(payload["model_evaluations"]),
        "neighbor_rebuilds": int(payload["neighbor_rebuilds"]),
        "model_time_s": None,
        "neighbor_time_s": None,
        "graph_evaluations": int(payload["graph_evaluations"]),
        "uncompacted_graph_evaluations": int(payload["uncompacted_graph_evaluations"]),
        "avoided_graph_evaluations": int(payload["avoided_graph_evaluations"]),
        "cache_hit_rate": None,
        "converged": int(payload["converged"]),
        "finite": None,
        "status": payload["status"],
    }


def _add_atom_throughput(
    rows: list[dict[str, Any]],
    manifest_dir: Path,
) -> None:
    for row in rows:
        manifest = _load(manifest_dir / f"{row['workload_id']}.json")
        atom_count = sum(int(job["atom_count"]) for job in manifest["jobs"])
        if row["task"] == "fixed_horizon_nve":
            atom_count *= int(manifest["metadata"]["measured_steps"])
            row["atom_throughput_unit"] = "atom_steps_per_second"
        else:
            row["atom_throughput_unit"] = "atoms_completed_per_second"
        row["atom_throughput_per_s"] = atom_count / row["wall_time_s"]


def collect_rows(raw_dir: Path, manifest_dir: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        rows.extend(
            [
                _tensor_row(raw_dir / f"{model}_eval_tensor_B128.json", "static_evaluation"),
                _mps_row(raw_dir / f"{model}_eval_mps32.json", "static_evaluation"),
                _tensor_row(raw_dir / f"{model}_nve_tensor_B32.json", "fixed_horizon_nve"),
                _mps_row(raw_dir / f"{model}_nve_mps32.json", "fixed_horizon_nve"),
                _optimization_row(raw_dir / f"{model}_stepvar_active_B64.json"),
                _optimization_row(raw_dir / f"{model}_stepvar_refill_B64.json"),
                _mps_row(
                    raw_dir / f"{model}_stepvar_mps32.json",
                    "variable_horizon_optimization",
                ),
            ]
        )

    references = {
        (row["model"], row["task"]): row
        for row in rows
        if row["method"] == "mps32"
    }
    for row in rows:
        reference = references[(row["model"], row["task"])]
        row["speedup_vs_mps32"] = (
            1.0
            if row["method"] == "mps32"
            else reference["wall_time_s"] / row["wall_time_s"]
        )
        row["device_memory_fraction_vs_mps32"] = (
            row["peak_device_GB"] / reference["peak_device_GB"]
        )
    if manifest_dir is not None:
        _add_atom_throughput(rows, manifest_dir)
    return rows


def _effect(speedup: float) -> str:
    if speedup >= 1.0 + PRACTICAL_GATE:
        return "wins"
    if speedup <= 1.0 - PRACTICAL_GATE:
        return "loses"
    return "parity"


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (row["model"], row["task"], row["method"]): row
        for row in rows
    }
    mechanisms: list[dict[str, Any]] = []
    for model in MODELS:
        for task, mechanism in (
            ("static_evaluation", "packed_graph_and_model_batching"),
            ("fixed_horizon_nve", "persistent_tensor_state_batched_integration_and_cache"),
        ):
            tensor = indexed[(model, task, "tensor")]
            mechanisms.append(
                {
                    "model": model,
                    "task": task,
                    "mechanism": mechanism,
                    "speedup_vs_mps32": tensor["speedup_vs_mps32"],
                    "device_memory_fraction_vs_mps32": tensor[
                        "device_memory_fraction_vs_mps32"
                    ],
                    "decision": _effect(tensor["speedup_vs_mps32"]),
                }
            )

        drain = indexed[(model, "variable_horizon_optimization", "active_drain")]
        refill = indexed[(model, "variable_horizon_optimization", "active_refill")]
        refill_speedup = drain["wall_time_s"] / refill["wall_time_s"]
        mechanisms.append(
            {
                "model": model,
                "task": "variable_horizon_optimization",
                "mechanism": "active_refill_over_active_drain",
                "speedup_vs_active_drain": refill_speedup,
                "reserved_memory_ratio_vs_active_drain": (
                    refill["peak_reserved_GB"] / drain["peak_reserved_GB"]
                ),
                "model_evaluation_ratio_vs_active_drain": (
                    refill["model_evaluations"] / drain["model_evaluations"]
                ),
                "decision": _effect(refill_speedup),
            }
        )
        mechanisms.append(
            {
                "model": model,
                "task": "variable_horizon_optimization",
                "mechanism": "active_compaction_batch_engine",
                "speedup_vs_mps32": drain["speedup_vs_mps32"],
                "device_memory_fraction_vs_mps32": drain[
                    "device_memory_fraction_vs_mps32"
                ],
                "decision": _effect(drain["speedup_vs_mps32"]),
            }
        )

    return {
        "schema_version": 1,
        "reference": "CUDA MPS with 32 independent ASE workers",
        "practical_difference_fraction": PRACTICAL_GATE,
        "rows": rows,
        "mechanism_effects": mechanisms,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task",
        "model",
        "method",
        "batch_size",
        "wall_time_s",
        "throughput_per_s",
        "throughput_unit",
        "atom_throughput_per_s",
        "atom_throughput_unit",
        "speedup_vs_mps32",
        "peak_allocated_GB",
        "peak_reserved_GB",
        "peak_device_GB",
        "device_memory_fraction_vs_mps32",
        "model_evaluations",
        "neighbor_rebuilds",
        "model_time_s",
        "neighbor_time_s",
        "converged",
        "finite",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("benchmarks/workloads/manifests"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.raw_dir, args.manifest_dir)
    summary = build_summary(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
