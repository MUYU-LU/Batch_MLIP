#!/usr/bin/env python3
"""Combine controlled EVAL/NVE artifacts into JSON, CSV, and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _phase(profile: dict[str, Any], name: str) -> float:
    return float(profile.get("phases", {}).get(name, {}).get("total_seconds", 0.0))


def _model_time(profile: dict[str, Any]) -> float:
    return sum(
        float(values.get("total_seconds", 0.0))
        for name, values in profile.get("phases", {}).items()
        if name.startswith("model.")
    )


def _graph_metrics(profile: dict[str, Any]) -> dict[str, float | int | None]:
    model_events = [
        event
        for event in profile.get("events", [])
        if event.get("name") == "model_evaluation" and event.get("systems")
    ]
    systems = sum(int(event["systems"]) for event in model_events)
    rebuild_events = [
        event for event in profile.get("events", []) if event.get("name") == "neighbor_rebuild"
    ]
    return {
        "mean_directed_edges_per_system": (
            sum(int(event.get("edges", 0)) for event in model_events) / systems if systems else None
        ),
        "mean_candidate_edges_per_system": (
            sum(int(event["candidate_edges"]) for event in model_events) / systems
            if systems and all("candidate_edges" in event for event in model_events)
            else None
        ),
        "neighbor_rebuild_events": len(rebuild_events),
        "rebuilt_systems": sum(int(event.get("rebuilt_systems", 0)) for event in rebuild_events),
    }


def _normalized_status(point: dict[str, Any]) -> str:
    status = str(point["status"])
    if status == "error" and "out of memory" in str(point.get("error", "")).lower():
        return "oom"
    return status


def _row(document: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    summary = point.get("summary", {})
    validation = point.get("validation", {})
    speedup = point.get("speedup_vs_ase_b1", {})
    profile = point.get("runtime_profile", {})
    graph_metrics = _graph_metrics(profile)
    md_energy = summary.get("md_energy", {})
    jobs = summary.get("jobs")
    atom_count = None
    workload_id = str(point["workload_id"])
    if "-H46-" in workload_id:
        atom_count = 46
    elif "-H276-" in workload_id:
        atom_count = 276
    atoms_per_s = None
    if atom_count is not None and jobs is not None and summary.get("wall_time_s"):
        atoms_per_s = atom_count * int(jobs) / float(summary["wall_time_s"])
    gpu_memory_gb = document.get("environment", {}).get("gpu_total_memory_bytes")
    if gpu_memory_gb is not None:
        gpu_memory_gb = float(gpu_memory_gb) / 1e9
    peak_allocated_gb = summary.get("peak_allocated_GB")
    peak_reserved_gb = summary.get("peak_reserved_GB")
    peaks = [float(value) for value in (peak_allocated_gb, peak_reserved_gb) if value is not None]
    memory_gate_fraction = max(peaks) / gpu_memory_gb if peaks and gpu_memory_gb else None
    measured_speedup = speedup.get("measured")
    return {
        "model": document["model"],
        "task": document["task"],
        "pool_size": document["pool_size"],
        "workload_id": workload_id,
        "method": point["method"],
        "batch_size": point.get("batch_size", 1),
        "status": _normalized_status(point),
        "wall_time_s": summary.get("wall_time_s"),
        "end_to_end_time_s": summary.get("end_to_end_time_s"),
        "speedup_vs_ase_b1_measured": measured_speedup,
        "speedup_vs_ase_b1_derived": None,
        "speedup_reference": "measured_same_workload" if measured_speedup else None,
        "speedup_vs_ase_b1_end_to_end": speedup.get("end_to_end"),
        "throughput_per_s": summary.get("throughput_per_s"),
        "atoms_per_s": atoms_per_s,
        "peak_allocated_GB": peak_allocated_gb,
        "peak_reserved_GB": peak_reserved_gb,
        "gpu_memory_GB": gpu_memory_gb,
        "memory_gate_fraction": memory_gate_fraction,
        "model_time_s": _model_time(profile),
        "neighbor_search_time_s": _phase(profile, "graph.neighbor_search"),
        "graph_update_time_s": _phase(profile, "calculator.neighbor_update"),
        **graph_metrics,
        "max_energy_error_eV_per_atom": validation.get("max_abs_energy_error_eV_per_atom"),
        "max_force_error_eV_per_A": validation.get("max_abs_force_error_eV_per_A"),
        "endpoint_position_rmsd_A_max": validation.get("endpoint_position_rmsd_A_max"),
        "endpoint_velocity_rmsd_A_per_fs_max": validation.get(
            "endpoint_velocity_rmsd_A_per_fs_max"
        ),
        "max_abs_energy_drift_eV_per_atom": md_energy.get("max_abs_energy_drift_eV_per_atom"),
        "mean_abs_energy_drift_eV_per_atom": md_energy.get("mean_abs_energy_drift_eV_per_atom"),
        "validation_pass": validation.get("passed"),
        "concurrent_workers": document.get("execution", {}).get("concurrent_workers"),
        "torch_cpu_threads": document.get("execution", {}).get("torch_cpu_threads"),
        "source_status": point["status"],
    }


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["method"] != "native_batch" or not row["status"].startswith("passed"):
            continue
        memory_fraction = row["memory_gate_fraction"]
        if memory_fraction is not None and float(memory_fraction) >= 0.85:
            continue
        groups.setdefault((row["model"], row["task"], row["workload_id"]), []).append(row)
    selected = []
    for _, group in sorted(groups.items()):
        maximum = max(float(item["throughput_per_s"] or 0.0) for item in group)
        indistinguishable = [
            item for item in group if float(item["throughput_per_s"] or 0.0) >= 0.98 * maximum
        ]
        selected.append(min(indistinguishable, key=lambda item: int(item["batch_size"])))
    return selected


def _render_markdown(best: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Task | Workload | Selected B | Speedup | Basis | Peak reserved GB | Throughput |",
        "|:--|:--|:--|--:|--:|:--|--:|--:|",
    ]
    for row in best:
        lines.append(
            "| {model} | {task} | {workload_id} | {batch_size} | {speedup:.3f}x | "
            "{basis} | {peak:.3f} | {throughput:.3f} |".format(
                model=row["model"],
                task=row["task"],
                workload_id=row["workload_id"],
                batch_size=row["batch_size"],
                speedup=float(
                    row["speedup_vs_ase_b1_measured"] or row["speedup_vs_ase_b1_derived"] or 0.0
                ),
                basis=row["speedup_reference"] or "unavailable",
                peak=float(row["peak_reserved_GB"] or 0.0),
                throughput=float(row["throughput_per_s"] or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def _add_derived_r256_speedups(rows: list[dict[str, Any]]) -> None:
    """Use exact-repeat R32 ASE throughput when R256 intentionally omits ASE."""
    ase_throughput: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["method"] != "ase_b1" or row["pool_size"] != 32:
            continue
        workload_family = re.sub(r"-R32-", "-R*-", row["workload_id"])
        ase_throughput[(row["model"], row["task"], workload_family)] = float(
            row["throughput_per_s"]
        )
    for row in rows:
        if (
            row["method"] != "native_batch"
            or row["pool_size"] != 256
            or row["speedup_vs_ase_b1_measured"] is not None
            or not row["status"].startswith("passed")
        ):
            continue
        workload_family = re.sub(r"-R256-", "-R*-", row["workload_id"])
        reference = ase_throughput.get((row["model"], row["task"], workload_family))
        if reference:
            row["speedup_vs_ase_b1_derived"] = float(row["throughput_per_s"]) / reference
            row["speedup_reference"] = "measured_R32_ASE_throughput_exact_repeats"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    rows = [_row(document, point) for document in documents for point in document["points"]]
    _add_derived_r256_speedups(rows)
    best = _best_rows(rows)
    output = {
        "schema_version": 1,
        "input_files": [str(path) for path in args.inputs],
        "rows": rows,
        "best_native_by_workload": best,
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
    }
    for path in (args.output_json, args.output_csv, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.output_markdown.write_text(_render_markdown(best), encoding="utf-8")


if __name__ == "__main__":
    main()
