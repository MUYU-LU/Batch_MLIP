#!/usr/bin/env python3
"""Summarize the AtomBit blockwise-refill experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MODE_ORDER = ("active", "immediate", "frozen_k5", "block_k5")
WORKLOAD_ORDER = ("H46", "STEPVAR")


def _rms_difference(left: list[Any], right: list[Any]) -> float:
    left_flat: list[float] = []
    right_flat: list[float] = []

    def flatten(values: list[Any], target: list[float]) -> None:
        for value in values:
            if isinstance(value, list):
                flatten(value, target)
            else:
                target.append(float(value))

    flatten(left, left_flat)
    flatten(right, right_flat)
    if len(left_flat) != len(right_flat):
        raise ValueError("cannot compare arrays with different sizes")
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_flat, right_flat, strict=True))
        / len(left_flat)
    )


def _endpoint_difference(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    candidate_records = candidate["records"]
    reference_records = reference["records"]
    if len(candidate_records) != len(reference_records):
        raise ValueError("record counts differ")

    energy_differences = []
    position_differences = []
    cell_differences = []
    step_differences = []
    convergence_mismatches = 0
    for current, control in zip(candidate_records, reference_records, strict=True):
        if current["source"] != control["source"]:
            raise ValueError("record order or source identifiers differ")
        atom_count = len(current["positions_A"])
        energy_differences.append(
            abs(current["energy_eV"] - control["energy_eV"]) * 1000.0 / atom_count
        )
        position_differences.append(
            _rms_difference(current["positions_A"], control["positions_A"])
        )
        cell_differences.append(_rms_difference(current["cell_A"], control["cell_A"]))
        step_differences.append(current["steps"] - control["steps"])
        convergence_mismatches += current["converged"] != control["converged"]

    return {
        "convergence_mismatches": convergence_mismatches,
        "max_energy_difference_meV_per_atom": max(energy_differences),
        "max_position_rmsd_A": max(position_differences),
        "max_cell_rmsd_A": max(cell_differences),
        "min_step_difference": min(step_differences),
        "max_step_difference": max(step_differences),
        "total_step_difference": sum(step_differences),
    }


def _event_count(result: dict[str, Any], name: str) -> int:
    profile = result.get("runtime_profile") or {}
    return sum(event["name"] == name for event in profile.get("events", ()))


def _scheduler_seconds(result: dict[str, Any]) -> float:
    profile = result.get("runtime_profile") or {}
    return sum(
        values["total_seconds"]
        for name, values in profile.get("phases", {}).items()
        if name.startswith("scheduler.")
    )


def _row(
    *,
    workload: str,
    mode: str,
    result: dict[str, Any],
    active: dict[str, Any],
    immediate: dict[str, Any],
) -> dict[str, Any]:
    records = result["records"]
    total_atoms = sum(len(record["positions_A"]) for record in records)
    gpu_memory = result["environment"]["gpu_total_memory_bytes"]
    endpoint = {
        "vs_active": _endpoint_difference(result, active),
        "vs_immediate": _endpoint_difference(result, immediate),
    }
    return {
        "workload": workload,
        "mode": mode,
        "status": result["status"],
        "jobs": result["jobs"],
        "total_atoms": total_atoms,
        "wall_seconds": result["timing_seconds"],
        "systems_per_second": result["systems_per_second"],
        "atoms_per_second": total_atoms / result["timing_seconds"],
        "speedup_vs_active": active["timing_seconds"] / result["timing_seconds"],
        "speedup_vs_immediate": immediate["timing_seconds"] / result["timing_seconds"],
        "peak_allocated_GiB": result["peak_allocated_bytes"] / 2**30,
        "peak_reserved_GiB": result["peak_reserved_bytes"] / 2**30,
        "peak_reserved_fraction": result["peak_reserved_bytes"] / gpu_memory,
        "model_evaluations": result["model_evaluations"],
        "graph_evaluations": result["graph_evaluations"],
        "optimizer_steps_total": result["optimizer_steps_total"],
        "neighbor_rebuilds": result["neighbor_rebuilds"],
        "refill_events": _event_count(result, "refill"),
        "convergence_checks": _event_count(result, "convergence_check"),
        "scheduler_seconds": _scheduler_seconds(result),
        "converged_jobs": result["converged"],
        "max_final_physical_force_eV_per_A": max(
            record["max_force_eV_per_A"] for record in records
        ),
        "endpoint": endpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = {
        workload: {
            mode: json.loads((args.raw_dir / f"{workload}_{mode}.json").read_text())
            for mode in MODE_ORDER
        }
        for workload in WORKLOAD_ORDER
    }
    rows = [
        _row(
            workload=workload,
            mode=mode,
            result=raw[workload][mode],
            active=raw[workload]["active"],
            immediate=raw[workload]["immediate"],
        )
        for workload in WORKLOAD_ORDER
        for mode in MODE_ORDER
    ]
    block_rows = [row for row in rows if row["mode"] == "block_k5"]
    decision = {
        "all_2048_jobs_converged": sum(row["converged_jobs"] for row in rows) == 2048,
        "blockwise_energy_gate_passed": all(
            row["endpoint"]["vs_immediate"][
                "max_energy_difference_meV_per_atom"
            ]
            <= 1.0
            for row in block_rows
        ),
        "blockwise_speed_gate_passed": any(
            row["speedup_vs_active"] >= 1.02
            and row["speedup_vs_immediate"] >= 1.02
            for row in block_rows
        ),
        "blockwise_memory_gate_passed": all(
            row["peak_reserved_fraction"] <= 0.85 for row in block_rows
        ),
        "selected": None,
        "experimental_timing_candidate": "frozen_k5_h46",
        "automatic_policy": "unchanged_active_drain",
    }
    summary = {
        "schema_version": 1,
        "raw_directory": str(args.raw_dir),
        "timing_repeats": 1,
        "rows": rows,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_fields = [
        "workload",
        "mode",
        "wall_seconds",
        "systems_per_second",
        "atoms_per_second",
        "speedup_vs_active",
        "speedup_vs_immediate",
        "peak_allocated_GiB",
        "peak_reserved_GiB",
        "peak_reserved_fraction",
        "model_evaluations",
        "graph_evaluations",
        "optimizer_steps_total",
        "neighbor_rebuilds",
        "refill_events",
        "convergence_checks",
        "scheduler_seconds",
        "converged_jobs",
        "max_energy_difference_meV_per_atom_vs_immediate",
        "max_position_rmsd_A_vs_immediate",
        "max_cell_rmsd_A_vs_immediate",
        "total_step_difference_vs_immediate",
    ]
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in csv_fields if key in row}
            endpoint = row["endpoint"]["vs_immediate"]
            flattened.update(
                {
                    "max_energy_difference_meV_per_atom_vs_immediate": endpoint[
                        "max_energy_difference_meV_per_atom"
                    ],
                    "max_position_rmsd_A_vs_immediate": endpoint[
                        "max_position_rmsd_A"
                    ],
                    "max_cell_rmsd_A_vs_immediate": endpoint["max_cell_rmsd_A"],
                    "total_step_difference_vs_immediate": endpoint[
                        "total_step_difference"
                    ],
                }
            )
            writer.writerow(flattened)


if __name__ == "__main__":
    main()
