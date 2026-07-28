#!/usr/bin/env python3
"""Summarize the AtomBit refill allocator-control experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

CASE_SPECS = (
    ("H46", "dual_H46_active_B64"),
    ("H46", "dual_H46_refill_B64"),
    ("STEPVAR-H276", "dual_active_B64"),
    ("STEPVAR-H276", "dual_refill_B64"),
    ("STEPVAR-H276", "deprecated_alias_B64"),
    ("STEPVAR-H276", "allocator_gc80_B64"),
    ("STEPVAR-H276", "native_gc80_B64"),
    ("STEPVAR-H276", "auto_B48"),
    ("STEPVAR-H276", "auto_B32"),
    ("STEPVAR-H276", "serial_B64"),
)


def _flatten(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, list):
            output.extend(_flatten(value))
        else:
            output.append(float(value))
    return output


def _rms(left: list[Any], right: list[Any]) -> float:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_flat, right_flat, strict=True))
        / len(left_flat)
    )


def _endpoint_difference(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    energy = []
    position = []
    cell = []
    steps = []
    convergence_mismatches = 0
    for current, control in zip(
        candidate["records"], reference["records"], strict=True
    ):
        if current["source"] != control["source"]:
            raise ValueError("record order differs")
        atom_count = len(current["positions_A"])
        energy.append(
            abs(current["energy_eV"] - control["energy_eV"])
            * 1000.0
            / atom_count
        )
        position.append(_rms(current["positions_A"], control["positions_A"]))
        cell.append(_rms(current["cell_A"], control["cell_A"]))
        steps.append(current["steps"] - control["steps"])
        convergence_mismatches += current["converged"] != control["converged"]
    return {
        "convergence_mismatches": convergence_mismatches,
        "max_energy_difference_meV_per_atom": max(energy),
        "max_position_rmsd_A": max(position),
        "max_cell_rmsd_A": max(cell),
        "max_abs_step_difference": max(abs(value) for value in steps),
        "total_step_difference": sum(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = {
        case: json.loads((args.raw_dir / f"{case}.json").read_text())
        for _, case in CASE_SPECS
    }
    active = raw["dual_active_B64"]
    refill = raw["dual_refill_B64"]
    h46_active = raw["dual_H46_active_B64"]
    h46_refill = raw["dual_H46_refill_B64"]
    rows = []
    for workload, case in CASE_SPECS:
        result = raw[case]
        workload_active = h46_active if workload == "H46" else active
        workload_refill = h46_refill if workload == "H46" else refill
        allocator = result["allocator_metrics"]
        row = {
            "workload": workload,
            "case": case,
            "method": result["method"],
            "batch_size": result["batch_size"],
            "linear_algebra_backend": result["linear_algebra_backend"],
            "wall_seconds": result["timing_seconds"],
            "systems_per_second": result["systems_per_second"],
            "speedup_vs_dual_active": (
                workload_active["timing_seconds"] / result["timing_seconds"]
            ),
            "peak_allocated_GiB": result["peak_allocated_bytes"] / 2**30,
            "peak_reserved_GiB": result["peak_reserved_bytes"] / 2**30,
            "peak_reserved_fraction": (
                result["peak_reserved_bytes"]
                / result["environment"]["gpu_total_memory_bytes"]
            ),
            "allocation_retries": allocator.get("allocation_retries"),
            "inactive_split_peak_GiB": (
                allocator.get("inactive_split_bytes_peak", 0) / 2**30
            ),
            "model_evaluations": result["model_evaluations"],
            "graph_evaluations": result["graph_evaluations"],
            "optimizer_steps_total": result["optimizer_steps_total"],
            "converged_jobs": result["converged"],
            "endpoint_vs_dual_refill": _endpoint_difference(
                result, workload_refill
            ),
        }
        rows.append(row)

    dual_difference = _endpoint_difference(
        refill, raw["deprecated_alias_B64"]
    )
    decision = {
        "dual_refill_all_jobs_converged": refill["converged"] == 256,
        "dual_refill_memory_gate_passed": (
            refill["peak_reserved_bytes"]
            / refill["environment"]["gpu_total_memory_bytes"]
            <= 0.85
        ),
        "dual_refill_throughput_gate_passed": (
            refill["timing_seconds"] <= active["timing_seconds"]
        ),
        "dual_refill_endpoint_gate_passed": (
            dual_difference["max_energy_difference_meV_per_atom"] <= 1.0
            and dual_difference["convergence_mismatches"] == 0
        ),
        "selected_allocator_environment": {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "selected_mode_for_matched_step_variable_H276": "immediate_refill_B64",
        "selected_mode_for_matched_H46": "immediate_refill_B64",
        "production_planner_change": "none_required_already_sets_both_variables",
    }
    summary = {
        "schema_version": 1,
        "timing_repeats": 1,
        "rows": rows,
        "dual_refill_vs_dual_active": {
            "H46": {
                "speedup": (
                    h46_active["timing_seconds"] / h46_refill["timing_seconds"]
                ),
                "additional_peak_reserved_GiB": (
                    h46_refill["peak_reserved_bytes"]
                    - h46_active["peak_reserved_bytes"]
                )
                / 2**30,
            },
            "STEPVAR-H276": {
                "speedup": active["timing_seconds"] / refill["timing_seconds"],
                "additional_peak_reserved_GiB": (
                    refill["peak_reserved_bytes"]
                    - active["peak_reserved_bytes"]
                )
                / 2**30,
            },
        },
        "dual_variable_vs_old_alias_endpoint_difference": dual_difference,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "workload",
        "case",
        "method",
        "batch_size",
        "linear_algebra_backend",
        "wall_seconds",
        "systems_per_second",
        "speedup_vs_dual_active",
        "peak_allocated_GiB",
        "peak_reserved_GiB",
        "peak_reserved_fraction",
        "allocation_retries",
        "inactive_split_peak_GiB",
        "model_evaluations",
        "graph_evaluations",
        "optimizer_steps_total",
        "converged_jobs",
    ]
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields} for row in rows
        )


if __name__ == "__main__":
    main()
