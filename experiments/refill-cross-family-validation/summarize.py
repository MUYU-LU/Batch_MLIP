#!/usr/bin/env python3
"""Summarize the cross-family active-drain versus refill experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

FAMILIES = (
    "GUFJOG44",
    "XATMOV88",
    "XAFPAY172",
    "OBEQIX220",
    "ROFB296",
    "ROFA-MIX",
)
BATCH_SIZE = 64


def _flatten(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, list):
            result.extend(_flatten(value))
        else:
            result.append(float(value))
    return result


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
    current_by_source = {
        record["source"]: record for record in candidate["records"]
    }
    reference_by_source = {
        record["source"]: record for record in reference["records"]
    }
    if current_by_source.keys() != reference_by_source.keys():
        raise ValueError("active and refill sources differ")

    energy = []
    position = []
    cell = []
    step_difference = []
    convergence_mismatches = 0
    for source, current in current_by_source.items():
        control = reference_by_source[source]
        atom_count = len(current["positions_A"])
        energy.append(
            abs(current["energy_eV"] - control["energy_eV"])
            * 1000.0
            / atom_count
        )
        position.append(_rms(current["positions_A"], control["positions_A"]))
        cell.append(_rms(current["cell_A"], control["cell_A"]))
        step_difference.append(current["steps"] - control["steps"])
        convergence_mismatches += current["converged"] != control["converged"]
    return {
        "convergence_mismatches": convergence_mismatches,
        "max_energy_difference_meV_per_atom": max(energy),
        "energy_difference_above_1_meV_per_atom": sum(
            value > 1.0 for value in energy
        ),
        "energy_difference_above_5_meV_per_atom": sum(
            value > 5.0 for value in energy
        ),
        "max_position_rmsd_A": max(position),
        "max_cell_rmsd_A": max(cell),
        "max_abs_step_difference": max(map(abs, step_difference)),
        "total_step_difference": sum(step_difference),
    }


def _method_row(family: str, result: dict[str, Any]) -> dict[str, Any]:
    steps = [record["steps"] for record in result["records"]]
    active_sizes = _flatten(result["active_batch_sizes"])
    allocator = result.get("allocator_metrics", {})
    total_memory = result["environment"]["gpu_total_memory_bytes"]
    return {
        "family": family,
        "method": result["method"],
        "wall_seconds": result["timing_seconds"],
        "systems_per_second": result["systems_per_second"],
        "peak_allocated_GiB": result["peak_allocated_bytes"] / 2**30,
        "peak_reserved_GiB": result["peak_reserved_bytes"] / 2**30,
        "peak_reserved_fraction": result["peak_reserved_bytes"] / total_memory,
        "allocation_retries": allocator.get("allocation_retries"),
        "model_evaluations": result["model_evaluations"],
        "graph_evaluations": result["graph_evaluations"],
        "uncompacted_graph_evaluations": result[
            "uncompacted_graph_evaluations"
        ],
        "avoided_graph_evaluations": result["avoided_graph_evaluations"],
        "neighbor_rebuilds": result["neighbor_rebuilds"],
        "optimizer_steps_total": result["optimizer_steps_total"],
        "converged_jobs": result["converged"],
        "step_mean": statistics.fmean(steps),
        "step_stddev": statistics.pstdev(steps),
        "step_cv": (
            statistics.pstdev(steps) / statistics.fmean(steps)
            if statistics.fmean(steps)
            else 0.0
        ),
        "step_min": min(steps),
        "step_max": max(steps),
        "mean_active_occupancy": statistics.fmean(active_sizes) / BATCH_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = {
        (family, method): json.loads(
            (args.raw_dir / f"{family}_{method}.json").read_text()
        )
        for family in FAMILIES
        for method in ("active", "refill")
    }
    rows = [
        _method_row(family, raw[family, method])
        for family in FAMILIES
        for method in ("active", "refill")
    ]
    comparisons = []
    for family in FAMILIES:
        active = raw[family, "active"]
        refill = raw[family, "refill"]
        speedup = active["timing_seconds"] / refill["timing_seconds"]
        endpoint = _endpoint_difference(refill, active)
        memory_fraction = (
            refill["peak_reserved_bytes"]
            / refill["environment"]["gpu_total_memory_bytes"]
        )
        gates = {
            "all_jobs_converged": (
                active["converged"] == 256 and refill["converged"] == 256
            ),
            "memory": memory_fraction <= 0.85,
            "endpoint": (
                endpoint["convergence_mismatches"] == 0
                and endpoint["max_energy_difference_meV_per_atom"] <= 5.0
            ),
            "automatic_speedup": speedup >= 1.05,
        }
        comparisons.append(
            {
                "family": family,
                "refill_speedup": speedup,
                "refill_wall_seconds_saved": (
                    active["timing_seconds"] - refill["timing_seconds"]
                ),
                "refill_additional_reserved_GiB": (
                    refill["peak_reserved_bytes"]
                    - active["peak_reserved_bytes"]
                )
                / 2**30,
                "endpoint_difference": endpoint,
                "gates": gates,
                "recommend_refill": all(gates.values()),
                "timing_interpretation": (
                    "refill"
                    if speedup >= 1.05
                    else "active"
                    if speedup <= 0.95
                    else "inconclusive"
                    if 0.98 < speedup < 1.02
                    else "subthreshold"
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "timing_repeats": 1,
        "resident_batch": BATCH_SIZE,
        "jobs_per_family": 256,
        "rows": rows,
        "comparisons": comparisons,
        "limitations": [
            "One timing observation per method does not estimate variance.",
            "ROFA-MIX is an unbucketed heterogeneous negative control.",
            "Recommendations apply only to evidence-matched workloads.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
