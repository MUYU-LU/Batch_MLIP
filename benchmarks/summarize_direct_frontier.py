#!/usr/bin/env python3
"""Summarize direct ASE, matscipy, and auto frontier measurements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase(profile: dict[str, Any], name: str) -> float:
    return float(profile.get("phases", {}).get(name, {}).get("total_seconds", 0.0))


def _observed(profile: dict[str, Any]) -> str:
    values = sorted(
        {
            str(event["backend"])
            for event in profile.get("events", [])
            if event.get("name") == "neighbor_rebuild" and event.get("backend")
        }
    )
    return ",".join(values)


def _distribution(workload_id: str) -> str:
    for value in ("H46", "H276", "MIX"):
        if f"-{value}-" in workload_id:
            return value
    raise ValueError(f"cannot infer distribution from {workload_id}")


def _controlled_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = _load(path)
        for point in payload["points"]:
            summary = point["summary"]
            profile = point["runtime_profile"]
            method = point["method"]
            backend = "ase" if method == "ase_b1" else point["neighbor_backend"]
            validation = point.get("validation", {})
            rows.append(
                {
                    "source": str(path),
                    "model": payload["model"],
                    "task": payload["task"],
                    "workload": point["workload_id"],
                    "distribution": _distribution(point["workload_id"]),
                    "pool": payload["pool_size"],
                    "batch": summary["resident_batch_size"],
                    "method": method,
                    "requested_backend": backend,
                    "observed_backend": _observed(profile),
                    "status": point["status"],
                    "wall_time_s": summary["wall_time_s"],
                    "throughput_per_s": summary["throughput_per_s"],
                    "speedup_vs_ase": point.get("speedup_vs_ase_b1", {}).get("measured", 1.0),
                    "peak_allocated_GB": summary.get("peak_allocated_GB"),
                    "peak_reserved_GB": summary.get("peak_reserved_GB"),
                    "memory_gate_fraction": point.get("memory_gate_fraction"),
                    "neighbor_time_s": _phase(profile, "graph.neighbor_search"),
                    "neighbor_fraction": _phase(profile, "graph.neighbor_search")
                    / max(float(summary["wall_time_s"]), 1e-30),
                    "neighbor_rebuilds": point.get("telemetry", {}).get("total_neighbor_rebuilds"),
                    "validation_passed": validation.get("passed", True),
                    "energy_error_eV_per_atom": validation.get("max_abs_energy_error_eV_per_atom"),
                    "force_error_eV_per_A": validation.get("max_abs_force_error_eV_per_A"),
                    "position_rmsd_A": validation.get("endpoint_position_rmsd_A_max"),
                    "velocity_rmsd_A_per_fs": validation.get("endpoint_velocity_rmsd_A_per_fs_max"),
                }
            )
    return rows


def _compare_records(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    if [row["source"] for row in reference] != [row["source"] for row in candidate]:
        raise ValueError("optimization records are not in the same source order")
    energy = []
    force = []
    position = []
    cell = []
    steps = []
    for expected, actual in zip(reference, candidate, strict=True):
        atom_count = len(expected["positions_A"])
        energy.append(abs(actual["energy_eV"] - expected["energy_eV"]) / atom_count)
        force.append(abs(actual["max_force_eV_per_A"] - expected["max_force_eV_per_A"]))
        position.append(
            float(
                np.sqrt(
                    np.mean(
                        (np.asarray(actual["positions_A"]) - np.asarray(expected["positions_A"]))
                        ** 2
                    )
                )
            )
        )
        cell.append(
            float(
                np.sqrt(
                    np.mean((np.asarray(actual["cell_A"]) - np.asarray(expected["cell_A"])) ** 2)
                )
            )
        )
        steps.append(abs(int(actual["steps"]) - int(expected["steps"])))
    return {
        "convergence_match": all(
            bool(actual["converged"]) == bool(expected["converged"])
            for expected, actual in zip(reference, candidate, strict=True)
        ),
        "energy_error_eV_per_atom": max(energy),
        "force_error_eV_per_A": max(force),
        "position_rmsd_A": max(position),
        "cell_rmsd_A": max(cell),
        "step_difference": max(steps),
    }


def _optimization_rows(paths: list[Path]) -> list[dict[str, Any]]:
    payloads = [_load(path) | {"_source": str(path)} for path in paths]
    references = {}
    for payload in payloads:
        model = "MACE-OFF-Small" if payload.get("mlip") == "mace-off" else "AtomBit"
        key = (model, payload["optimizer"], payload["atom_count"], payload["pool_size"])
        if payload["method"] == "ase":
            references[key] = payload["points"][0]

    rows = []
    for payload in payloads:
        model = "MACE-OFF-Small" if payload.get("mlip") == "mace-off" else "AtomBit"
        key = (model, payload["optimizer"], payload["atom_count"], payload["pool_size"])
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"missing ASE optimization reference for {key}")
        for point in payload["points"]:
            profile = (point.get("runtime_profiles") or [{}])[0]
            is_ase = payload["method"] == "ase"
            backend = "ase" if is_ase else payload["parameters"]["neighbor_backend"]
            comparison = (
                {
                    "convergence_match": True,
                    "energy_error_eV_per_atom": 0.0,
                    "force_error_eV_per_A": 0.0,
                    "position_rmsd_A": 0.0,
                    "cell_rmsd_A": 0.0,
                    "step_difference": 0,
                }
                if is_ase
                else _compare_records(reference["records"], point["records"])
            )
            wall_time = float(point["timing"]["median_seconds"])
            rows.append(
                {
                    "source": payload["_source"],
                    "model": model,
                    "task": f"variable_{payload['optimizer']}",
                    "workload": (
                        f"{payload['optimizer'].upper()}-H{payload['atom_count']}"
                        f"-R{payload['pool_size']}"
                    ),
                    "distribution": f"H{payload['atom_count']}",
                    "pool": payload["pool_size"],
                    "batch": 1 if is_ase else point["batch_size"],
                    "method": payload["method"],
                    "requested_backend": backend,
                    "observed_backend": _observed(profile),
                    "status": point["status"],
                    "wall_time_s": wall_time,
                    "throughput_per_s": point["systems_per_second"],
                    "speedup_vs_ase": float(reference["timing"]["median_seconds"]) / wall_time,
                    "peak_allocated_GB": (
                        None
                        if point.get("peak_memory_bytes") is None
                        else float(point["peak_memory_bytes"]) / 1e9
                    ),
                    "peak_reserved_GB": None,
                    "memory_gate_fraction": (
                        None
                        if point.get("peak_memory_bytes") is None
                        else float(point["peak_memory_bytes"])
                        / float(payload["environment"]["gpu_total_memory_bytes"])
                    ),
                    "neighbor_time_s": _phase(profile, "graph.neighbor_search"),
                    "neighbor_fraction": _phase(profile, "graph.neighbor_search")
                    / max(wall_time, 1e-30),
                    "neighbor_rebuilds": point.get("neighbor_rebuilds"),
                    "validation_passed": comparison["convergence_match"],
                    **comparison,
                }
            )
    return rows


def _add_backend_speedup(rows: list[dict[str, Any]]) -> None:
    lookup = {
        (row["model"], row["task"], row["workload"], row["batch"], row["requested_backend"]): row
        for row in rows
    }
    for row in rows:
        reference = lookup.get(
            (row["model"], row["task"], row["workload"], row["batch"], "matscipy")
        )
        row["speedup_vs_matscipy"] = (
            None
            if reference is None
            else float(reference["wall_time_s"]) / float(row["wall_time_s"])
        )


def _selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        memory_fraction = row.get("memory_gate_fraction")
        if (
            row["requested_backend"] != "ase"
            and row["status"] == "passed"
            and bool(row["validation_passed"])
            and (memory_fraction is None or float(memory_fraction) < 0.85)
        ):
            groups[(row["model"], row["task"], row["workload"])].append(row)
    return [min(values, key=lambda row: float(row["wall_time_s"])) for values in groups.values()]


def _write_markdown(path: Path, selected: list[dict[str, Any]]) -> None:
    lines = [
        "# Direct ASE/matscipy/auto frontier",
        "",
        "| Model | Task | Workload | Backend | B | ASE speedup | matscipy speedup | Peak GB | Neighbor % |",
        "|:--|:--|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in sorted(
        selected, key=lambda value: (value["task"], value["model"], value["workload"])
    ):
        peak = max(
            value
            for value in (row["peak_allocated_GB"], row["peak_reserved_GB"])
            if value is not None
        )
        matscipy = row["speedup_vs_matscipy"]
        lines.append(
            f"| {row['model']} | {row['task']} | {row['workload']} | "
            f"{row['requested_backend']} ({row['observed_backend'] or '-'}) | {row['batch']} | "
            f"{row['speedup_vs_ase']:.3f}x | "
            f"{'-' if matscipy is None else f'{matscipy:.3f}x'} | {peak:.3f} | "
            f"{100.0 * row['neighbor_fraction']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled", type=Path, nargs="*", default=[])
    parser.add_argument("--optimization", type=Path, nargs="*", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = _controlled_rows(args.controlled) + _optimization_rows(args.optimization)
    _add_backend_speedup(rows)
    selected = _selected(rows)
    result = {
        "schema_version": 1,
        "rows": rows,
        "selected": selected,
        "all_statuses_passed": all(row["status"] == "passed" for row in rows),
        "all_validation_passed": all(bool(row["validation_passed"]) for row in rows),
        "selected_validation_passed": all(bool(row["validation_passed"]) for row in selected),
        "selected_memory_safe": all(
            row.get("memory_gate_fraction") is None or float(row["memory_gate_fraction"]) < 0.85
            for row in selected
        ),
    }
    for path in (args.output_json, args.output_csv, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(args.output_markdown, selected)


if __name__ == "__main__":
    main()
