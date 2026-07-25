#!/usr/bin/env python
"""Summarize fixed-slot refill, drain, repack, allocator, and MPS results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def endpoint_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_records = {record["source"]: record for record in reference["records"]}
    candidate_records = {record["source"]: record for record in candidate["records"]}
    if reference_records.keys() != candidate_records.keys():
        raise ValueError("refill outputs contain different source identities")
    return {
        "convergence_mismatches": sum(
            reference_records[key]["converged"]
            != candidate_records[key]["converged"]
            for key in reference_records
        ),
        "step_mismatches": sum(
            reference_records[key]["steps"] != candidate_records[key]["steps"]
            for key in reference_records
        ),
        "max_step_difference": max(
            abs(
                reference_records[key]["steps"]
                - candidate_records[key]["steps"]
            )
            for key in reference_records
        ),
        "max_energy_error_eV_per_atom": max(
            abs(
                reference_records[key]["energy_eV"]
                - candidate_records[key]["energy_eV"]
            )
            / len(reference_records[key]["positions_A"])
            for key in reference_records
        ),
        "max_position_rmsd_A": max(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(reference_records[key]["positions_A"])
                            - np.asarray(candidate_records[key]["positions_A"])
                        )
                    )
                )
            )
            for key in reference_records
        ),
        "max_cell_rmsd_A": max(
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(reference_records[key]["cell_A"])
                            - np.asarray(candidate_records[key]["cell_A"])
                        )
                    )
                )
            )
            for key in reference_records
        ),
    }


def _tensor_row(
    payload: dict[str, Any],
    *,
    model: str,
    workload: str,
    method: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "workload": workload,
        "method": method,
        "wall_time_s": float(payload["timing_seconds"]),
        "systems_per_second": float(payload["systems_per_second"]),
        "peak_allocated_GiB": float(payload["peak_allocated_bytes"]) / 2**30,
        "peak_reserved_or_device_GiB": float(payload["peak_reserved_bytes"]) / 2**30,
        "model_evaluations": int(payload["model_evaluations"]),
        "graph_evaluations": int(payload["graph_evaluations"]),
        "neighbor_rebuilds": int(payload["neighbor_rebuilds"]),
        "converged": int(payload["converged"]),
    }


def _mps_row(
    payload: dict[str, Any],
    *,
    model: str,
    workload: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "workload": workload,
        "method": "mps32",
        "wall_time_s": float(payload["timing"]["wall_seconds"]),
        "systems_per_second": float(payload["timing"]["systems_per_second"]),
        "peak_allocated_GiB": None,
        "peak_reserved_or_device_GiB": (
            float(payload["peak_gpu_memory_bytes_nvidia_smi"]) / 2**30
        ),
        "model_evaluations": int(payload["model_evaluations_total"]),
        "graph_evaluations": None,
        "neighbor_rebuilds": None,
        "converged": int(payload["converged"]),
    }


def summarize(raw_dir: Path, atlas_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for model in ("atombit", "mace"):
        for directory, workload in (("h46", "H46-R256"), ("stage1", "H276-STEPVAR-R256")):
            payloads = {
                method: _load(raw_dir / directory / f"{model}_{method}.json")
                for method in ("active", "repack", "slots")
            }
            for method, payload in payloads.items():
                rows.append(
                    _tensor_row(
                        payload,
                        model=model,
                        workload=workload,
                        method=method,
                    )
                )
            mps = (
                _load(raw_dir / "h46" / f"{model}_mps32.json")
                if directory == "h46"
                else _load(atlas_dir / f"{model}_stepvar_mps32.json")
            )
            rows.append(_mps_row(mps, model=model, workload=workload))
            comparisons.append(
                {
                    "model": model,
                    "workload": workload,
                    "slot_speedup_vs_active": (
                        payloads["active"]["timing_seconds"]
                        / payloads["slots"]["timing_seconds"]
                    ),
                    "slot_speedup_vs_repack": (
                        payloads["repack"]["timing_seconds"]
                        / payloads["slots"]["timing_seconds"]
                    ),
                    "slot_speedup_vs_mps32": (
                        mps["timing"]["wall_seconds"]
                        / payloads["slots"]["timing_seconds"]
                    ),
                    "endpoint_vs_repack": endpoint_comparison(
                        payloads["repack"],
                        payloads["slots"],
                    ),
                }
            )

    expandable = _load(raw_dir / "stage1" / "atombit_slots_expandable.json")
    standard = _load(raw_dir / "stage1" / "atombit_slots.json")
    allocator = {
        "model": "atombit",
        "workload": "H276-STEPVAR-R256",
        "configuration": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "wall_time_s": float(expandable["timing_seconds"]),
        "speedup_vs_standard_slots": (
            standard["timing_seconds"] / expandable["timing_seconds"]
        ),
        "peak_allocated_GiB": float(expandable["peak_allocated_bytes"]) / 2**30,
        "peak_reserved_GiB": float(expandable["peak_reserved_bytes"]) / 2**30,
        "standard_peak_reserved_GiB": (
            float(standard["peak_reserved_bytes"]) / 2**30
        ),
    }
    return {
        "schema_version": 1,
        "timing_repeats": 1,
        "rows": rows,
        "comparisons": comparisons,
        "allocator_ablation": allocator,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "workload",
        "method",
        "wall_time_s",
        "systems_per_second",
        "peak_allocated_GiB",
        "peak_reserved_or_device_GiB",
        "model_evaluations",
        "graph_evaluations",
        "neighbor_rebuilds",
        "converged",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--atlas-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.raw_dir, args.atlas_dir)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    write_csv(args.output_csv, result["rows"])


if __name__ == "__main__":
    main()
