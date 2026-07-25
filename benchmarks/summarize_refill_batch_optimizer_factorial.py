#!/usr/bin/env python
"""Summarize the BFGS atom-count by resident-batch refill factorial."""

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


def _endpoint_comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    reference_records = {record["source"]: record for record in reference["records"]}
    candidate_records = {record["source"]: record for record in candidate["records"]}
    if reference_records.keys() != candidate_records.keys():
        raise ValueError("outputs contain different source identities")
    return {
        "convergence_mismatches": sum(
            reference_records[key]["converged"] != candidate_records[key]["converged"]
            for key in reference_records
        ),
        "step_mismatches": sum(
            reference_records[key]["steps"] != candidate_records[key]["steps"]
            for key in reference_records
        ),
        "max_step_difference": max(
            abs(reference_records[key]["steps"] - candidate_records[key]["steps"])
            for key in reference_records
        ),
        "max_energy_error_eV_per_atom": max(
            abs(reference_records[key]["energy_eV"] - candidate_records[key]["energy_eV"])
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
    atoms: int,
    batch_size: int,
    method: str,
    optimizer: str = "bfgs",
) -> dict[str, Any]:
    return {
        "optimizer": optimizer,
        "model": model,
        "atoms": atoms,
        "batch_size": batch_size,
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
    atoms: int,
    optimizer: str = "bfgs",
) -> dict[str, Any]:
    return {
        "optimizer": optimizer,
        "model": model,
        "atoms": atoms,
        "batch_size": None,
        "method": "mps32",
        "wall_time_s": float(payload["timing"]["wall_seconds"]),
        "systems_per_second": float(payload["timing"]["systems_per_second"]),
        "peak_allocated_GiB": None,
        "peak_reserved_or_device_GiB": (float(payload["peak_gpu_memory_bytes_nvidia_smi"]) / 2**30),
        "model_evaluations": int(payload["model_evaluations_total"]),
        "graph_evaluations": None,
        "neighbor_rebuilds": None,
        "converged": int(payload["converged"]),
    }


def _step_summary(payload: dict[str, Any]) -> dict[str, float]:
    steps = np.asarray([record["steps"] for record in payload["records"]], dtype=np.float64)
    return {
        "minimum": float(steps.min()),
        "median": float(np.median(steps)),
        "p90": float(np.percentile(steps, 90)),
        "maximum": float(steps.max()),
        "mean": float(steps.mean()),
        "standard_deviation": float(steps.std()),
    }


def summarize(
    raw_dir: Path,
    *,
    h46_mps_dir: Path,
    h276_mps_dir: Path,
    fire_dir: Path | None = None,
    fire_mps_dir: Path | None = None,
    line_search_dir: Path | None = None,
    line_search_mps_dir: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    mps_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for model in ("atombit", "mace"):
        mps_payloads[(model, 46)] = _load(h46_mps_dir / f"{model}_mps32.json")
        mps_payloads[(model, 276)] = _load(h276_mps_dir / f"{model}_H276_mps32.json")
        rows.append(_mps_row(mps_payloads[(model, 46)], model=model, atoms=46))
        rows.append(_mps_row(mps_payloads[(model, 276)], model=model, atoms=276))

        for atoms in (46, 276):
            mps_seconds = float(mps_payloads[(model, atoms)]["timing"]["wall_seconds"])
            candidates: list[dict[str, Any]] = []
            for batch_size in (32, 64, 128):
                payloads = {
                    method: _load(raw_dir / f"{model}_H{atoms}_B{batch_size}_{method}.json")
                    for method in ("active", "repack", "slots")
                }
                for method, payload in payloads.items():
                    row = _tensor_row(
                        payload,
                        model=model,
                        atoms=atoms,
                        batch_size=batch_size,
                        method=method,
                    )
                    rows.append(row)
                    candidates.append(row)
                slot_seconds = float(payloads["slots"]["timing_seconds"])
                comparisons.append(
                    {
                        "model": model,
                        "atoms": atoms,
                        "batch_size": batch_size,
                        "slot_speedup_vs_active": (
                            float(payloads["active"]["timing_seconds"]) / slot_seconds
                        ),
                        "slot_speedup_vs_repack": (
                            float(payloads["repack"]["timing_seconds"]) / slot_seconds
                        ),
                        "slot_speedup_vs_mps32": mps_seconds / slot_seconds,
                        "endpoint_vs_repack": _endpoint_comparison(
                            payloads["repack"],
                            payloads["slots"],
                        ),
                    }
                )
            best = min(candidates, key=lambda row: row["wall_time_s"])
            selections.append(
                {
                    "model": model,
                    "atoms": atoms,
                    "method": best["method"],
                    "batch_size": best["batch_size"],
                    "wall_time_s": best["wall_time_s"],
                    "speedup_vs_mps32": mps_seconds / best["wall_time_s"],
                    "peak_allocated_GiB": best["peak_allocated_GiB"],
                    "peak_reserved_GiB": best["peak_reserved_or_device_GiB"],
                    "mps32_wall_time_s": mps_seconds,
                    "mps32_peak_device_GiB": (
                        float(mps_payloads[(model, atoms)]["peak_gpu_memory_bytes_nvidia_smi"])
                        / 2**30
                    ),
                }
            )
    fire_comparisons: list[dict[str, Any]] = []
    fire_selections: list[dict[str, Any]] = []
    if (fire_dir is None) != (fire_mps_dir is None):
        raise ValueError("fire_dir and fire_mps_dir must be provided together")
    if fire_dir is not None and fire_mps_dir is not None:
        for model in ("atombit", "mace"):
            for atoms in (46, 276):
                mps = _load(fire_mps_dir / f"{model}_H{atoms}_mps32.json")
                mps_seconds = float(mps["timing"]["wall_seconds"])
                rows.append(
                    _mps_row(
                        mps,
                        model=model,
                        atoms=atoms,
                        optimizer="fire",
                    )
                )
                candidates = []
                for batch_size in (32, 128):
                    payloads = {
                        method: _load(fire_dir / f"{model}_H{atoms}_B{batch_size}_{method}.json")
                        for method in ("active", "slots")
                    }
                    for method, payload in payloads.items():
                        row = _tensor_row(
                            payload,
                            model=model,
                            atoms=atoms,
                            batch_size=batch_size,
                            method=method,
                            optimizer="fire",
                        )
                        rows.append(row)
                        candidates.append(row)
                    active_seconds = float(payloads["active"]["timing_seconds"])
                    slot_seconds = float(payloads["slots"]["timing_seconds"])
                    fire_comparisons.append(
                        {
                            "model": model,
                            "atoms": atoms,
                            "batch_size": batch_size,
                            "slot_speedup_vs_active": (active_seconds / slot_seconds),
                            "slot_speedup_vs_mps32": (mps_seconds / slot_seconds),
                            "active_steps": _step_summary(payloads["active"]),
                            "endpoint_vs_active": _endpoint_comparison(
                                payloads["active"],
                                payloads["slots"],
                            ),
                        }
                    )
                best = min(candidates, key=lambda row: row["wall_time_s"])
                fire_selections.append(
                    {
                        "model": model,
                        "atoms": atoms,
                        "method": best["method"],
                        "batch_size": best["batch_size"],
                        "wall_time_s": best["wall_time_s"],
                        "speedup_vs_mps32": (mps_seconds / best["wall_time_s"]),
                        "peak_allocated_GiB": best["peak_allocated_GiB"],
                        "peak_reserved_GiB": best["peak_reserved_or_device_GiB"],
                        "mps32_wall_time_s": mps_seconds,
                        "mps32_peak_device_GiB": (
                            float(mps["peak_gpu_memory_bytes_nvidia_smi"]) / 2**30
                        ),
                    }
                )
    line_search_comparisons: list[dict[str, Any]] = []
    if (line_search_dir is None) != (line_search_mps_dir is None):
        raise ValueError("line_search_dir and line_search_mps_dir must be provided together")
    if line_search_dir is not None and line_search_mps_dir is not None:
        for model in ("atombit", "mace"):
            for atoms in (46, 276):
                tensor = _load(line_search_dir / f"{model}_H{atoms}_B128_active.json")
                mps = _load(line_search_mps_dir / f"{model}_H{atoms}_mps32.json")
                tensor_row = _tensor_row(
                    tensor,
                    model=model,
                    atoms=atoms,
                    batch_size=128,
                    method="active",
                    optimizer="bfgslinesearch",
                )
                rows.append(tensor_row)
                rows.append(
                    _mps_row(
                        mps,
                        model=model,
                        atoms=atoms,
                        optimizer="bfgslinesearch",
                    )
                )
                mps_seconds = float(mps["timing"]["wall_seconds"])
                line_search_comparisons.append(
                    {
                        "model": model,
                        "atoms": atoms,
                        "batch_size": 128,
                        "wall_time_s": tensor_row["wall_time_s"],
                        "mps32_wall_time_s": mps_seconds,
                        "speedup_vs_mps32": (mps_seconds / tensor_row["wall_time_s"]),
                        "steps": _step_summary(tensor),
                        "peak_allocated_GiB": tensor_row["peak_allocated_GiB"],
                        "peak_reserved_GiB": tensor_row["peak_reserved_or_device_GiB"],
                        "mps32_peak_device_GiB": (
                            float(mps["peak_gpu_memory_bytes_nvidia_smi"]) / 2**30
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "timing_repeats": 1,
        "rows": rows,
        "bfgs_comparisons": comparisons,
        "bfgs_best_tensor_selections": selections,
        "fire_comparisons": fire_comparisons,
        "fire_best_tensor_selections": fire_selections,
        "bfgslinesearch_comparisons": line_search_comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "optimizer",
        "model",
        "atoms",
        "batch_size",
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
    parser.add_argument("--h46-mps-dir", type=Path, required=True)
    parser.add_argument("--h276-mps-dir", type=Path, required=True)
    parser.add_argument("--fire-dir", type=Path)
    parser.add_argument("--fire-mps-dir", type=Path)
    parser.add_argument("--line-search-dir", type=Path)
    parser.add_argument("--line-search-mps-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.raw_dir,
        h46_mps_dir=args.h46_mps_dir,
        h276_mps_dir=args.h276_mps_dir,
        fire_dir=args.fire_dir,
        fire_mps_dir=args.fire_mps_dir,
        line_search_dir=args.line_search_dir,
        line_search_mps_dir=args.line_search_mps_dir,
    )
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    _write_csv(args.output_csv, result["rows"])


if __name__ == "__main__":
    main()
