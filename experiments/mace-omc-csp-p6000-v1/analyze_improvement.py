#!/usr/bin/env python3
"""Compare improved MACE batching with rebuild batching and MPS16."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {record["source"]: record for record in payload["records"]}
    if len(result) != int(payload["pool_size"]):
        raise ValueError("result contains duplicate or missing source IDs")
    return result


def _paired_outcomes(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("paired results do not cover identical sources")
    left_only = 0
    right_only = 0
    common_nonconverged = 0
    energy_mev_per_atom = []
    for source, left_record in left.items():
        right_record = right[source]
        left_converged = bool(left_record["converged"])
        right_converged = bool(right_record["converged"])
        left_only += left_converged and not right_converged
        right_only += right_converged and not left_converged
        common_nonconverged += not left_converged and not right_converged
        atom_count = len(left_record["positions_A"])
        energy_mev_per_atom.append(
            1000.0
            * abs(left_record["energy_eV"] - right_record["energy_eV"])
            / atom_count
        )
    return {
        "convergence_flag_mismatches": left_only + right_only,
        "left_only_converged": left_only,
        "right_only_converged": right_only,
        "common_nonconverged": common_nonconverged,
        "energy_difference_meV_per_atom": _quantiles(
            energy_mev_per_atom
        ),
    }


def _phase_totals(payload: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for worker in payload["workers"]:
        for chunk in worker["chunks"]:
            for name, phase in chunk["runtime_profile"]["phases"].items():
                target = result.setdefault(name, {"count": 0, "seconds": 0.0})
                target["count"] += int(phase["count"])
                target["seconds"] += float(phase["total_seconds"])
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--mps16", type=Path, required=True)
    parser.add_argument("--capacity-policy", type=Path, required=True)
    parser.add_argument("--capacity-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rebuild = json.loads(args.rebuild.read_text(encoding="utf-8"))
    improved = json.loads(args.improved.read_text(encoding="utf-8"))
    mps16 = json.loads(args.mps16.read_text(encoding="utf-8"))
    capacity_policy = json.loads(
        args.capacity_policy.read_text(encoding="utf-8")
    )
    capacity_calibration = json.loads(
        args.capacity_calibration.read_text(encoding="utf-8")
    )
    for payload in (rebuild, improved, mps16):
        if payload["status"] != "complete":
            raise ValueError("all compared production runs must be complete")
    identity_keys = ("pool_size", "workload_manifest_sha256")
    for key in identity_keys:
        if len({rebuild[key], improved[key], mps16[key]}) != 1:
            raise ValueError(f"production runs disagree on {key}")
    if len(
        {
            rebuild["checkpoint"]["sha256"],
            improved["checkpoint"]["sha256"],
            mps16["checkpoint"]["sha256"],
        }
    ) != 1:
        raise ValueError("production runs use different model files")
    if improved["contract"].get("mace_graph_mode") != "cached":
        raise ValueError("improved run did not use cached MACE tensor graphs")
    allocator = improved["scheduling"].get("allocator", {})
    if allocator.get("selected_policy") != "expandable_segments":
        raise ValueError("improved run did not select expandable segments")
    if any(
        worker.get("allocator", {}).get("selected_policy")
        != "expandable_segments"
        for worker in improved["workers"]
    ):
        raise ValueError("not every improved worker used expandable segments")

    rebuild_records = _records(rebuild)
    improved_records = _records(improved)
    mps_records = _records(mps16)
    phases = _phase_totals(improved)
    worker_seconds = sum(
        float(worker["task_seconds"]) for worker in improved["workers"]
    )
    selected_phases = {}
    for name in (
        "model.forward",
        "graph.mace_tensor_state",
        "calculator.neighbor_update",
        "graph.cache_validity",
        "graph.cache_selection",
        "graph.neighbor_search",
        "optimizer.bfgs_update",
        "scheduler.active_compaction",
    ):
        phase = phases.get(name, {"count": 0, "seconds": 0.0})
        selected_phases[name] = {
            **phase,
            "worker_time_fraction": float(phase["seconds"]) / worker_seconds,
        }

    chunks = [
        chunk
        for worker in improved["workers"]
        for chunk in worker["chunks"]
    ]
    peak_allocated = max(
        int(chunk["peak_allocated_bytes"]) for chunk in chunks
    )
    peak_reserved = max(int(chunk["peak_reserved_bytes"]) for chunk in chunks)
    post_cleanup_reserved = max(
        int(chunk["post_cleanup_reserved_bytes"]) for chunk in chunks
    )
    predicted_peak = max(int(chunk["predicted_peak_bytes"]) for chunk in chunks)
    memory_budget = int(improved["scheduling"]["memory_budget_bytes_per_gpu"])
    capacity = improved["scheduling"]["capacity_planning"]
    if capacity["mode"] != "offline_hardware_model":
        raise ValueError("improved run did not use its signed capacity model")
    if capacity["policy_sha256"] != capacity_policy["policy_sha256"]:
        raise ValueError("improved run used a different capacity policy")
    calibration_hash = capacity_calibration["calibration_sha256"]
    if (
        capacity["source_calibration_sha256"] != calibration_hash
        or capacity_policy["source_calibration_sha256"] != calibration_hash
    ):
        raise ValueError("capacity policy and calibration provenance disagree")
    if peak_reserved > memory_budget:
        raise ValueError("improved run exceeded its signed memory budget")

    improved_seconds = float(improved["timing"]["execution_seconds"])
    rebuild_seconds = float(rebuild["timing"]["execution_seconds"])
    mps_seconds = float(mps16["timing"]["production_makespan_seconds"])
    improved_script = float(improved["timing"]["script_seconds"])
    rebuild_script = float(rebuild["timing"]["script_seconds"])
    mps_script = float(mps16["timing"]["script_seconds"])
    result = {
        "schema_version": 1,
        "status": "accepted",
        "artifacts": {
            "rebuild_sha256": _sha256(args.rebuild),
            "improved_sha256": _sha256(args.improved),
            "mps16_sha256": _sha256(args.mps16),
            "capacity_policy_sha256": _sha256(args.capacity_policy),
            "capacity_calibration_sha256": _sha256(
                args.capacity_calibration
            ),
        },
        "workload": {
            "id": improved["workload_id"],
            "manifest_sha256": improved["workload_manifest_sha256"],
            "pool_size": improved["pool_size"],
            "same_source_coverage": True,
        },
        "timing": {
            "improved_execution_seconds": improved_seconds,
            "rebuild_execution_seconds": rebuild_seconds,
            "mps16_makespan_seconds": mps_seconds,
            "improved_full_script_seconds": improved_script,
            "rebuild_full_script_seconds": rebuild_script,
            "mps16_full_script_seconds": mps_script,
            "speedup_over_rebuild_execution": (
                rebuild_seconds / improved_seconds
            ),
            "speedup_over_mps16_execution": mps_seconds / improved_seconds,
            "speedup_over_rebuild_full_script": (
                rebuild_script / improved_script
            ),
            "speedup_over_mps16_full_script": mps_script / improved_script,
        },
        "convergence": {
            "improved": improved["converged_count"],
            "rebuild": rebuild["converged_count"],
            "mps16": mps16["converged_count"],
            "improved_vs_rebuild": _paired_outcomes(
                improved_records,
                rebuild_records,
            ),
            "improved_vs_mps16": _paired_outcomes(
                improved_records,
                mps_records,
            ),
        },
        "memory": {
            "budget_bytes": memory_budget,
            "predicted_peak_bytes": predicted_peak,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "post_cleanup_reserved_bytes": post_cleanup_reserved,
            "peak_reserved_fraction_of_budget": peak_reserved / memory_budget,
            "peak_reserved_fraction_of_device": (
                peak_reserved
                / (
                    memory_budget
                    / float(improved["scheduling"]["memory_fraction"])
                )
            ),
        },
        "capacity": capacity,
        "worker_phases": selected_phases,
        "decision": {
            "cached_graph": "accepted_for_mace_omc_csp",
            "expandable_segments": "accepted_for_mace_variable_cell_bfgs",
            "persistent_chunk_cleanup": "required",
            "mace_omc_csp_policy": "freeze_candidate",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["timing"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
