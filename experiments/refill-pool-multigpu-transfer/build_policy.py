#!/usr/bin/env python3
"""Extend refill policy v1 with validated single-GPU pool-size records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _record(row: dict[str, Any]) -> dict[str, Any]:
    valid = (
        row["endpoint_gate_passed"]
        and row["memory_gate_passed"]
        and row["all_jobs_converged"]
    )
    use_refill = row["refill_speedup"] >= 1.05 and valid
    return {
        "family": row["family"],
        "split": row["split"],
        "pool_size": row["pool_size"],
        "resident_capacity": row["resident_capacity"],
        "mean_atom_count": row["mean_atom_count"],
        "atom_count_cv": row["atom_count_cv"],
        "mean_edge_count": row["mean_edge_count"],
        "edge_count_cv": row["edge_count_cv"],
        "homogeneous_atom_count": True,
        "storage": "slots",
        "measured_refill_speedup": row["refill_speedup"],
        "refill_peak_reserved_fraction": row[
            "refill_peak_reserved_fraction"
        ],
        "endpoint_gate_passed": row["endpoint_gate_passed"],
        "all_jobs_converged": row["all_jobs_converged"],
        "selected_mode": "refill" if use_refill else "active",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_policy.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validation = summary["validation"]
    if validation["single_gpu_speed_predictions_correct"] != 4:
        raise ValueError("single-GPU held-out speed validation failed")
    if validation["single_gpu_scientific_gate_failures"] != 0:
        raise ValueError("single-GPU held-out scientific validation failed")
    if validation["multi_gpu_policy_accepted"]:
        raise ValueError("multi-GPU records must not enter this policy")

    additions = [
        _record(row)
        for row in summary["rows"]
        if row["gpu_count"] == 1
    ]
    keys = {
        (
            record["family"],
            record["pool_size"],
            record["resident_capacity"],
        )
        for record in base["records"]
    }
    if any(
        (
            record["family"],
            record["pool_size"],
            record["resident_capacity"],
        )
        in keys
        for record in additions
    ):
        raise ValueError("pool-transfer record duplicates base evidence")

    payload = {
        **base,
        "policy_id": "atombit-smooth-rms-bfgs-h100-refill-v2",
        "source_experiments": [
            "refill-offline-predictor",
            "refill-pool-multigpu-transfer",
        ],
        "records": base["records"] + additions,
        "validation": {
            "r256_family_holdout": base["validation"],
            "pool_transfer_family_holdout": {
                "speed_predictions_correct": validation[
                    "single_gpu_speed_predictions_correct"
                ],
                "speed_predictions_total": validation[
                    "single_gpu_speed_predictions_total"
                ],
                "scientific_gate_failures": validation[
                    "single_gpu_scientific_gate_failures"
                ],
            },
            "multi_gpu_transfer": {
                "accepted": False,
                "speed_predictions_correct": validation[
                    "multi_gpu_speed_predictions_correct"
                ],
                "speed_predictions_total": validation[
                    "multi_gpu_speed_predictions_total"
                ],
                "scientific_gate_failures": validation[
                    "multi_gpu_scientific_gate_failures"
                ],
            },
        },
        "limitations": [
            "One timing observation per matrix point.",
            "Prediction is restricted to the exact execution contract.",
            "Pool size and resident capacity must match measured records.",
            "Multi-GPU refill failed transfer validation and is not selected.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
