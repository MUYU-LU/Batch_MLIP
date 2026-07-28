#!/usr/bin/env python3
"""Build the shipped refill evidence table from validated unique workloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_STATE_SHA256 = (
    "200b8a378d848c00b9a690583e34e49dacad79975cf3c7c4c92c39a0e24d620d"
)


def _record(row: dict[str, Any]) -> dict[str, Any]:
    scientifically_valid = (
        row["endpoint_gate_passed"]
        and row["memory_gate_passed"]
        and row["active_converged"] == 256
        and row["refill_converged"] == 256
    )
    use_refill = (
        row["refill_storage"] == "slots"
        and row["refill_speedup"] >= 1.05
        and scientifically_valid
    )
    return {
        "family": row["family"],
        "split": row["split"],
        "pool_size": row["pool_size"],
        "resident_capacity": row["batch_size"],
        "mean_atom_count": row["mean_atom_count"],
        "atom_count_cv": row["atom_count_cv"],
        "mean_edge_count": row["mean_edge_count"],
        "edge_count_cv": row["edge_count_cv"],
        "homogeneous_atom_count": row["homogeneous_atom_count"],
        "storage": row["refill_storage"],
        "measured_refill_speedup": row["refill_speedup"],
        "refill_peak_reserved_fraction": row[
            "refill_peak_reserved_fraction"
        ],
        "endpoint_gate_passed": row["endpoint_gate_passed"],
        "all_jobs_converged": (
            row["active_converged"] == 256
            and row["refill_converged"] == 256
        ),
        "selected_mode": "refill" if use_refill else "active",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary["contract_validation"] != "passed":
        raise ValueError("calibration contract validation did not pass")
    if summary["locked_policy_validation"]["speed_predictions_correct"] != 9:
        raise ValueError("locked speed predictor failed held-out validation")
    records = [_record(row) for row in summary["rows"]]
    payload = {
        "schema_version": 1,
        "policy_id": "atombit-smooth-rms-bfgs-h100-refill-v1",
        "source_experiment": "refill-offline-predictor",
        "source_baseline_commit": summary["baseline_commit"],
        "contract": {
            "calculator_type": (
                "batch_mlip.models.potential.AtomBitBatchCalculator"
            ),
            "model_type": "src.model.AtomBitModel",
            "model_parameter_count": 344653,
            "model_state_sha256": MODEL_STATE_SHA256,
            "model_dtype": "torch.float32",
            "force_mode": "autograd",
            "neighbor_backend": "auto",
            "e0_offsets": False,
            "model_call_kwargs": {},
            "optimizer_type": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "variable_cell": True,
            "cell_filter_type": "FrechetCellFilter",
            "linear_algebra_backend": "auto",
            "fmax_eV_per_A": 0.05,
            "smax_eV_per_A3": None,
            "max_steps": 500,
            "cutoff_A": 6.0,
            "skin_A": 0.5,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_total_memory_bytes": 85017886720,
            "torch": "2.9.1+cu128",
            "cuda": "12.8",
            "cpu_threads": 1,
            "cublas_workspace_config": ":4096:8",
            "allocator": "expandable_segments:True",
        },
        "matching": {
            "atom_relative_tolerance": 0.05,
            "edge_relative_tolerance": 0.25,
            "requires_exact_pool_size": True,
            "requires_exact_resident_capacity": True,
            "requires_homogeneous_atom_count": True,
        },
        "selection": {
            "minimum_refill_speedup": 1.05,
            "maximum_reserved_fraction": 0.85,
            "endpoint_energy_difference_meV_per_atom_max": 5.0,
            "fallback_mode": "active",
        },
        "records": records,
        "validation": summary["locked_policy_validation"],
        "limitations": summary["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
