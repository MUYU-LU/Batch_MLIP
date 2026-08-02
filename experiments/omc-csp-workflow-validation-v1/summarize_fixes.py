#!/usr/bin/env python3
"""Summarize fixes and rejected policy candidates from workflow validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(directory: Path, name: str) -> dict[str, Any]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _endpoint_summary(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    right_records = {record["source"]: record for record in right["records"]}
    differences = []
    for record in left["records"]:
        reference = right_records[record["source"]]
        differences.append(
            abs(record["energy_eV"] - reference["energy_eV"]) / len(record["positions_A"])
        )
    return {
        "maximum_absolute_energy_difference_eV_per_atom": max(differences),
        "count_over_5meV_per_atom": sum(value > 0.005 for value in differences),
        "count_over_1meV_per_atom": sum(value > 0.001 for value in differences),
    }


def _worker_imbalance(payload: dict[str, Any]) -> float:
    workers = payload["scheduling"]["workers"]
    values = [float(worker.get("task_seconds", worker.get("wall_seconds"))) for worker in workers]
    return max(values) / (sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_p64 = _read(args.results, "auto-jaydui-p64-g1.json")
    fixed_p64 = _read(args.results, "fixed-auto-integrated-jaydui-p64-g1.json")
    mix_standalone = _read(args.results, "fixed-auto-mix-p512-g8-target2.json")
    mix_target1 = _read(args.results, "fixed-auto-mix-p512-g8-target1.json")
    mix_integrated = _read(args.results, "fixed-auto-integrated-mix-p512-g8-target2.json")
    mps = _read(args.results, "mps-mix-p512-g8-w8.json")
    large_target2 = _read(args.results, "fixed-auto-obeqix-p2048-g8-target2.json")
    large_target4 = _read(args.results, "fixed-auto-obeqix-p2048-g8-target4.json")
    large_executor = _read(args.results, "fixed-executor-obeqix-p2048-g8-target2.json")
    b1 = _read(args.results, "bfgs-b1-vs-ase-mix-discrepant.json")

    payload = {
        "schema_version": 1,
        "status": "complete_with_bounded_limitations",
        "accepted_fixes": {
            "single_gpu_inner_scheduler": {
                "execution_seconds_before": old_p64["timing"]["execution_seconds"],
                "execution_seconds_after": fixed_p64["timing"]["execution_seconds"],
                "speedup": old_p64["timing"]["execution_seconds"]
                / fixed_p64["timing"]["execution_seconds"],
                "execution_chunks_before": old_p64["scheduling"]["execution_chunk_count"],
                "execution_chunks_after": fixed_p64["scheduling"]["execution_chunk_count"],
                "worker_startup_seconds_after": fixed_p64["scheduling"][
                    "worker_startup_wall_seconds"
                ],
                "optimization_pilot_runs": fixed_p64["scheduling"]["optimization_pilot_runs"],
            },
            "worker_peak_memory_reporting": {
                "source": "worker execution CUDA contexts",
                "p64_maximum_reserved_bytes": max(
                    value["reserved_bytes"] for value in fixed_p64["peak_memory"].values()
                ),
                "parent_context_reported_separately": "parent_peak_memory" in fixed_p64,
                "execution_prediction_count": sum(
                    chunk["predicted_peak_bytes"] is not None
                    for chunk in mix_integrated["scheduling"]["planned_chunks"]
                ),
                "capacity_bounds_present": all(
                    chunk["capacity_bound_bytes"] is not None
                    for chunk in mix_integrated["scheduling"]["planned_chunks"]
                ),
            },
            "ase_compatible_convergence_contract": {
                "smax_eV_per_A3": None,
                "mixed_p512_vs_ase": _endpoint_summary(mix_integrated, mps),
                "b1_diagnostic": b1["comparison"],
            },
            "global_manifest_prefetch": {
                "p2048_standalone_execution_seconds": large_target2["timing"]["execution_seconds"],
                "p2048_integrated_execution_seconds": large_executor["timing"]["execution_seconds"],
                "p2048_speedup": large_target2["timing"]["execution_seconds"]
                / large_executor["timing"]["execution_seconds"],
                "endpoint_comparison": _endpoint_summary(
                    large_executor,
                    large_target2,
                ),
                "prefetch_hit_rate": large_executor["scheduling"]["structure_materialization"][
                    "prefetch_hit_rate"
                ],
                "plug_and_play_entrypoint": mix_integrated["scheduling"]["entrypoint"],
            },
        },
        "mps_comparison": {
            "automatic_execution_seconds": mix_integrated["timing"]["execution_seconds"],
            "mps_warm_production_seconds": mps["timing"]["production_makespan_seconds"],
            "automatic_full_script_seconds": mix_integrated["timing"]["script_seconds"],
            "mps_full_script_seconds": mps["timing"]["script_seconds"],
            "warm_mps_over_cold_automatic_execution_speedup": mps["timing"][
                "production_makespan_seconds"
            ]
            / mix_integrated["timing"]["execution_seconds"],
            "automatic_full_script_speedup_over_mps": mps["timing"]["script_seconds"]
            / mix_integrated["timing"]["script_seconds"],
        },
        "rejected_candidates": {
            "p512_target_one_chunk_per_device": {
                "baseline_seconds": mix_standalone["timing"]["execution_seconds"],
                "candidate_seconds": mix_target1["timing"]["execution_seconds"],
                "baseline_worker_max_over_mean": _worker_imbalance(mix_standalone),
                "candidate_worker_max_over_mean": _worker_imbalance(mix_target1),
                "reason": "slower and less balanced",
            },
            "p2048_target_four_chunks_per_device": {
                "baseline_seconds": large_target2["timing"]["execution_seconds"],
                "candidate_seconds": large_target4["timing"]["execution_seconds"],
                "baseline_worker_max_over_mean": _worker_imbalance(large_target2),
                "candidate_worker_max_over_mean": _worker_imbalance(large_target4),
                "endpoint_comparison": _endpoint_summary(
                    large_target4,
                    large_target2,
                ),
                "reason": "1.5 percent speed gain failed the endpoint gate",
            },
        },
        "bounded_limitations": [
            "multi-GPU process/model cold start remains and is amortized by BatchExecutor reuse",
            "float32 batched trajectories can select different local minima even when B1 matches ASE",
            "outer tail remains workload dependent; unvalidated chunk-count extrapolation is disabled",
            "refill remains evidence gated and was not extrapolated to unmeasured B256 descriptors",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
