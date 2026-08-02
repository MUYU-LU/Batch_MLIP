#!/usr/bin/env python3
"""Summarize the contract-identical MACE P6000 tensor/MPS16 comparison."""

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
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _records_by_source(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {record["source"]: record for record in payload["records"]}
    if len(records) != len(payload["records"]):
        raise ValueError("duplicate source ID in result records")
    return records


def _assert_fair_contract(
    tensor: dict[str, Any],
    mps: dict[str, Any],
) -> None:
    if tensor["status"] != "complete" or mps["status"] != "complete":
        raise ValueError("both production runs must be complete")
    for key in ("pool_size", "workload_manifest_sha256"):
        if tensor[key] != mps[key]:
            raise ValueError(f"production results disagree on {key}")
    if tensor["checkpoint"]["sha256"] != mps["checkpoint"]["sha256"]:
        raise ValueError("production results use different model files")
    expected = {
        "fmax_eV_per_A": 0.01,
        "max_steps": 3000,
        "cutoff_A": 4.5,
        "model_dtype": "torch.float64",
    }
    tensor_contract = tensor["contract"]
    mps_contract = mps["contract"]
    for key, value in expected.items():
        if tensor_contract[key] != value:
            raise ValueError(f"tensor contract disagrees on {key}")
        mps_value = mps_contract[key]
        if key == "model_dtype":
            mps_value = f"torch.{mps_value}"
        if mps_value != value:
            raise ValueError(f"MPS contract disagrees on {key}")


def _phase_totals(tensor: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}
    for worker in tensor["workers"]:
        for chunk in worker["chunks"]:
            for name, phase in chunk["runtime_profile"]["phases"].items():
                target = totals.setdefault(name, {"count": 0, "seconds": 0.0})
                target["count"] += int(phase["count"])
                target["seconds"] += float(phase["total_seconds"])
    return totals


def _short_horizon_gate(
    tensor_path: Path,
    mps_path: Path,
) -> dict[str, Any]:
    tensor = json.loads(tensor_path.read_text(encoding="utf-8"))
    mps = json.loads(mps_path.read_text(encoding="utf-8"))
    tensor_records = _records_by_source(tensor)
    mps_records = _records_by_source(mps)
    if tensor_records.keys() != mps_records.keys():
        raise ValueError("short-horizon source coverage differs")
    energy = []
    positions = []
    cells = []
    stress = []
    for source in tensor_records:
        left = tensor_records[source]
        right = mps_records[source]
        energy.append(abs(left["energy_eV"] - right["energy_eV"]))
        positions.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["positions_A"])
                        - np.asarray(right["positions_A"])
                    )
                )
            )
        )
        cells.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["cell_A"]) - np.asarray(right["cell_A"])
                    )
                )
            )
        )
        stress.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["stress_eV_per_A3"])
                        - np.asarray(right["stress_eV_per_A3"])
                    )
                )
            )
        )
    return {
        "pool_size": len(tensor_records),
        "max_abs_energy_difference_eV": max(energy),
        "max_abs_position_difference_A": max(positions),
        "max_abs_cell_difference_A": max(cells),
        "max_abs_stress_difference_eV_per_A3": max(stress),
    }


def _b1_diagnostics(
    paths: list[Path],
    tensor_records: dict[str, dict[str, Any]],
    mps_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    for path in sorted(paths):
        b1 = json.loads(path.read_text(encoding="utf-8"))
        source = b1["source"]
        entry: dict[str, Any] = {
            "source": source,
            "b1": {
                key: b1[key]
                for key in (
                    "converged",
                    "steps",
                    "energy_eV",
                    "max_force_eV_per_A",
                )
            },
        }
        position_differences = {}
        for label, record in (
            ("tensor", tensor_records[source]),
            ("mps16", mps_records[source]),
        ):
            position_difference = float(
                np.max(
                    np.abs(
                        np.asarray(b1["positions_A"])
                        - np.asarray(record["positions_A"])
                    )
                )
            )
            position_differences[label] = position_difference
            entry[f"b1_vs_{label}"] = {
                "abs_energy_difference_eV": abs(
                    b1["energy_eV"] - record["energy_eV"]
                ),
                "max_abs_position_difference_A": position_difference,
                "max_abs_cell_difference_A": float(
                    np.max(
                        np.abs(
                            np.asarray(b1["cell_A"])
                            - np.asarray(record["cell_A"])
                        )
                    )
                ),
            }
        entry["closer_endpoint_by_max_position"] = min(
            position_differences,
            key=position_differences.__getitem__,
        )
        diagnostics.append(entry)
    return diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--mps16", type=Path, required=True)
    parser.add_argument("--tensor-smoke", type=Path, required=True)
    parser.add_argument("--mps16-smoke", type=Path, required=True)
    parser.add_argument("--b1-diagnostic", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tensor = json.loads(args.tensor.read_text(encoding="utf-8"))
    mps = json.loads(args.mps16.read_text(encoding="utf-8"))
    _assert_fair_contract(tensor, mps)
    tensor_records = _records_by_source(tensor)
    mps_records = _records_by_source(mps)
    if tensor_records.keys() != mps_records.keys():
        raise ValueError("production source coverage differs")

    tensor_only_converged = 0
    mps_only_converged = 0
    common_nonconverged = 0
    energy_mev_per_atom = []
    energy_mev_per_atom_common_converged = []
    force_differences = []
    for source in tensor_records:
        left = tensor_records[source]
        right = mps_records[source]
        left_converged = bool(left["converged"])
        right_converged = bool(right["converged"])
        tensor_only_converged += left_converged and not right_converged
        mps_only_converged += right_converged and not left_converged
        common_nonconverged += not left_converged and not right_converged
        atom_count = len(left["positions_A"])
        difference = abs(left["energy_eV"] - right["energy_eV"])
        difference *= 1000.0 / atom_count
        energy_mev_per_atom.append(difference)
        if left_converged and right_converged:
            energy_mev_per_atom_common_converged.append(difference)
        force_differences.append(
            abs(left["max_force_eV_per_A"] - right["max_force_eV_per_A"])
        )

    phases = _phase_totals(tensor)
    worker_seconds = sum(float(worker["task_seconds"]) for worker in tensor["workers"])
    tensor_worker_times = [
        float(worker["task_seconds"]) for worker in tensor["workers"]
    ]
    mps_gpu_times = [
        float(worker["result"]["timing"]["wall_seconds"])
        for worker in mps["workers"]
    ]
    peak_allocated = max(
        int(value["allocated_bytes"]) for value in tensor["peak_memory"].values()
    )
    peak_reserved = max(
        int(value["reserved_bytes"]) for value in tensor["peak_memory"].values()
    )
    memory_budget = int(tensor["scheduling"]["memory_budget_bytes_per_gpu"])
    predicted_peak = max(
        int(chunk["predicted_peak_bytes"])
        for chunk in tensor["scheduling"]["resident_plan_chunks"]
    )
    device_memory = memory_budget / float(
        tensor["scheduling"]["memory_fraction"]
    )
    tensor_seconds = float(tensor["timing"]["execution_seconds"])
    mps_seconds = float(mps["timing"]["production_makespan_seconds"])
    tensor_script = float(tensor["timing"]["script_seconds"])
    mps_script = float(mps["timing"]["script_seconds"])

    selected_phases = {}
    for name in (
        "model.forward",
        "graph.mace_atomic_data",
        "graph.mace_collate",
        "graph.state_to_ase",
        "graph.to_device",
        "optimizer.bfgs_update",
        "scheduler.active_compaction",
    ):
        phase = phases[name]
        selected_phases[name] = {
            **phase,
            "worker_time_fraction": float(phase["seconds"]) / worker_seconds,
        }

    diagnostics = _b1_diagnostics(
        args.b1_diagnostic,
        tensor_records,
        mps_records,
    )
    closer_counts = {
        label: sum(
            item["closer_endpoint_by_max_position"] == label
            for item in diagnostics
        )
        for label in ("tensor", "mps16")
    }
    result = {
        "schema_version": 1,
        "status": "complete_with_capacity_rejection",
        "artifacts": {
            "tensor_sha256": _sha256(args.tensor),
            "mps16_sha256": _sha256(args.mps16),
        },
        "workload": {
            "id": tensor["workload_id"],
            "manifest_sha256": tensor["workload_manifest_sha256"],
            "pool_size": tensor["pool_size"],
            "same_source_coverage": True,
        },
        "tensor": {
            "execution_seconds": tensor_seconds,
            "full_script_seconds": tensor_script,
            "systems_per_second": tensor["pool_size"] / tensor_seconds,
            "converged_count": tensor["converged_count"],
            "model_batch_calls": tensor["model_evaluations"],
            "structure_graph_evaluations": tensor["graph_evaluations"],
            "resident_chunk_count": tensor["scheduling"][
                "resident_plan_chunk_count"
            ],
            "execution_chunk_count": tensor["scheduling"][
                "execution_chunk_count"
            ],
            "step_distribution": _quantiles(
                [float(record["steps"]) for record in tensor_records.values()]
            ),
            "worker_max_over_mean": max(tensor_worker_times)
            / float(np.mean(tensor_worker_times)),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_fraction_of_device": peak_reserved / device_memory,
            "memory_budget_bytes": memory_budget,
            "predicted_peak_bytes": predicted_peak,
            "actual_reserved_over_predicted": peak_reserved / predicted_peak,
        },
        "mps16": {
            "production_makespan_seconds": mps_seconds,
            "full_script_seconds": mps_script,
            "systems_per_second": mps["pool_size"] / mps_seconds,
            "converged_count": mps["converged_count"],
            "model_single_structure_calls": mps["model_evaluations"],
            "step_distribution": _quantiles(
                [float(record["steps"]) for record in mps_records.values()]
            ),
            "gpu_shard_max_over_mean": max(mps_gpu_times)
            / float(np.mean(mps_gpu_times)),
            "peak_sampled_device_bytes": mps[
                "peak_gpu_memory_bytes_nvidia_smi"
            ],
        },
        "comparison": {
            "execution_speedup_over_mps16": mps_seconds / tensor_seconds,
            "full_script_speedup_over_mps16": mps_script / tensor_script,
            "convergence_rate_percentage_point_difference": 100.0
            * (tensor["converged_count"] - mps["converged_count"])
            / tensor["pool_size"],
            "convergence_flag_mismatches": (
                tensor_only_converged + mps_only_converged
            ),
            "tensor_only_converged": tensor_only_converged,
            "mps16_only_converged": mps_only_converged,
            "common_nonconverged": common_nonconverged,
            "energy_difference_meV_per_atom": _quantiles(
                energy_mev_per_atom
            ),
            "common_converged_energy_difference_meV_per_atom": _quantiles(
                energy_mev_per_atom_common_converged
            ),
            "max_force_difference_eV_per_A": _quantiles(force_differences),
        },
        "short_horizon_gate": _short_horizon_gate(
            args.tensor_smoke,
            args.mps16_smoke,
        ),
        "b1_long_horizon_diagnostics": {
            "closer_endpoint_counts_by_max_position": closer_counts,
            "records": diagnostics,
        },
        "tensor_worker_phases": selected_phases,
        "decision": {
            "performance_evidence": "accepted_for_exact_contract",
            "endpoint_claim": "schedule_sensitive_local_minima_not_identical",
            "capacity_policy": "rejected_for_production",
            "production_freeze": False,
            "required_follow_up": [
                "allocator_high_water_aware_capacity_margin",
                "validated_mace_cached_graph_path",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
