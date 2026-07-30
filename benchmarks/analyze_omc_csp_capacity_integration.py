#!/usr/bin/env python3
"""Analyze no-probe capacity integration against frozen source and MPS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from benchmarks.analyze_omc_csp_scheduler_development import (
        distribution,
        endpoint_comparison,
    )
except ModuleNotFoundError:
    from analyze_omc_csp_scheduler_development import (
        distribution,
        endpoint_comparison,
    )


PHYSICAL_ENDPOINT_TOLERANCES = {
    "max_energy_error_eV_per_atom": 1e-4,
    "max_final_fmax_error_eV_per_A": 0.03,
    "max_stress_tensor_error_eV_per_A3": 0.01,
    "max_position_rmsd_A": 0.02,
    "max_cell_rmsd_A": 0.02,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _external_seconds(path: Path) -> float:
    return float(
        _load(path.with_suffix(".external.json"))[
            "external_process_wall_seconds"
        ]
    )


def _worker_peak(payload: dict[str, Any]) -> int:
    return max(
        int(chunk["peak_reserved_bytes"])
        for worker in payload["scheduling"]["workers"]
        for chunk in worker["chunks"]
    )


def _endpoint_maximum(summary: dict[str, Any]) -> float:
    return max(
        (
            float(distribution["max"])
            for distribution in summary["metrics"].values()
        ),
        default=0.0,
    )


def physical_endpoint_comparison(
    candidate_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the project's established variable-cell endpoint tolerances."""

    candidate = {str(record["source"]): record for record in candidate_records}
    reference = {str(record["source"]): record for record in reference_records}
    if len(candidate) != len(candidate_records):
        raise ValueError("candidate endpoint source IDs are not unique")
    if len(reference) != len(reference_records):
        raise ValueError("reference endpoint source IDs are not unique")
    common = sorted(set(candidate) & set(reference))
    values: dict[str, list[float]] = {
        "energy_error_eV_per_atom": [],
        "final_fmax_error_eV_per_A": [],
        "stress_tensor_error_eV_per_A3": [],
        "position_rmsd_A": [],
        "cell_rmsd_A": [],
        "absolute_step_difference": [],
    }
    convergence_mismatches = 0
    for source in common:
        actual = candidate[source]
        expected = reference[source]
        actual_positions = np.asarray(actual["positions_A"], dtype=float)
        expected_positions = np.asarray(expected["positions_A"], dtype=float)
        actual_cell = np.asarray(actual["cell_A"], dtype=float)
        expected_cell = np.asarray(expected["cell_A"], dtype=float)
        if actual_positions.shape != expected_positions.shape:
            raise ValueError(f"position shape differs for {source}")
        if actual_cell.shape != expected_cell.shape:
            raise ValueError(f"cell shape differs for {source}")
        atom_count = len(actual_positions)
        if atom_count == 0:
            raise ValueError(f"empty endpoint for {source}")
        values["energy_error_eV_per_atom"].append(
            abs(float(actual["energy_eV"]) - float(expected["energy_eV"]))
            / atom_count
        )
        values["final_fmax_error_eV_per_A"].append(
            abs(
                float(actual["max_force_eV_per_A"])
                - float(expected["max_force_eV_per_A"])
            )
        )
        values["stress_tensor_error_eV_per_A3"].append(
            float(
                np.max(
                    np.abs(
                        np.asarray(actual["stress_eV_per_A3"], dtype=float)
                        - np.asarray(
                            expected["stress_eV_per_A3"],
                            dtype=float,
                        )
                    )
                )
            )
        )
        values["position_rmsd_A"].append(
            float(np.sqrt(np.mean(np.square(actual_positions - expected_positions))))
        )
        values["cell_rmsd_A"].append(
            float(np.sqrt(np.mean(np.square(actual_cell - expected_cell))))
        )
        values["absolute_step_difference"].append(
            abs(int(actual["steps"]) - int(expected["steps"]))
        )
        convergence_mismatches += int(
            bool(actual["converged"]) != bool(expected["converged"])
        )
    metrics = {
        name: distribution(metric_values)
        for name, metric_values in values.items()
        if metric_values
    }
    maximums = {
        "max_energy_error_eV_per_atom": metrics[
            "energy_error_eV_per_atom"
        ]["max"],
        "max_final_fmax_error_eV_per_A": metrics[
            "final_fmax_error_eV_per_A"
        ]["max"],
        "max_stress_tensor_error_eV_per_A3": metrics[
            "stress_tensor_error_eV_per_A3"
        ]["max"],
        "max_position_rmsd_A": metrics["position_rmsd_A"]["max"],
        "max_cell_rmsd_A": metrics["cell_rmsd_A"]["max"],
    }
    failure_counts = {
        name: sum(
            value > PHYSICAL_ENDPOINT_TOLERANCES[name]
            for value in values[metric]
        )
        for name, metric in (
            ("max_energy_error_eV_per_atom", "energy_error_eV_per_atom"),
            ("max_final_fmax_error_eV_per_A", "final_fmax_error_eV_per_A"),
            (
                "max_stress_tensor_error_eV_per_A3",
                "stress_tensor_error_eV_per_A3",
            ),
            ("max_position_rmsd_A", "position_rmsd_A"),
            ("max_cell_rmsd_A", "cell_rmsd_A"),
        )
    }
    failed = [
        name
        for name, threshold in PHYSICAL_ENDPOINT_TOLERANCES.items()
        if maximums[name] > threshold
    ]
    same_source_set = set(candidate) == set(reference)
    if not same_source_set:
        failed.append("same_source_set")
    if convergence_mismatches:
        failed.append("convergence_state_mismatch")
    return {
        "same_source_set": same_source_set,
        "convergence_state_mismatch_count": convergence_mismatches,
        "metrics": metrics,
        "maximums": maximums,
        "failure_counts": failure_counts,
        "tolerances": PHYSICAL_ENDPOINT_TOLERANCES,
        "failed_checks": failed,
        "passed": not failed,
    }


def analyze(completion_path: Path) -> dict[str, Any]:
    completion = _load(completion_path)
    rows = []
    for item in completion["workloads"]:
        offline_path = Path(item["offline_capacity"]["output"])
        source_path = Path(
            item["frozen_references"]["probe_backed_source"]["path"]
        )
        mps_path = Path(item["frozen_references"]["ase_cuda_mps"]["path"])
        offline = _load(offline_path)
        source = _load(source_path)
        mps = _load(mps_path)
        offline_seconds = float(
            item["offline_capacity"]["external_process_wall_seconds"]
        )
        source_seconds = _external_seconds(source_path)
        mps_seconds = _external_seconds(mps_path)
        peak = _worker_peak(offline)
        total_memory = int(offline["environment"]["gpu_total_memory_bytes"])
        source_endpoints = endpoint_comparison(
            offline["records"],
            source["records"],
        )
        source_physical_endpoints = physical_endpoint_comparison(
            offline["records"],
            source["records"],
        )
        mps_endpoints = endpoint_comparison(
            offline["records"],
            mps["records"],
        )
        mps_physical_endpoints = physical_endpoint_comparison(
            offline["records"],
            mps["records"],
        )
        source_mps_physical_endpoints = physical_endpoint_comparison(
            source["records"],
            mps["records"],
        )
        source_endpoint_maximum = _endpoint_maximum(source_endpoints)
        mps_endpoint_maximum = _endpoint_maximum(mps_endpoints)
        scheduling = offline["scheduling"]
        chunk_bound_ratios = [
            int(chunk["peak_reserved_bytes"])
            / int(chunk["predicted_peak_bytes"])
            for worker in scheduling["workers"]
            for chunk in worker["chunks"]
        ]
        predicted_bounds_hold = all(ratio <= 1.0 for ratio in chunk_bound_ratios)
        capacity_gates = {
            "status_complete": offline["status"] == "complete",
            "contract_and_identity_equal": bool(
                item["contract_and_identity_equal"]
            ),
            "offline_policy_selected": (
                scheduling["capacity_planning"]["mode"]
                == "offline_hardware_model"
            ),
            "zero_probe": (
                scheduling["probe"]["system_count"] == 0
                and scheduling["probe"]["model_forward_count"] == 0
            ),
            "no_parent_production_materialization": (
                scheduling["structure_materialization"][
                    "parent_system_count"
                ]
                == 0
            ),
            "predicted_chunk_bounds_hold": predicted_bounds_hold,
            "memory_at_most_85_percent": peak / total_memory <= 0.85,
            "convergence_non_regression": (
                int(offline["converged_count"])
                >= int(source["converged_count"])
                and int(offline["converged_count"])
                >= int(mps["converged_count"])
            ),
        }
        gates = {
            **capacity_gates,
            "source_endpoints_within_established_tolerances": (
                source_physical_endpoints["passed"]
            ),
        }
        rows.append(
            {
                "workload_id": item["workload_id"],
                "offline_external_seconds": offline_seconds,
                "probe_backed_external_seconds": source_seconds,
                "mps_external_seconds": mps_seconds,
                "speedup_over_probe_backed": (
                    source_seconds / offline_seconds
                ),
                "speedup_over_mps": mps_seconds / offline_seconds,
                "offline_converged": int(offline["converged_count"]),
                "probe_backed_converged": int(source["converged_count"]),
                "mps_converged": int(mps["converged_count"]),
                "peak_reserved_bytes": peak,
                "peak_reserved_fraction": peak / total_memory,
                "resident_plan_chunks": int(
                    scheduling["resident_plan_chunk_count"]
                ),
                "execution_chunks": int(
                    scheduling["execution_chunk_count"]
                ),
                "maximum_actual_to_predicted_memory_ratio": max(
                    chunk_bound_ratios
                ),
                "source_endpoint_maximum_difference": (
                    source_endpoint_maximum
                ),
                "mps_endpoint_maximum_difference": (
                    mps_endpoint_maximum
                ),
                "source_physical_endpoints": source_physical_endpoints,
                "mps_physical_endpoints": mps_physical_endpoints,
                "source_mps_physical_endpoints": (
                    source_mps_physical_endpoints
                ),
                "capacity_gates": capacity_gates,
                "all_capacity_gates_passed": all(capacity_gates.values()),
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            }
        )
    offline_total = sum(row["offline_external_seconds"] for row in rows)
    source_total = sum(
        row["probe_backed_external_seconds"] for row in rows
    )
    mps_total = sum(row["mps_external_seconds"] for row in rows)
    return {
        "schema_version": 1,
        "status": "complete",
        "rows": rows,
        "aggregate": {
            "workloads": len(rows),
            "jobs": 2048 * len(rows),
            "offline_external_seconds": offline_total,
            "probe_backed_external_seconds": source_total,
            "mps_external_seconds": mps_total,
            "speedup_over_probe_backed": source_total / offline_total,
            "speedup_over_mps": mps_total / offline_total,
            "maximum_peak_reserved_fraction": max(
                row["peak_reserved_fraction"] for row in rows
            ),
            "maximum_actual_to_predicted_memory_ratio": max(
                row["maximum_actual_to_predicted_memory_ratio"]
                for row in rows
            ),
            "all_capacity_gates_passed": all(
                row["all_capacity_gates_passed"] for row in rows
            ),
            "all_gates_passed": all(
                row["all_gates_passed"] for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.completion)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "capacity_integration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        key
        for key in report["rows"][0]
        if key
        not in {
            "capacity_gates",
            "gates",
            "source_physical_endpoints",
            "mps_physical_endpoints",
            "source_mps_physical_endpoints",
            "all_capacity_gates_passed",
            "all_gates_passed",
        }
    ]
    with (args.output_dir / "capacity_integration.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fieldnames}
            for row in report["rows"]
        )
    lines = [
        "# OMC-CSP Offline Capacity Integration",
        "",
        "| Workload | Offline (s) | Probe (s) | MPS (s) | vs probe | vs MPS | Peak | Capacity | Strict endpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['workload_id']} | "
            f"{row['offline_external_seconds']:.3f} | "
            f"{row['probe_backed_external_seconds']:.3f} | "
            f"{row['mps_external_seconds']:.3f} | "
            f"{row['speedup_over_probe_backed']:.3f}x | "
            f"{row['speedup_over_mps']:.3f}x | "
            f"{100.0 * row['peak_reserved_fraction']:.2f}% | "
            f"{'pass' if row['all_capacity_gates_passed'] else 'fail'} | "
            f"{'pass' if row['all_gates_passed'] else 'fail'} |"
        )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            f"Aggregate speedup over probe-backed source: "
            f"`{aggregate['speedup_over_probe_backed']:.3f}x`.",
            f"Aggregate speedup over MPS: "
            f"`{aggregate['speedup_over_mps']:.3f}x`.",
            f"Maximum peak reserved fraction: "
            f"`{100.0 * aggregate['maximum_peak_reserved_fraction']:.2f}%`.",
            f"Maximum actual/predicted memory ratio: "
            f"`{aggregate['maximum_actual_to_predicted_memory_ratio']:.4f}`.",
            f"Capacity gates: "
            f"`{'pass' if aggregate['all_capacity_gates_passed'] else 'fail'}`.",
            f"Capacity plus strict endpoint gates: "
            f"`{'pass' if aggregate['all_gates_passed'] else 'fail'}`.",
        ]
    )
    (args.output_dir / "capacity_integration.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
