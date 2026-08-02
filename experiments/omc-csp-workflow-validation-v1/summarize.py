#!/usr/bin/env python3
"""Summarize correctness, memory, policy reachability, and P512 MPS speed."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auto_summary(
    result: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    scheduling = result["scheduling"]
    records = result["records"]
    source_ids = [record["source"] for record in records]
    manifest = _load(manifest_dir / f"{result['workload_id']}.json")
    expected_source_ids = [job["system_id"] for job in manifest["jobs"]]
    memory = result["peak_memory"]
    parent_reserved = [
        int(value.get("reserved_bytes") or 0)
        for value in memory.values()
    ]
    worker_reserved = [
        int(chunk.get("peak_reserved_bytes") or 0)
        for worker in scheduling.get("workers", [])
        for chunk in worker.get("chunks", [])
    ]
    reserved = worker_reserved or parent_reserved
    total_memory = int(result["environment"]["gpu_total_memory_bytes"])
    maximum_reserved = max(reserved, default=0)
    summary = scheduling["summary"]
    events = [
        event
        for worker in scheduling.get("workers", [])
        for chunk in worker.get("chunks", [])
        for event in (chunk.get("runtime_profile") or {}).get("events", [])
    ]
    phase_seconds: Counter[str] = Counter()
    for worker in scheduling.get("workers", []):
        for chunk in worker.get("chunks", []):
            phases = (chunk.get("runtime_profile") or {}).get("phases", {})
            for name, values in phases.items():
                phase_seconds[name] += float(values.get("total_seconds") or 0.0)
    backend_counts = Counter(
        str(event["selected_backend"])
        for event in events
        if event.get("name") == "neighbor_rebuild"
        and event.get("selected_backend") is not None
    )
    execution_seconds = float(result["timing"]["execution_seconds"])
    startup_seconds = float(result["timing"].get("worker_startup_seconds") or 0.0)
    predicted_peaks = [
        int(chunk.get("predicted_peak_bytes") or 0)
        for chunk in scheduling.get("planned_chunks", [])
    ]
    worker_wall_seconds = [
        float(worker["wall_seconds"])
        for worker in scheduling.get("workers", [])
    ]
    mean_worker_seconds = (
        sum(worker_wall_seconds) / len(worker_wall_seconds)
        if worker_wall_seconds
        else 0.0
    )
    maximum_worker_seconds = max(worker_wall_seconds, default=0.0)
    worker_seconds_sum = sum(worker_wall_seconds)
    materialization_seconds = float(
        scheduling["structure_materialization"]["worker_seconds"]
    )
    return {
        "workload_id": result["workload_id"],
        "pool_size": result["pool_size"],
        "execution_seconds": result["timing"]["execution_seconds"],
        "script_seconds": result["timing"]["script_seconds"],
        "systems_per_second": (
            result["pool_size"] / result["timing"]["execution_seconds"]
        ),
        "converged_count": result["converged_count"],
        "coverage_unique": len(source_ids) == len(set(source_ids)) == result["pool_size"],
        "source_order_equal": source_ids == expected_source_ids,
        "active_gpu_count": scheduling["active_gpu_count"],
        "resident_plan_chunks": scheduling["resident_plan_chunk_count"],
        "execution_chunks": scheduling["execution_chunk_count"],
        "resident_capacities": summary["resident_capacities"],
        "active_refill": scheduling.get(
            "active_refill",
            summary.get("batch_mode") == "active_refill",
        ),
        "refill_reasons": summary["refill_reasons"],
        "capacity_mode": scheduling["capacity_planning"]["mode"],
        "optimization_pilot_runs": scheduling.get("optimization_pilot_runs", 0),
        "optimization_pilot_telemetry_present": (
            "optimization_pilot_runs" in scheduling
        ),
        "neighbor_backend": result["contract"].get("neighbor_backend", "auto"),
        "loader_policy": scheduling["structure_materialization"]["loader_policy"],
        "worker_backend": scheduling["worker_backend"],
        "worker_startup_seconds": startup_seconds,
        "worker_startup_fraction": startup_seconds / execution_seconds,
        "worker_run_seconds": result["timing"].get("worker_execution_seconds"),
        "materialization_worker_seconds": materialization_seconds,
        "materialization_worker_fraction": (
            materialization_seconds / worker_seconds_sum
            if worker_seconds_sum
            else 0.0
        ),
        "phase_worker_seconds": dict(phase_seconds),
        "selected_neighbor_backend_rebuilds": dict(backend_counts),
        "neighbor_rebuild_events": sum(backend_counts.values()),
        "model_evaluation_events": sum(
            event.get("name") == "model_evaluation" for event in events
        ),
        "neighbor_rebuild_fraction": (
            sum(backend_counts.values())
            / max(
                1,
                sum(event.get("name") == "model_evaluation" for event in events),
            )
        ),
        "maximum_predicted_peak_bytes": max(predicted_peaks, default=0),
        "parent_reported_maximum_reserved_bytes": max(
            parent_reserved,
            default=0,
        ),
        "maximum_reserved_bytes": maximum_reserved,
        "maximum_reserved_fraction": maximum_reserved / total_memory,
        "memory_gate": maximum_reserved / total_memory <= 0.85,
        "worker_wall_seconds": worker_wall_seconds,
        "worker_wall_max_over_mean": (
            maximum_worker_seconds / mean_worker_seconds
            if mean_worker_seconds
            else 0.0
        ),
        "worker_tail_fraction": (
            (maximum_worker_seconds - mean_worker_seconds)
            / maximum_worker_seconds
            if maximum_worker_seconds
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--frozen-auto-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = (
        "auto-jaydui-p64-g1",
        "auto-mix-p512-g8",
        "auto-obeqix-p2048-g8",
    )
    automatic = {
        name: _auto_summary(
            _load(args.results / f"{name}.json"),
            args.manifest_dir,
        )
        for name in names
    }
    mix_auto_raw = _load(args.results / "auto-mix-p512-g8.json")
    mix_mps = _load(args.results / "mps-mix-p512-g8-w8.json")
    auto_records = {item["source"]: item for item in mix_auto_raw["records"]}
    mps_records = {item["source"]: item for item in mix_mps["records"]}
    common = sorted(set(auto_records) & set(mps_records))
    energy_deltas_per_atom = []
    for source in common:
        auto = auto_records[source]
        mps = mps_records[source]
        atom_count = max(1, len(auto["positions_A"]))
        energy_deltas_per_atom.append(
            abs(float(auto["energy_eV"]) - float(mps["energy_eV"]))
            / atom_count
        )

    mix_auto_seconds = automatic["auto-mix-p512-g8"]["execution_seconds"]
    mix_mps_seconds = float(
        mix_mps["timing"]["production_makespan_seconds"]
    )
    endpoint_failures = sum(
        value > 0.005 for value in energy_deltas_per_atom
    )
    frozen_auto = _load(args.frozen_auto_reference)
    frozen_records = {
        item["source"]: item for item in frozen_auto["records"]
    }
    frozen_common = sorted(set(auto_records) & set(frozen_records))
    frozen_energy_deltas = [
        abs(
            float(auto_records[source]["energy_eV"])
            - float(frozen_records[source]["energy_eV"])
        )
        / max(1, len(auto_records[source]["positions_A"]))
        for source in frozen_common
    ]
    issues = []
    single = automatic["auto-jaydui-p64-g1"]
    if single["refill_reasons"] == [
        "multi-GPU refill has no accepted scientific policy"
    ]:
        issues.append(
            {
                "severity": "high",
                "component": "API routing / inner scheduler",
                "finding": (
                    "A source-backed workload assigned one GPU used the "
                    "multi-GPU path and unconditionally disabled refill."
                ),
            }
        )
    if single["resident_plan_chunks"] == 1 and single["execution_chunks"] > 1:
        issues.append(
            {
                "severity": "medium",
                "component": "outer scheduler",
                "finding": (
                    "The one-GPU small pool split one memory-safe resident "
                    "batch into multiple execution chunks."
                ),
            }
        )
    if any(
        not item["optimization_pilot_telemetry_present"]
        for item in automatic.values()
    ):
        issues.append(
            {
                "severity": "low",
                "component": "runtime telemetry",
                "finding": (
                    "The nonpersistent source-backed result omitted the "
                    "explicit optimization_pilot_runs field."
                ),
            }
        )
    if any(not item["memory_gate"] for item in automatic.values()):
        issues.append(
            {
                "severity": "critical",
                "component": "capacity planning",
                "finding": "At least one automatic run exceeded 85% reserved memory.",
            }
        )
    if any(not item["source_order_equal"] for item in automatic.values()):
        issues.append(
            {
                "severity": "critical",
                "component": "result reassembly",
                "finding": "At least one automatic result changed manifest order.",
            }
        )
    if any(
        item["maximum_reserved_bytes"]
        > 2 * max(1, item["parent_reported_maximum_reserved_bytes"])
        for item in automatic.values()
    ):
        issues.append(
            {
                "severity": "high",
                "component": "memory telemetry",
                "finding": (
                    "Top-level source-backed peak_memory reported the parent "
                    "CUDA context rather than process-worker peaks."
                ),
            }
        )
    if any(item["optimization_pilot_runs"] for item in automatic.values()):
        issues.append(
            {
                "severity": "high",
                "component": "plug-and-play planning",
                "finding": "An automatic run executed an optimization pilot.",
            }
        )
    mix = automatic["auto-mix-p512-g8"]
    if mix["worker_startup_fraction"] > 0.4:
        issues.append(
            {
                "severity": "high",
                "component": "worker lifecycle",
                "finding": (
                    "Cold worker startup consumed more than 40% of mixed-P512 "
                    "automatic execution time."
                ),
            }
        )
    if mix["execution_chunks"] >= 4 * mix["resident_plan_chunks"]:
        issues.append(
            {
                "severity": "medium",
                "component": "outer scheduler",
                "finding": (
                    "The mixed P512 outer scheduler fragmented each safe "
                    "resident plan into four execution chunks on average."
                ),
            }
        )
    if (
        mix["maximum_predicted_peak_bytes"]
        > 2 * max(1, mix["maximum_reserved_bytes"])
    ):
        issues.append(
            {
                "severity": "medium",
                "component": "execution-chunk memory telemetry",
                "finding": (
                    "Subdivided mixed-P512 chunks retained the parent "
                    "resident-batch peak prediction instead of a child "
                    "prediction."
                ),
            }
        )
    if endpoint_failures:
        issues.append(
            {
                "severity": "high",
                "component": "batched BFGS versus ASE endpoint equivalence",
                "finding": (
                    f"{endpoint_failures} mixed-P512 endpoints exceeded "
                    "5 meV/atom versus ASE BFGS under MPS; the same current "
                    "endpoints remain within 5 meV/atom of the frozen "
                    "automatic scheduler result."
                ),
            }
        )
    if any(value > 0.005 for value in frozen_energy_deltas):
        issues.append(
            {
                "severity": "critical",
                "component": "automatic scheduler numerical stability",
                "finding": (
                    "Current automatic endpoints exceeded 5 meV/atom versus "
                    "the frozen automatic result."
                ),
            }
        )
    large = automatic["auto-obeqix-p2048-g8"]
    if large["worker_wall_max_over_mean"] > 1.2:
        issues.append(
            {
                "severity": "medium",
                "component": "outer scheduler tail balance",
                "finding": (
                    "The slowest OBEQIX P2048 worker exceeded mean worker "
                    "time by more than 20%."
                ),
            }
        )

    output = {
        "schema_version": 1,
        "status": "complete",
        "automatic": automatic,
        "mixed_p512_vs_mps8": {
            "automatic_execution_seconds": mix_auto_seconds,
            "automatic_worker_run_seconds": mix["worker_run_seconds"],
            "mps_production_makespan_seconds": mix_mps_seconds,
            "mixed_scope_mps_production_over_auto_cold_execution": (
                mix_mps_seconds / mix_auto_seconds
            ),
            "automatic_worker_run_speedup_over_mps_production": (
                mix_mps_seconds / float(mix["worker_run_seconds"])
            ),
            "automatic_full_script_speedup_over_mps": (
                float(mix_mps["timing"]["script_seconds"])
                / float(mix_auto_raw["timing"]["script_seconds"])
            ),
            "automatic_converged_count": mix_auto_raw["converged_count"],
            "mps_converged_count": mix_mps["converged_count"],
            "common_endpoint_count": len(common),
            "maximum_energy_delta_eV_per_atom": max(
                energy_deltas_per_atom,
                default=0.0,
            ),
            "energy_5meV_per_atom_gate": all(
                value <= 0.005 for value in energy_deltas_per_atom
            ),
            "energy_5meV_per_atom_failures": endpoint_failures,
            "automatic_peak_reserved_bytes": mix["maximum_reserved_bytes"],
            "mps_peak_memory_bytes_nvidia_smi": mix_mps[
                "peak_gpu_memory_bytes_nvidia_smi"
            ],
        },
        "mixed_p512_current_vs_frozen_automatic": {
            "common_endpoint_count": len(frozen_common),
            "maximum_energy_delta_eV_per_atom": max(
                frozen_energy_deltas,
                default=0.0,
            ),
            "energy_5meV_per_atom_failures": sum(
                value > 0.005 for value in frozen_energy_deltas
            ),
            "current_execution_seconds": mix_auto_seconds,
            "frozen_execution_seconds": frozen_auto["timing"][
                "execution_seconds"
            ],
        },
        "issues": issues,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
