#!/usr/bin/env python3
"""Audit and summarize the OMC-CSP scheduler development baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

AUTO_METHOD = "current_auto"
MPS_METHOD = "ase_cuda_mps"
EXPECTED_WORKLOADS = 23
EXPECTED_WORKLOADS_BY_SPLIT = {
    "development": 23,
    "validation": 10,
    "test": 16,
}
MEMORY_LIMIT_FRACTION = 0.85


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Iterable[float | int]) -> dict[str, float]:
    """Return stable nearest-rank summary statistics."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("distribution requires at least one value")

    def percentile(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _command_output(method: dict[str, Any]) -> Path:
    command = list(method["command"])
    try:
        output_index = command.index("--output") + 1
    except ValueError as error:
        raise ValueError("method command does not contain --output") from error
    return Path(command[output_index])


def _records_by_source(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        source = str(record["source"])
        if source in indexed:
            raise ValueError(f"duplicate endpoint source: {source}")
        indexed[source] = record
    return indexed


def _array_max_abs_difference(left: Any, right: Any) -> float:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("endpoint array shapes differ")
        return max(
            (
                _array_max_abs_difference(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            ),
            default=0.0,
        )
    if isinstance(left, float | int) and isinstance(right, float | int):
        return abs(float(left) - float(right))
    raise ValueError("endpoint values are not comparable numeric arrays")


def endpoint_comparison(
    automatic_records: list[dict[str, Any]],
    mps_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare endpoints by immutable source ID rather than result order."""

    automatic = _records_by_source(automatic_records)
    mps = _records_by_source(mps_records)
    automatic_sources = set(automatic)
    mps_sources = set(mps)
    common_sources = sorted(automatic_sources & mps_sources)
    metrics: dict[str, list[float]] = {
        "absolute_energy_eV": [],
        "absolute_max_force_eV_per_A": [],
        "absolute_max_stress_eV_per_A3": [],
        "max_position_component_A": [],
        "max_cell_component_A": [],
        "max_stress_component_eV_per_A3": [],
        "absolute_step_difference": [],
    }
    convergence_mismatches = 0
    convergence_categories = Counter(
        {
            "both_converged": 0,
            "automatic_only_converged": 0,
            "mps_only_converged": 0,
            "neither_converged": 0,
        }
    )
    for source in common_sources:
        left = automatic[source]
        right = mps[source]
        automatic_converged = bool(left["converged"])
        mps_converged = bool(right["converged"])
        convergence_mismatches += int(automatic_converged != mps_converged)
        if automatic_converged and mps_converged:
            convergence_categories["both_converged"] += 1
        elif automatic_converged:
            convergence_categories["automatic_only_converged"] += 1
        elif mps_converged:
            convergence_categories["mps_only_converged"] += 1
        else:
            convergence_categories["neither_converged"] += 1
        metrics["absolute_energy_eV"].append(
            abs(float(left["energy_eV"]) - float(right["energy_eV"]))
        )
        metrics["absolute_max_force_eV_per_A"].append(
            abs(float(left["max_force_eV_per_A"]) - float(right["max_force_eV_per_A"]))
        )
        metrics["absolute_max_stress_eV_per_A3"].append(
            abs(float(left["max_abs_stress_eV_per_A3"]) - float(right["max_abs_stress_eV_per_A3"]))
        )
        metrics["max_position_component_A"].append(
            _array_max_abs_difference(left["positions_A"], right["positions_A"])
        )
        metrics["max_cell_component_A"].append(
            _array_max_abs_difference(left["cell_A"], right["cell_A"])
        )
        metrics["max_stress_component_eV_per_A3"].append(
            _array_max_abs_difference(
                left["stress_eV_per_A3"],
                right["stress_eV_per_A3"],
            )
        )
        metrics["absolute_step_difference"].append(abs(int(left["steps"]) - int(right["steps"])))
    return {
        "automatic_record_count": len(automatic),
        "mps_record_count": len(mps),
        "common_source_count": len(common_sources),
        "same_source_set": automatic_sources == mps_sources,
        "automatic_only_sources": sorted(automatic_sources - mps_sources),
        "mps_only_sources": sorted(mps_sources - automatic_sources),
        "convergence_state_mismatch_count": convergence_mismatches,
        "convergence_categories": dict(convergence_categories),
        "metrics": {name: distribution(values) for name, values in metrics.items() if values},
        "threshold_gate_applied": False,
        "threshold_gate_reason": (
            "The frozen epoch contract requires endpoint retention and "
            "convergence-count non-regression but does not declare cross-"
            "optimizer endpoint tolerances."
        ),
    }


def _contract_audit(
    automatic: dict[str, Any],
    mps: dict[str, Any],
) -> dict[str, Any]:
    auto_contract = automatic["contract"]
    mps_contract = mps["contract"]
    auto_checkpoint = automatic["checkpoint"]
    mps_checkpoint = mps["checkpoint"]
    checks = {
        "same_checkpoint_sha256": (auto_checkpoint["sha256"] == mps_checkpoint["sha256"]),
        "same_cutoff": math.isclose(
            float(auto_contract["cutoff_A"]),
            float(mps_contract["cutoff_A"]),
        ),
        "same_fmax": math.isclose(
            float(auto_contract["fmax_eV_per_A"]),
            float(mps_contract["fmax_eV_per_A"]),
        ),
        "same_max_steps": (int(auto_contract["max_steps"]) == int(mps_contract["max_steps"])),
        "same_optimizer_state_dtype": (
            str(auto_contract["optimizer_dtype"]).lower().replace("torch.", "")
            == str(mps_contract["optimizer_dtype"]).lower().replace("torch.", "")
        ),
        "both_bfgs": (
            "bfgs" in str(auto_contract["optimizer"]).lower()
            and "bfgs" in str(mps_contract["optimizer"]).lower()
        ),
        "both_frechet_cell_filter": (
            "frechet" in str(auto_contract["cell_filter"]).lower()
            and "frechet" in str(mps_contract["cell_filter"]).lower()
        ),
        "automatic_deterministic": bool(
            automatic["reproducibility"]["torch_deterministic_algorithms"]
        ),
        "mps_deterministic": bool(mps_contract["deterministic"]),
    }
    return {
        "checks": checks,
        "same_requested_contract": all(checks.values()),
    }


def _auto_memory(automatic: dict[str, Any]) -> dict[str, Any]:
    scheduling = automatic["scheduling"]
    workers = automatic.get("workers", scheduling["workers"])
    worker_peak_by_device: dict[str, int] = defaultdict(int)
    predicted_actual_ratios: list[float] = []
    for worker in workers:
        device = str(worker["device"])
        for chunk in worker["chunks"]:
            actual = int(chunk["peak_reserved_bytes"])
            worker_peak_by_device[device] = max(
                worker_peak_by_device[device],
                actual,
            )
            if actual > 0:
                predicted_actual_ratios.append(int(chunk["predicted_peak_bytes"]) / actual)

    parent_peak_by_device = {
        str(device): int(values["reserved_bytes"])
        for device, values in automatic.get("peak_memory", {}).items()
    }
    probe = scheduling["probe"]
    probe_device = str(probe["device"])
    parent_peak_by_device[probe_device] = max(
        parent_peak_by_device.get(probe_device, 0),
        int(probe["peak_reserved_bytes"]),
    )
    devices = set(worker_peak_by_device) | set(parent_peak_by_device)
    tensor_conservative_by_device = {
        device: (worker_peak_by_device.get(device, 0) + parent_peak_by_device.get(device, 0))
        for device in devices
    }
    recovery = automatic.get("tail_recovery", {})
    recovery_worker_peak = {
        str(device): int(value)
        for device, value in recovery.get(
            "peak_reserved_bytes_by_device",
            {},
        ).items()
    }
    recovery_parent_reserved = {
        str(device): int(value)
        for device, value in recovery.get(
            "parent_reserved_bytes_during_recovery_by_device",
            {},
        ).items()
    }
    recovery_devices = set(recovery_worker_peak) | set(recovery_parent_reserved)
    recovery_conservative_by_device = {
        device: (
            recovery_worker_peak.get(device, 0)
            + recovery_parent_reserved.get(device, 0)
        )
        for device in recovery_devices
    }
    all_devices = devices | recovery_devices
    conservative_by_device = {
        device: max(
            tensor_conservative_by_device.get(device, 0),
            recovery_conservative_by_device.get(device, 0),
        )
        for device in all_devices
    }
    gpu_total = int(automatic["environment"]["gpu_total_memory_bytes"])
    max_worker_peak = max(worker_peak_by_device.values(), default=0)
    max_conservative_peak = max(conservative_by_device.values(), default=0)
    return {
        "gpu_total_bytes": gpu_total,
        "worker_peak_reserved_bytes_by_device": dict(sorted(worker_peak_by_device.items())),
        "parent_peak_reserved_bytes_by_device": dict(sorted(parent_peak_by_device.items())),
        "tensor_conservative_peak_reserved_bytes_by_device": dict(
            sorted(tensor_conservative_by_device.items())
        ),
        "recovery_worker_peak_reserved_bytes_by_device": dict(
            sorted(recovery_worker_peak.items())
        ),
        "recovery_parent_reserved_bytes_by_device": dict(
            sorted(recovery_parent_reserved.items())
        ),
        "recovery_conservative_peak_reserved_bytes_by_device": dict(
            sorted(recovery_conservative_by_device.items())
        ),
        "conservative_peak_reserved_bytes_by_device": dict(sorted(conservative_by_device.items())),
        "max_worker_peak_reserved_bytes": max_worker_peak,
        "max_worker_peak_fraction": max_worker_peak / gpu_total,
        "max_conservative_peak_reserved_bytes": max_conservative_peak,
        "max_conservative_peak_fraction": max_conservative_peak / gpu_total,
        "predicted_to_actual_chunk_peak_ratio": distribution(predicted_actual_ratios),
        "accounting": (
            "Within each temporal stage, the per-device safety upper bound "
            "adds concurrent parent and worker allocations. Tensor execution "
            "and tail recovery are sequential, so their stage peaks are "
            "combined by maximum rather than addition."
        ),
    }


def _phase_summary(automatic: dict[str, Any]) -> dict[str, Any]:
    workers = automatic.get("workers", automatic["scheduling"]["workers"])
    selected_names = {
        "neighbor_update": "calculator.neighbor_update",
        "model_forward": "model.forward",
        "model_autograd": "model.autograd",
        "bfgs_update": "optimizer.bfgs_update",
        "active_compaction": "scheduler.active_compaction",
    }
    seconds = dict.fromkeys(selected_names, 0.0)
    neighbor_search_seconds = 0.0
    chunk_wall_seconds = 0.0
    for worker in workers:
        for chunk in worker["chunks"]:
            chunk_wall_seconds += float(chunk["wall_seconds"])
            phases = chunk["runtime_profile"]["phases"]
            for group, name in selected_names.items():
                if name in phases:
                    seconds[group] += float(phases[name]["total_seconds"])
            if "graph.neighbor_search" in phases:
                neighbor_search_seconds += float(phases["graph.neighbor_search"]["total_seconds"])
    selected_total = sum(seconds.values())
    unprofiled = max(0.0, chunk_wall_seconds - selected_total)
    seconds["unprofiled"] = unprofiled
    fractions = {
        name: value / chunk_wall_seconds if chunk_wall_seconds else 0.0
        for name, value in seconds.items()
    }
    return {
        "chunk_wall_work_seconds": chunk_wall_seconds,
        "exclusive_phase_seconds": seconds,
        "exclusive_phase_fraction": fractions,
        "neighbor_search_diagnostic_seconds": neighbor_search_seconds,
        "accounting": (
            "Fractions use non-overlapping top-level phase names over summed "
            "chunk work time. Detailed nested phases are diagnostic only."
        ),
    }


def _load_imbalance(values: list[float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0, "max_over_mean": 0.0}
    mean = statistics.fmean(values)
    return {
        "minimum": min(values),
        "mean": mean,
        "maximum": max(values),
        "max_over_mean": max(values) / mean if mean else 0.0,
    }


def _timing_summary(
    automatic: dict[str, Any],
    mps: dict[str, Any],
    *,
    automatic_external: float,
    mps_external: float,
) -> dict[str, Any]:
    auto_timing = automatic["timing"]
    auto_script = float(auto_timing["script_seconds"])
    auto_execution = float(auto_timing["execution_seconds"])
    profiling = float(auto_timing["profiling_seconds"])
    startup = float(auto_timing["worker_startup_seconds"])
    worker_run = float(auto_timing["worker_execution_seconds"])
    recovery = float(auto_timing.get("tail_recovery_seconds", 0.0))
    scheduler_other = max(
        0.0,
        auto_execution - profiling - startup - worker_run - recovery,
    )
    auto_components = {
        "external_wrapper": max(0.0, automatic_external - auto_script),
        "structure_model_setup_and_output": max(
            0.0,
            auto_script - auto_execution,
        ),
        "profiling": profiling,
        "worker_startup": startup,
        "scheduler_other": scheduler_other,
        "worker_execution": worker_run,
        "tail_recovery": recovery,
    }
    mps_script = float(mps["timing"]["script_seconds"])
    mps_production = float(mps["timing"]["production_makespan_seconds"])
    mps_components = {
        "external_wrapper": max(0.0, mps_external - mps_script),
        "setup_and_output": max(0.0, mps_script - mps_production),
        "production": mps_production,
    }
    auto_worker_walls = [
        float(worker["wall_seconds"])
        for worker in automatic.get("workers", automatic["scheduling"]["workers"])
    ]
    mps_gpu_walls = [float(worker["result"]["timing"]["wall_seconds"]) for worker in mps["workers"]]
    return {
        "automatic_external_seconds": automatic_external,
        "mps_external_seconds": mps_external,
        "external_speedup_over_mps": mps_external / automatic_external,
        "automatic_production_seconds": worker_run + recovery,
        "mps_production_seconds": mps_production,
        "production_speedup_over_mps": mps_production / (worker_run + recovery),
        "automatic_components_seconds": auto_components,
        "mps_components_seconds": mps_components,
        "automatic_worker_load": _load_imbalance(auto_worker_walls),
        "mps_gpu_load": _load_imbalance(mps_gpu_walls),
    }


def _scope(workload_id: str) -> str:
    if "-INTER-" in workload_id:
        return "inter_family"
    if "-INTRA-" in workload_id:
        return "intra_family"
    return "unknown"


def analyze_workload(run: dict[str, Any]) -> dict[str, Any]:
    """Analyze one paired automatic and MPS workload run."""

    automatic_path = _command_output(run["methods"][AUTO_METHOD])
    mps_path = _command_output(run["methods"][MPS_METHOD])
    automatic = _load(automatic_path)
    mps = _load(mps_path)
    expected_workload_id = str(run["workload_id"])
    expected_manifest_hash = str(run["workload_manifest_sha256"])
    pool_size = int(run["pool_size"])
    endpoints = endpoint_comparison(automatic["records"], mps["records"])
    contract = _contract_audit(automatic, mps)
    memory = _auto_memory(automatic)
    phases = _phase_summary(automatic)
    timing = _timing_summary(
        automatic,
        mps,
        automatic_external=float(run["methods"][AUTO_METHOD]["external_process_wall_seconds"]),
        mps_external=float(run["methods"][MPS_METHOD]["external_process_wall_seconds"]),
    )
    identity_checks = {
        "automatic_status_complete": automatic["status"] == "complete",
        "mps_status_complete": mps["status"] == "complete",
        "automatic_returncode_zero": (int(run["methods"][AUTO_METHOD]["returncode"]) == 0),
        "mps_returncode_zero": (int(run["methods"][MPS_METHOD]["returncode"]) == 0),
        "automatic_workload_id_matches": (automatic["workload_id"] == expected_workload_id),
        "mps_workload_id_matches": mps["workload_id"] == expected_workload_id,
        "automatic_manifest_hash_matches": (
            automatic["workload_manifest_sha256"] == expected_manifest_hash
        ),
        "mps_manifest_hash_matches": (mps["workload_manifest_sha256"] == expected_manifest_hash),
        "automatic_pool_size_matches": int(automatic["pool_size"]) == pool_size,
        "mps_pool_size_matches": int(mps["pool_size"]) == pool_size,
        "endpoint_source_sets_match": bool(endpoints["same_source_set"]),
        "automatic_exact_job_count": (int(endpoints["automatic_record_count"]) == pool_size),
        "mps_exact_job_count": int(endpoints["mps_record_count"]) == pool_size,
    }
    automatic_converged = int(automatic["converged_count"])
    mps_converged = int(mps["converged_count"])
    gates = {
        "identity_and_coverage": all(identity_checks.values()),
        "same_requested_contract": bool(contract["same_requested_contract"]),
        "no_oom": (automatic["status"] == "complete" and mps["status"] == "complete"),
        "automatic_memory_at_most_85_percent": (
            memory["max_conservative_peak_fraction"] <= MEMORY_LIMIT_FRACTION
        ),
        "automatic_convergence_not_below_mps": (automatic_converged >= mps_converged),
        "automatic_external_faster_than_mps": (
            timing["automatic_external_seconds"] < timing["mps_external_seconds"]
        ),
    }
    schedule = automatic["scheduling"]
    policy = schedule["policy_manifest"]
    recovery = automatic.get("tail_recovery", {})
    row = {
        "workload_id": expected_workload_id,
        "pool_size": pool_size,
        "scope": _scope(expected_workload_id),
        "gpu_count": int(schedule["gpu_count"]),
        "identity": identity_checks,
        "contract": contract,
        "timing": timing,
        "memory": memory,
        "phases": phases,
        "convergence": {
            "automatic": automatic_converged,
            "mps": mps_converged,
            "difference": automatic_converged - mps_converged,
            "automatic_steps": distribution(
                int(record["steps"]) for record in automatic["records"]
            ),
            "mps_steps": distribution(int(record["steps"]) for record in mps["records"]),
        },
        "endpoints": endpoints,
        "schedule": {
            "decision": schedule["decision"],
            "cost_bucket_count": int(policy["outer_scheduler"]["cost_bucket_count"]),
            "resident_wave_count": int(policy["outer_scheduler"]["resident_wave_count"]),
            "resident_plan_chunk_count": int(schedule["resident_plan_chunk_count"]),
            "execution_chunk_count": int(schedule["execution_chunk_count"]),
            "execution_chunk_sizes": [
                int(chunk["system_count"]) for chunk in schedule["planned_chunks"]
            ],
            "active_refill": bool(schedule["active_refill"]),
            "active_compaction": bool(schedule["active_compaction"]),
            "worker_backend": schedule["worker_backend"],
        },
        "counts": {
            "automatic_graph_evaluations": int(automatic["graph_evaluations"]),
            "automatic_model_calls": int(automatic["model_evaluations"]),
            "mps_model_evaluations": int(mps["model_evaluations"]),
            "automatic_optimizer_steps": int(automatic["optimizer_steps"]),
            "mps_optimizer_steps": int(mps["optimizer_steps"]),
        },
        "tail_recovery": {
            "enabled": bool(recovery.get("enabled", False)),
            "attempted_count": int(recovery.get("attempted_count", 0)),
            "converged_count": int(recovery.get("converged_count", 0)),
            "total_seconds": float(recovery.get("total_seconds", 0.0)),
            "model_evaluations": int(
                recovery.get("model_evaluations", 0)
            ),
            "optimizer_steps": int(recovery.get("optimizer_steps", 0)),
        },
        "gates": gates,
        "raw": {
            "automatic_path": str(automatic_path),
            "automatic_sha256": _sha256(automatic_path),
            "mps_path": str(mps_path),
            "mps_sha256": _sha256(mps_path),
        },
    }
    row["all_scientific_and_resource_gates_pass"] = all(
        gates[name]
        for name in (
            "identity_and_coverage",
            "same_requested_contract",
            "no_oom",
            "automatic_memory_at_most_85_percent",
            "automatic_convergence_not_below_mps",
        )
    )
    return row


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pool: dict[str, Any] = {}
    for pool_size in sorted({int(row["pool_size"]) for row in rows}):
        selected = [row for row in rows if row["pool_size"] == pool_size]
        speedups = [row["timing"]["external_speedup_over_mps"] for row in selected]
        by_pool[str(pool_size)] = {
            "workloads": len(selected),
            "external_speedup_over_mps": distribution(speedups),
            "automatic_wins": sum(speedup > 1.0 for speedup in speedups),
            "scientific_and_resource_gates_pass": sum(
                bool(row["all_scientific_and_resource_gates_pass"]) for row in selected
            ),
            "convergence_non_regression_pass": sum(
                bool(row["gates"]["automatic_convergence_not_below_mps"]) for row in selected
            ),
        }

    component_totals: dict[str, float] = defaultdict(float)
    phase_totals: dict[str, float] = defaultdict(float)
    total_external = 0.0
    total_chunk_work = 0.0
    for row in rows:
        total_external += float(row["timing"]["automatic_external_seconds"])
        for name, seconds in row["timing"]["automatic_components_seconds"].items():
            component_totals[name] += float(seconds)
        total_chunk_work += float(row["phases"]["chunk_wall_work_seconds"])
        for name, seconds in row["phases"]["exclusive_phase_seconds"].items():
            phase_totals[name] += float(seconds)

    weakest = sorted(
        rows,
        key=lambda row: row["timing"]["external_speedup_over_mps"],
    )[:5]
    memory_ratios = [row["memory"]["predicted_to_actual_chunk_peak_ratio"]["p50"] for row in rows]
    worker_imbalances = [row["timing"]["automatic_worker_load"]["max_over_mean"] for row in rows]
    total_jobs = sum(int(row["pool_size"]) for row in rows)
    automatic_converged = sum(int(row["convergence"]["automatic"]) for row in rows)
    mps_converged = sum(int(row["convergence"]["mps"]) for row in rows)
    automatic_only_converged = sum(
        int(row["endpoints"]["convergence_categories"]["automatic_only_converged"]) for row in rows
    )
    mps_only_converged = sum(
        int(row["endpoints"]["convergence_categories"]["mps_only_converged"]) for row in rows
    )
    recovery_attempted = sum(
        int(row["tail_recovery"]["attempted_count"]) for row in rows
    )
    recovery_converged = sum(
        int(row["tail_recovery"]["converged_count"]) for row in rows
    )
    return {
        "workloads": len(rows),
        "method_runs": 2 * len(rows),
        "external_speedup_over_mps": distribution(
            row["timing"]["external_speedup_over_mps"] for row in rows
        ),
        "automatic_external_wins": sum(
            row["timing"]["external_speedup_over_mps"] > 1.0 for row in rows
        ),
        "all_identity_and_coverage_pass": all(
            row["gates"]["identity_and_coverage"] for row in rows
        ),
        "all_contract_checks_pass": all(row["gates"]["same_requested_contract"] for row in rows),
        "all_no_oom_pass": all(row["gates"]["no_oom"] for row in rows),
        "all_memory_gates_pass": all(
            row["gates"]["automatic_memory_at_most_85_percent"] for row in rows
        ),
        "all_convergence_non_regression_pass": all(
            row["gates"]["automatic_convergence_not_below_mps"] for row in rows
        ),
        "convergence": {
            "jobs": total_jobs,
            "automatic_converged": automatic_converged,
            "automatic_rate": automatic_converged / total_jobs,
            "mps_converged": mps_converged,
            "mps_rate": mps_converged / total_jobs,
            "difference": automatic_converged - mps_converged,
            "automatic_only_converged": automatic_only_converged,
            "mps_only_converged": mps_only_converged,
            "automatic_nonconverged_tail": total_jobs - automatic_converged,
            "deterministic_ase_tail_union_converged": (automatic_converged + mps_only_converged),
        },
        "tail_recovery": {
            "attempted": recovery_attempted,
            "converged": recovery_converged,
            "success_fraction": (
                recovery_converged / recovery_attempted
                if recovery_attempted
                else None
            ),
            "total_seconds": sum(
                float(row["tail_recovery"]["total_seconds"])
                for row in rows
            ),
            "model_evaluations": sum(
                int(row["tail_recovery"]["model_evaluations"])
                for row in rows
            ),
            "optimizer_steps": sum(
                int(row["tail_recovery"]["optimizer_steps"])
                for row in rows
            ),
        },
        "scientific_and_resource_gate_failures": [
            row["workload_id"] for row in rows if not row["all_scientific_and_resource_gates_pass"]
        ],
        "by_pool": by_pool,
        "automatic_full_process_components": {
            "total_seconds": dict(component_totals),
            "fraction": {
                name: seconds / total_external for name, seconds in component_totals.items()
            },
            "accounted_fraction": (sum(component_totals.values()) / total_external),
        },
        "automatic_worker_phase_work": {
            "total_seconds": dict(phase_totals),
            "fraction": {
                name: seconds / total_chunk_work for name, seconds in phase_totals.items()
            },
        },
        "predicted_to_actual_chunk_memory_ratio_p50_across_workloads": (
            distribution(memory_ratios)
        ),
        "worker_max_over_mean_imbalance": distribution(worker_imbalances),
        "weakest_external_speedups": [
            {
                "workload_id": row["workload_id"],
                "speedup": row["timing"]["external_speedup_over_mps"],
            }
            for row in weakest
        ],
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    timing = row["timing"]
    components = timing["automatic_components_seconds"]
    phases = row["phases"]["exclusive_phase_fraction"]
    endpoint_metrics = row["endpoints"]["metrics"]
    return {
        "workload_id": row["workload_id"],
        "pool_size": row["pool_size"],
        "scope": row["scope"],
        "gpu_count": row["gpu_count"],
        "auto_external_s": timing["automatic_external_seconds"],
        "mps_external_s": timing["mps_external_seconds"],
        "external_speedup_over_mps": timing["external_speedup_over_mps"],
        "auto_production_s": timing["automatic_production_seconds"],
        "mps_production_s": timing["mps_production_seconds"],
        "production_speedup_over_mps": timing["production_speedup_over_mps"],
        "auto_setup_output_s": components["structure_model_setup_and_output"],
        "auto_worker_startup_s": components["worker_startup"],
        "auto_worker_execution_s": components["worker_execution"],
        "auto_worker_max_over_mean": timing["automatic_worker_load"]["max_over_mean"],
        "mps_gpu_max_over_mean": timing["mps_gpu_load"]["max_over_mean"],
        "neighbor_phase_fraction": phases["neighbor_update"],
        "model_forward_phase_fraction": phases["model_forward"],
        "model_autograd_phase_fraction": phases["model_autograd"],
        "bfgs_phase_fraction": phases["bfgs_update"],
        "compaction_phase_fraction": phases["active_compaction"],
        "auto_peak_memory_fraction": row["memory"]["max_conservative_peak_fraction"],
        "predicted_actual_memory_ratio_p50": row["memory"]["predicted_to_actual_chunk_peak_ratio"][
            "p50"
        ],
        "execution_chunks": row["schedule"]["execution_chunk_count"],
        "resident_waves": row["schedule"]["resident_wave_count"],
        "auto_converged": row["convergence"]["automatic"],
        "mps_converged": row["convergence"]["mps"],
        "convergence_difference": row["convergence"]["difference"],
        "tail_recovery_attempted": row["tail_recovery"]["attempted_count"],
        "tail_recovery_converged": row["tail_recovery"]["converged_count"],
        "tail_recovery_seconds": row["tail_recovery"]["total_seconds"],
        "convergence_state_mismatches": row["endpoints"]["convergence_state_mismatch_count"],
        "energy_abs_difference_p95_eV": endpoint_metrics["absolute_energy_eV"]["p95"],
        "position_component_difference_p95_A": endpoint_metrics["max_position_component_A"]["p95"],
        "cell_component_difference_p95_A": endpoint_metrics["max_cell_component_A"]["p95"],
        "all_scientific_and_resource_gates_pass": row["all_scientific_and_resource_gates_pass"],
    }


def _render_markdown(
    result: dict[str, Any],
    decision: dict[str, Any] | None,
) -> str:
    aggregate = result["aggregate"]
    lines = [
        f"# OMC-CSP Scheduler {result['split'].title()} Analysis",
        "",
        "## Scope",
        "",
        (
            f"This report audits {aggregate['workloads']} frozen "
            f"{result['split']} "
            "workloads and compares the current automatic tensor scheduler "
            "with four ASE BFGS CUDA-MPS workers per GPU. External full-process "
            "makespan is the primary performance metric."
        ),
        "",
        "## Primary Results",
        "",
        "| Pool | Workloads | Median speedup | Range | Auto wins | Convergence gate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for pool, summary in aggregate["by_pool"].items():
        speedup = summary["external_speedup_over_mps"]
        lines.append(
            f"| {pool} | {summary['workloads']} | {speedup['p50']:.3f}x | "
            f"{speedup['min']:.3f}-{speedup['max']:.3f}x | "
            f"{summary['automatic_wins']}/{summary['workloads']} | "
            f"{summary['convergence_non_regression_pass']}/"
            f"{summary['workloads']} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Exact identity and coverage: **{aggregate['all_identity_and_coverage_pass']}**",
            f"- Requested numerical contract: **{aggregate['all_contract_checks_pass']}**",
            f"- No OOM: **{aggregate['all_no_oom_pass']}**",
            f"- Conservative 85% memory limit: **{aggregate['all_memory_gates_pass']}**",
            (
                "- Convergence-count non-regression: "
                f"**{aggregate['all_convergence_non_regression_pass']}**"
            ),
            (
                "- Aggregate convergence: automatic "
                f"**{aggregate['convergence']['automatic_converged']}/"
                f"{aggregate['convergence']['jobs']}**, MPS "
                f"**{aggregate['convergence']['mps_converged']}/"
                f"{aggregate['convergence']['jobs']}**"
            ),
            (
                "- Automatic nonconverged tail: "
                f"**{aggregate['convergence']['automatic_nonconverged_tail']} "
                "jobs**"
            ),
            (
                "- ASE tail recovery: "
                f"**{aggregate['tail_recovery']['converged']}/"
                f"{aggregate['tail_recovery']['attempted']} recovered**, "
                f"cost **{aggregate['tail_recovery']['total_seconds']:.2f} s**"
            ),
            (
                "- Scientific/resource gate failures: "
                + (
                    ", ".join(aggregate["scientific_and_resource_gate_failures"])
                    if aggregate["scientific_and_resource_gate_failures"]
                    else "none"
                )
            ),
            "",
            "Endpoint differences are reported diagnostically. No endpoint-difference "
            "pass threshold is applied because the frozen epoch contract did not "
            "declare one.",
            "",
            "## Automatic Full-Process Time",
            "",
            "| Component | Fraction of aggregate external time |",
            "|---|---:|",
        ]
    )
    for name, fraction in sorted(
        aggregate["automatic_full_process_components"]["fraction"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| {name} | {100.0 * fraction:.1f}% |")
    lines.extend(
        [
            "",
            "## Automatic Worker Work",
            "",
            "| Phase | Fraction of summed chunk work |",
            "|---|---:|",
        ]
    )
    for name, fraction in sorted(
        aggregate["automatic_worker_phase_work"]["fraction"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        lines.append(f"| {name} | {100.0 * fraction:.1f}% |")
    lines.extend(
        [
            "",
            "## Weakest External Speedups",
            "",
            "| Workload | Speedup over MPS |",
            "|---|---:|",
        ]
    )
    for item in aggregate["weakest_external_speedups"]:
        lines.append(f"| {item['workload_id']} | {item['speedup']:.3f}x |")
    lines.extend(["", "## Refinement Decision", ""])
    if decision is None:
        lines.append("Pending expert selection after reviewing the machine-readable analysis.")
    else:
        lines.extend(
            [
                f"**{decision['title']}**",
                "",
                decision["hypothesis"],
                "",
                f"Mechanism: {decision['mechanism']}",
                "",
                f"Validation: {decision['validation']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Workload Table",
            "",
            "| Workload | Auto (s) | MPS (s) | Speedup | Conv. auto/MPS | Peak memory |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rows"]:
        timing = row["timing"]
        lines.append(
            f"| {row['workload_id']} | "
            f"{timing['automatic_external_seconds']:.2f} | "
            f"{timing['mps_external_seconds']:.2f} | "
            f"{timing['external_speedup_over_mps']:.3f}x | "
            f"{row['convergence']['automatic']}/"
            f"{row['convergence']['mps']} | "
            f"{100.0 * row['memory']['max_conservative_peak_fraction']:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def analyze(
    completion_path: Path,
    *,
    decision: dict[str, Any] | None = None,
    split: str = "development",
    expected_workloads: int = EXPECTED_WORKLOADS,
) -> dict[str, Any]:
    completion = _load(completion_path)
    if completion["status"] != "complete":
        raise ValueError(f"{split} matrix is not complete")
    if completion.get("split") != split:
        raise ValueError(
            f"expected completion split {split!r}, "
            f"found {completion.get('split')!r}"
        )
    if int(completion["workload_count"]) != expected_workloads:
        raise ValueError(
            f"expected {expected_workloads} workloads, "
            f"found {completion['workload_count']}"
        )
    rows = [analyze_workload(run) for run in completion["runs"]]
    rows.sort(key=lambda row: (row["pool_size"], row["workload_id"]))
    return {
        "schema_version": 1,
        "status": "complete",
        "analysis": f"omc_csp_scheduler_epoch1_{split}",
        "split": split,
        "primary_metric": "external_full_process_makespan_seconds",
        "memory_limit_fraction": MEMORY_LIMIT_FRACTION,
        "source": {
            "completion_path": str(completion_path),
            "completion_sha256": _sha256(completion_path),
        },
        "aggregate": _aggregate_rows(rows),
        "refinement_decision": decision,
        "rows": rows,
    }


def write_outputs(
    result: dict[str, Any],
    output_dir: Path,
    *,
    decision: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = str(result["split"])
    json_path = output_dir / f"{split}_analysis.json"
    csv_path = output_dir / f"{split}_analysis.csv"
    markdown_path = output_dir / f"{split}_analysis.md"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_rows = [_csv_row(row) for row in result["rows"]]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    markdown_path.write_text(
        _render_markdown(result, decision),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_WORKLOADS_BY_SPLIT),
        default="development",
    )
    args = parser.parse_args()
    decision = _load(args.decision) if args.decision else None
    result = analyze(
        args.completion,
        decision=decision,
        split=args.split,
        expected_workloads=EXPECTED_WORKLOADS_BY_SPLIT[args.split],
    )
    write_outputs(result, args.output_dir, decision=decision)
    print(
        json.dumps(
            {
                "status": result["status"],
                "workloads": result["aggregate"]["workloads"],
                "all_scientific_and_resource_gates_pass": not result["aggregate"][
                    "scientific_and_resource_gate_failures"
                ],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
