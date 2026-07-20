"""Registry-compatible run telemetry for controlled experiments."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_TELEMETRY_FIELDS = (
    "run_id",
    "study_id",
    "workload_id",
    "workload_manifest_sha256",
    "model_name",
    "model_checkpoint_sha256",
    "code_commit",
    "dirty_tree_hash",
    "algorithm",
    "cell_mode",
    "force_mode",
    "model_dtype",
    "optimizer_dtype",
    "skin_A",
    "cache_policy",
    "refill_policy",
    "refill_check_interval",
    "refill_threshold",
    "batch_policy",
    "resident_graph_limit",
    "atom_limit",
    "edge_limit",
    "memory_safety_fraction",
    "micro_pool_size",
    "gpu_count",
    "gpu_model",
    "gpu_memory_GB",
    "worker_mode",
    "cold_or_warm",
    "compile_mode",
    "seed",
    "repeat_index",
    "start_timestamp",
    "end_timestamp",
    "wall_time_s",
    "kernel_time_s",
    "neighbor_time_s",
    "filter_time_s",
    "optimizer_or_integrator_time_s",
    "pack_time_s",
    "compaction_time_s",
    "admission_time_s",
    "transfer_time_s",
    "io_time_s",
    "startup_time_s",
    "peak_allocated_GB",
    "peak_reserved_GB",
    "predicted_peak_GB",
    "total_model_calls",
    "total_neighbor_rebuilds",
    "cache_hit_rate",
    "mean_rebuild_interval",
    "candidate_edges",
    "active_edges",
    "inactive_work_fraction",
    "refill_events",
    "mean_insertion_size",
    "compile_shape_variants",
    "converged_jobs",
    "failed_jobs",
    "accepted_jobs",
    "time_to_first_result_s",
    "energy_error_max",
    "force_error_max",
    "stress_error_max",
    "nve_drift_eV_per_atom_ps",
    "temperature_error_K",
    "equivalence_tier",
    "validation_pass",
    "notes",
)

_REQUIRED_FIELDS = {
    "algorithm",
    "cell_mode",
    "code_commit",
    "cold_or_warm",
    "equivalence_tier",
    "gpu_count",
    "model_name",
    "repeat_index",
    "run_id",
    "study_id",
    "validation_pass",
    "wall_time_s",
    "worker_mode",
    "workload_id",
    "workload_manifest_sha256",
}


def _validate_scalar(name: str, value: Any) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise TypeError(f"telemetry field {name!r} must be a finite JSON scalar or null")


@dataclass(frozen=True)
class RunTelemetry:
    """One complete row in the controlled-experiment registry."""

    values: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported run telemetry schema")
        unknown = set(self.values) - set(RUN_TELEMETRY_FIELDS)
        if unknown:
            raise KeyError(f"unknown telemetry fields: {sorted(unknown)}")
        missing = [name for name in _REQUIRED_FIELDS if self.values.get(name) is None]
        if missing:
            raise ValueError(f"missing required telemetry fields: {sorted(missing)}")
        for name, value in self.values.items():
            _validate_scalar(name, value)
        digest = str(self.values["workload_manifest_sha256"])
        if len(digest) != 64:
            raise ValueError("workload_manifest_sha256 must be a SHA-256 digest")
        if int(self.values["gpu_count"]) <= 0 or int(self.values["repeat_index"]) < 0:
            raise ValueError("gpu_count and repeat_index are invalid")
        if float(self.values["wall_time_s"]) < 0.0:
            raise ValueError("wall_time_s must be non-negative")
        if self.values["equivalence_tier"] not in {"K0", "K1", "K2", "K3", "K4"}:
            raise ValueError("equivalence_tier must be K0 through K4")

    @classmethod
    def create(cls, **values: Any) -> RunTelemetry:
        complete = {field: values.get(field) for field in RUN_TELEMETRY_FIELDS}
        unknown = set(values) - set(RUN_TELEMETRY_FIELDS)
        if unknown:
            raise KeyError(f"unknown telemetry fields: {sorted(unknown)}")
        return cls(complete)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **{field: self.values.get(field) for field in RUN_TELEMETRY_FIELDS},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunTelemetry:
        values = dict(payload)
        version = int(values.pop("schema_version", 1))
        return cls(values=values, schema_version=version)


def runtime_profile_registry_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Project existing RuntimeProfiler phases into registry timing columns."""

    phases = profile.get("phases", {})

    def total(*names: str) -> float:
        return float(sum(phases.get(name, {}).get("total_seconds", 0.0) for name in names))

    events = profile.get("events", [])
    model_events = [event for event in events if event.get("name") == "model_evaluation"]
    refills = [
        event for event in events if event.get("name") == "refill" and event.get("triggered", True)
    ]
    insertions = [float(event.get("inserted", 0)) for event in refills]
    rebuilds = [int(event["neighbor_rebuilds"]) for event in model_events]
    active_edges = [int(event["edges"]) for event in model_events if "edges" in event]
    candidate_edges = [
        int(event["candidate_edges"]) for event in model_events if "candidate_edges" in event
    ]
    return {
        "wall_time_s": float(profile.get("total_seconds", 0.0)),
        "kernel_time_s": total("model.forward", "model.autograd"),
        "neighbor_time_s": total("graph.neighbor_search"),
        "filter_time_s": total("calculator.graph_view", "mace.tensor_projection"),
        "optimizer_or_integrator_time_s": total(
            "optimizer.bfgs_update", "optimizer.fire_update", "integrator.update"
        ),
        "pack_time_s": total("scheduler.refill_repack"),
        "compaction_time_s": total("scheduler.active_compaction"),
        "transfer_time_s": total("graph.to_device", "graph.geometry_to_host"),
        "peak_allocated_GB": (
            None
            if profile.get("peak_memory_bytes") is None
            else float(profile["peak_memory_bytes"]) / 1e9
        ),
        "total_model_calls": len(model_events),
        "total_neighbor_rebuilds": sum(rebuilds),
        "cache_hit_rate": (
            None if not model_events else sum(value == 0 for value in rebuilds) / len(model_events)
        ),
        "mean_rebuild_interval": (
            None if not rebuilds or sum(rebuilds) == 0 else len(model_events) / sum(rebuilds)
        ),
        "candidate_edges": (
            sum(candidate_edges) if len(candidate_edges) == len(model_events) else None
        ),
        "active_edges": (sum(active_edges) if len(active_edges) == len(model_events) else None),
        "refill_events": len(refills),
        "mean_insertion_size": (None if not insertions else sum(insertions) / len(insertions)),
    }


def write_run_telemetry_json(path: str | Path, telemetry: RunTelemetry) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(telemetry.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_run_telemetry_csv(path: str | Path, telemetry: RunTelemetry) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_TELEMETRY_FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({field: telemetry.values.get(field) for field in RUN_TELEMETRY_FIELDS})
