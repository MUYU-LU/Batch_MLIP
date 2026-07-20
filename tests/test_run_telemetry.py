from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from batch_mlip.profiling import (
    RUN_TELEMETRY_FIELDS,
    RunTelemetry,
    append_run_telemetry_csv,
    runtime_profile_registry_fields,
    write_run_telemetry_json,
)


def _telemetry(**overrides):
    values = {
        "run_id": "run-1",
        "study_id": "study-1",
        "workload_id": "TEST-v1",
        "workload_manifest_sha256": "a" * 64,
        "model_name": "toy",
        "code_commit": "deadbeef",
        "algorithm": "bfgs",
        "cell_mode": "fixed",
        "gpu_count": 1,
        "worker_mode": "single-process",
        "cold_or_warm": "warm",
        "repeat_index": 0,
        "wall_time_s": 1.25,
        "equivalence_tier": "K2",
        "validation_pass": True,
    }
    values.update(overrides)
    return RunTelemetry.create(**values)


def test_run_telemetry_round_trip_and_registry_header(tmp_path):
    telemetry = _telemetry()
    json_path = tmp_path / "run.json"
    csv_path = tmp_path / "registry.csv"

    write_run_telemetry_json(json_path, telemetry)
    append_run_telemetry_csv(csv_path, telemetry)

    assert RunTelemetry.from_dict(json.loads(json_path.read_text())) == telemetry
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == RUN_TELEMETRY_FIELDS
        assert len(list(reader)) == 1


def test_run_telemetry_fields_match_imported_protocol_registry():
    registry = (
        Path(__file__).parents[1] / "research" / "task-aware" / "experiment_registry_template.csv"
    )
    with registry.open(newline="", encoding="utf-8") as handle:
        expected = tuple(next(csv.reader(handle)))

    assert RUN_TELEMETRY_FIELDS == expected


def test_run_telemetry_validates_required_and_closed_fields():
    with pytest.raises(ValueError, match="missing required"):
        RunTelemetry.create(run_id="run-1")
    with pytest.raises(KeyError, match="unknown telemetry"):
        _telemetry(not_a_registry_field=1)
    with pytest.raises(ValueError, match="K0 through K4"):
        _telemetry(equivalence_tier="same-ish")


def test_runtime_profile_projection_uses_existing_phase_names():
    fields = runtime_profile_registry_fields(
        {
            "total_seconds": 4.0,
            "peak_memory_bytes": 2_000_000_000,
            "phases": {
                "model.forward": {"total_seconds": 1.2},
                "model.autograd": {"total_seconds": 0.3},
                "graph.neighbor_search": {"total_seconds": 0.5},
                "scheduler.refill_repack": {"total_seconds": 0.2},
            },
            "events": [
                {
                    "name": "model_evaluation",
                    "edges": 12,
                    "candidate_edges": 16,
                    "neighbor_rebuilds": 1,
                },
                {
                    "name": "model_evaluation",
                    "edges": 10,
                    "candidate_edges": 14,
                    "neighbor_rebuilds": 0,
                },
                {"name": "refill", "inserted": 4},
                {"name": "refill", "inserted": 2},
                {"name": "refill", "inserted": 0, "triggered": False},
            ],
        }
    )

    assert fields["wall_time_s"] == 4.0
    assert fields["kernel_time_s"] == pytest.approx(1.5)
    assert fields["neighbor_time_s"] == 0.5
    assert fields["pack_time_s"] == 0.2
    assert fields["peak_allocated_GB"] == 2.0
    assert fields["total_model_calls"] == 2
    assert fields["total_neighbor_rebuilds"] == 1
    assert fields["cache_hit_rate"] == 0.5
    assert fields["mean_rebuild_interval"] == 2.0
    assert fields["candidate_edges"] == 30
    assert fields["active_edges"] == 22
    assert fields["refill_events"] == 2
    assert fields["mean_insertion_size"] == 3.0
