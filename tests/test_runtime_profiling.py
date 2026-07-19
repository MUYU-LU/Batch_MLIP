from __future__ import annotations

import torch
from ase import Atoms

from batch_mlip import AtomBitBatchCalculator, RuntimeProfiler, relax
from batch_mlip.models.toy_models import QuadraticWellModel
from batch_mlip.profiling import profile_event, profile_phase


def _calculator() -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0),
        cutoff=2.5,
        device="cpu",
        dtype=torch.float64,
    )


def test_runtime_profiler_records_cpu_phases_and_events():
    with RuntimeProfiler(device="cpu") as profiler:
        with profile_phase("test.phase", systems=2):
            torch.arange(8).sum()
        profile_event("test.event", count=3, enabled=True)

    summary = profiler.summary()
    assert summary["schema_version"] == 1
    assert summary["total_seconds"] >= 0.0
    assert summary["phases"]["test.phase"]["count"] == 1
    assert summary["phases"]["test.phase"]["total_seconds"] >= 0.0
    assert summary["events"] == [
        {"name": "test.event", "count": 3, "enabled": True}
    ]
    assert summary["samples"][0]["systems"] == 2


def test_profiled_refill_bfgs_preserves_results_and_reports_runtime_phases():
    systems = [
        Atoms("H", positions=[[value, 0.0, 0.0]])
        for value in (0.3, -0.2, 0.1)
    ]
    options = {
        "optimizer": "bfgs",
        "fmax": 1e-30,
        "max_steps": 1,
        "refill_batch_size": 2,
    }

    reference = relax(systems, _calculator(), **options)
    with RuntimeProfiler(device="cpu") as profiler:
        measured = relax(systems, _calculator(), **options)

    torch.testing.assert_close(measured.state.positions, reference.state.positions)
    torch.testing.assert_close(measured.evaluation.energy, reference.evaluation.energy)
    torch.testing.assert_close(measured.evaluation.forces, reference.evaluation.forces)

    summary = profiler.summary(include_samples=False)
    expected_phases = {
        "calculator.graph_view",
        "calculator.neighbor_update",
        "graph.geometry_to_host",
        "graph.neighbor_search",
        "graph.to_device",
        "model.autograd",
        "model.forward",
        "optimizer.bfgs_update",
        "scheduler.refill_repack",
    }
    assert expected_phases <= summary["phases"].keys()
    refill_events = [
        event for event in summary["events"] if event["name"] == "refill"
    ]
    assert refill_events
    assert sum(event["inserted"] for event in refill_events) == 1
    assert {event["policy"] for event in refill_events} == {"immediate"}

    evaluations = [
        event
        for event in summary["events"]
        if event["name"] == "optimizer_evaluation"
    ]
    assert evaluations[0]["active_systems"] == 2
    assert evaluations[0]["pending_systems"] == 1
    assert evaluations[-1]["pending_systems"] == 0


def test_profiled_active_compaction_reports_removed_systems():
    systems = [
        Atoms("H", positions=[[1e-8, 0.0, 0.0]]),
        Atoms("H", positions=[[0.3, 0.0, 0.0]]),
    ]
    with RuntimeProfiler(device="cpu") as profiler:
        relax(
            systems,
            _calculator(),
            optimizer="bfgs",
            fmax=1e-5,
            max_steps=2,
            active_compaction=True,
        )

    summary = profiler.summary(include_samples=False)
    assert "scheduler.active_compaction" in summary["phases"]
    events = [
        event
        for event in summary["events"]
        if event["name"] == "active_compaction"
    ]
    assert events[0]["systems_before"] == 2
    assert events[0]["systems_after"] == 1
    assert events[0]["removed"] == 1
