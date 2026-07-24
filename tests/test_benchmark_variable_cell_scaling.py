from __future__ import annotations

import torch

from benchmarks.benchmark_variable_cell_scaling import timed_repeats
from benchmarks.summarize_smooth_optimizer_frontier import select_frontier


def test_timed_repeats_reports_allocated_and_reserved_memory() -> None:
    output, timing, peak_allocated, peak_reserved = timed_repeats(
        lambda: {"status": "ok"},
        repeats=2,
        device=torch.device("cpu"),
    )

    assert output == {"status": "ok"}
    assert len(timing["samples_seconds"]) == 2
    assert peak_allocated is None
    assert peak_reserved is None


def test_optimizer_frontier_rejects_memory_and_selects_smallest_near_peak() -> None:
    points = [
        {
            "batch_size": 32,
            "status": "passed",
            "systems_per_second": 9.9,
            "peak_reserved_memory_bytes": 40,
            "records": [{"converged": True}] * 4,
        },
        {
            "batch_size": 64,
            "status": "passed",
            "systems_per_second": 10.0,
            "peak_reserved_memory_bytes": 60,
            "records": [{"converged": True}] * 4,
        },
        {
            "batch_size": 128,
            "status": "passed",
            "systems_per_second": 12.0,
            "peak_reserved_memory_bytes": 95,
            "records": [{"converged": True}] * 4,
        },
    ]

    selected = select_frontier(
        points,
        pool_size=4,
        memory_limit_bytes=80,
    )

    assert selected is points[0]
