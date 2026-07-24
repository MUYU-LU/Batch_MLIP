from __future__ import annotations

import torch

from benchmarks.benchmark_variable_cell_scaling import timed_repeats


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
