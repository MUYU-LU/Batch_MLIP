import pytest

from benchmarks.summarize_heldout_auto_vs_mps import distribution


def test_distribution_reports_convergence_horizon_tail():
    summary = distribution([1, 2, 3, 4, 100])

    assert summary == {
        "min": 1,
        "mean": 22.0,
        "p50": 3,
        "p95": 100,
        "max": 100,
    }


def test_distribution_rejects_empty_values():
    with pytest.raises(ValueError, match="at least one"):
        distribution([])
