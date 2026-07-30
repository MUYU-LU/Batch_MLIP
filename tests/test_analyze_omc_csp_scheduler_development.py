from __future__ import annotations

import pytest

from benchmarks.analyze_omc_csp_scheduler_development import (
    _array_max_abs_difference,
    _auto_memory,
    distribution,
    endpoint_comparison,
)


def _record(source: str, *, energy: float, converged: bool = True):
    return {
        "source": source,
        "converged": converged,
        "energy_eV": energy,
        "max_force_eV_per_A": 0.04,
        "max_abs_stress_eV_per_A3": 0.001,
        "positions_A": [[energy, 0.0, 0.0]],
        "cell_A": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "stress_eV_per_A3": [[0.0, 0.0, 0.0]] * 3,
        "steps": 10,
    }


def test_distribution_uses_stable_nearest_rank_percentiles():
    assert distribution([1, 2, 3, 4, 100]) == {
        "min": 1.0,
        "mean": 22.0,
        "p50": 3.0,
        "p95": 100.0,
        "max": 100.0,
    }


def test_endpoint_comparison_joins_by_source_not_record_order():
    automatic = [_record("a", energy=1.0), _record("b", energy=2.0)]
    mps = [_record("b", energy=2.25), _record("a", energy=1.5)]

    summary = endpoint_comparison(automatic, mps)

    assert summary["same_source_set"]
    assert summary["common_source_count"] == 2
    assert summary["metrics"]["absolute_energy_eV"]["max"] == 0.5
    assert summary["convergence_categories"] == {
        "both_converged": 2,
        "automatic_only_converged": 0,
        "mps_only_converged": 0,
        "neither_converged": 0,
    }


def test_endpoint_comparison_rejects_duplicate_sources():
    duplicate = [_record("a", energy=1.0), _record("a", energy=2.0)]

    with pytest.raises(ValueError, match="duplicate endpoint source"):
        endpoint_comparison(duplicate, [_record("a", energy=1.0)])


def test_array_difference_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        _array_max_abs_difference([1.0], [1.0, 2.0])


def test_auto_memory_adds_parent_probe_to_worker_peak():
    automatic = {
        "environment": {"gpu_total_memory_bytes": 1000},
        "peak_memory": {"cuda:0": {"allocated_bytes": 40, "reserved_bytes": 100}},
        "scheduling": {
            "probe": {
                "device": "cuda:0",
                "peak_reserved_bytes": 90,
            },
            "workers": [
                {
                    "device": "cuda:0",
                    "chunks": [
                        {
                            "peak_reserved_bytes": 300,
                            "predicted_peak_bytes": 600,
                        },
                        {
                            "peak_reserved_bytes": 250,
                            "predicted_peak_bytes": 500,
                        },
                    ],
                },
                {
                    "device": "cuda:1",
                    "chunks": [
                        {
                            "peak_reserved_bytes": 200,
                            "predicted_peak_bytes": 600,
                        }
                    ],
                },
            ],
        },
        "workers": [
            {
                "device": "cuda:0",
                "chunks": [
                    {
                        "peak_reserved_bytes": 300,
                        "predicted_peak_bytes": 600,
                    },
                    {
                        "peak_reserved_bytes": 250,
                        "predicted_peak_bytes": 500,
                    },
                ],
            },
            {
                "device": "cuda:1",
                "chunks": [
                    {
                        "peak_reserved_bytes": 200,
                        "predicted_peak_bytes": 600,
                    }
                ],
            },
        ],
    }

    memory = _auto_memory(automatic)

    assert memory["max_worker_peak_reserved_bytes"] == 300
    assert memory["max_conservative_peak_reserved_bytes"] == 400
    assert memory["max_conservative_peak_fraction"] == 0.4
    assert memory["predicted_to_actual_chunk_peak_ratio"]["p50"] == 2.0


def test_auto_memory_takes_maximum_of_sequential_recovery_stage():
    automatic = {
        "environment": {"gpu_total_memory_bytes": 1000},
        "peak_memory": {
            "cuda:0": {"allocated_bytes": 40, "reserved_bytes": 100}
        },
        "scheduling": {
            "probe": {
                "device": "cuda:0",
                "peak_reserved_bytes": 90,
            },
            "workers": [
                {
                    "device": "cuda:0",
                    "chunks": [
                        {
                            "peak_reserved_bytes": 300,
                            "predicted_peak_bytes": 600,
                        }
                    ],
                }
            ],
        },
        "tail_recovery": {
            "peak_reserved_bytes_by_device": {"cuda:0": 500},
            "parent_reserved_bytes_during_recovery_by_device": {
                "cuda:0": 50
            },
        },
    }

    memory = _auto_memory(automatic)

    assert memory["tensor_conservative_peak_reserved_bytes_by_device"] == {
        "cuda:0": 400
    }
    assert memory["recovery_conservative_peak_reserved_bytes_by_device"] == {
        "cuda:0": 550
    }
    assert memory["max_conservative_peak_reserved_bytes"] == 550
