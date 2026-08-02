from __future__ import annotations

from argparse import Namespace

from benchmarks.benchmark_omc_csp_scheduler_auto import (
    _relaxation_options,
    _worker_peak_memory,
)


def test_worker_peak_memory_uses_execution_contexts_and_maximum_chunk():
    workers = [
        {
            "device": "cuda:0",
            "chunks": [
                {
                    "peak_allocated_bytes": 10,
                    "peak_reserved_bytes": 30,
                },
                {
                    "peak_allocated_bytes": 20,
                    "peak_reserved_bytes": 25,
                },
            ],
        },
        {
            "device": "cuda:1",
            "chunks": [
                {
                    "peak_allocated_bytes": None,
                    "peak_reserved_bytes": 40,
                }
            ],
        },
    ]

    assert _worker_peak_memory(workers) == {
        "cuda:0": {"allocated_bytes": 20, "reserved_bytes": 30},
        "cuda:1": {"allocated_bytes": 0, "reserved_bytes": 40},
    }


def test_worker_peak_memory_is_empty_without_worker_execution():
    assert _worker_peak_memory([]) == {}


def test_relaxation_contract_uses_ase_frechet_convergence_semantics():
    options = _relaxation_options(
        Namespace(
            fmax=0.05,
            max_steps=500,
            linear_algebra_backend="auto",
        )
    )

    assert options["smax"] is None
    assert options["fmax"] == 0.05
    assert options["optimizer_dtype"] == "float64"
