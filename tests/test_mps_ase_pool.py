from __future__ import annotations

import sys

import pytest

from benchmarks.benchmark_mps_ase_pool import (
    aggregate_throughput,
    consistent_worker_parameters,
    parse_args,
)


def test_nve_mps_throughput_counts_replica_steps():
    throughput, unit = aggregate_throughput(
        task="nve",
        pool_size=32,
        elapsed_seconds=16.0,
        measured_steps=1000,
    )

    assert throughput == 2000.0
    assert unit == "replica_steps_per_second"


def test_mps_worker_parameters_must_match():
    workers = [
        {"warmup_steps": 10, "measured_steps": 100, "timestep_fs": 0.5},
        {"warmup_steps": 10, "measured_steps": 99, "timestep_fs": 0.5},
    ]

    with pytest.raises(RuntimeError, match="inconsistent parameters"):
        consistent_worker_parameters(
            workers,
            ("warmup_steps", "measured_steps", "timestep_fs"),
        )


def test_evaluation_accepts_signed_manifest_without_optimizer(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "pipe"))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(tmp_path / "log"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mps_ase_pool.py",
            "--task",
            "evaluation",
            "--mlip",
            "mace",
            "--pool-size",
            "32",
            "--workers",
            "32",
            "--workload-manifest",
            "workload.json",
            "--output",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.task == "evaluation"
    assert args.optimizer is None


def test_optimization_still_requires_optimizer(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(tmp_path / "pipe"))
    monkeypatch.setenv("CUDA_MPS_LOG_DIRECTORY", str(tmp_path / "log"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_mps_ase_pool.py",
            "--mlip",
            "mace",
            "--atom-count",
            "46",
            "--output",
            "result.json",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()
