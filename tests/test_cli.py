from __future__ import annotations

from pathlib import Path

import yaml
from ase import Atoms
from ase.io import write
from atombit_batch.cli import run_config, validate_batch


def test_yaml_cli_run_and_validation(tmp_path: Path):
    input_file = tmp_path / "input.extxyz"
    output_file = tmp_path / "relaxed.extxyz"
    write(
        input_file,
        [Atoms("H", positions=[[1.0, 0.0, 0.0]]), Atoms("H", positions=[[-0.5, 0.2, 0.0]])],
    )
    config = {
        "schema_version": 1,
        "task": "relax",
        "input": str(input_file),
        "output": str(output_file),
        "validation_output": str(tmp_path / "validation.json"),
        "runtime": {"device": "cpu", "dtype": "float64", "skin": 0.2},
        "model": {
            "factory": "atombit_batch.toy_models:build_quadratic_model",
            "kwargs": {"k": 1.0},
            "cutoff": 2.0,
            "force_mode": "autograd",
        },
        "relax": {
            "fmax": 1e-4,
            "max_steps": 500,
            "active_compaction": True,
        },
        "reporting": {
            "summary": str(tmp_path / "summary.json"),
            "trajectory": str(tmp_path / "trajectory.extxyz"),
            "diagnostics": str(tmp_path / "diagnostics.jsonl"),
        },
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_config(config_path)
    assert output_file.exists()
    assert (tmp_path / "trajectory.extxyz").exists()
    assert (tmp_path / "diagnostics.jsonl").exists()
    assert all(summary["converged"])
    assert summary["model_evaluations"] == len(summary["active_batch_sizes"])
    assert summary["graph_evaluations"] == sum(summary["active_batch_sizes"])

    validation = validate_batch(config_path)
    assert validation["passed"]


def test_yaml_cli_accepts_optional_frechet_cell_filter(tmp_path: Path):
    input_file = tmp_path / "periodic.extxyz"
    output_file = tmp_path / "cell_relaxed.extxyz"
    write(
        input_file,
        Atoms(
            "H2",
            positions=[[0.5, 0.5, 0.5], [1.7, 1.6, 1.5]],
            cell=[4.0, 4.0, 4.0],
            pbc=True,
        ),
    )
    config = {
        "task": "relax",
        "input": str(input_file),
        "output": str(output_file),
        "runtime": {"device": "cpu", "dtype": "float64", "skin": 0.0},
        "model": {
            "factory": "atombit_batch.toy_models:build_pair_harmonic_model",
            "kwargs": {"k": 2.0, "r0": 1.4, "cutoff": 3.0},
            "cutoff": 3.0,
            "force_mode": "autograd",
        },
        "relax": {
            "optimizer": "fire",
            "active_compaction": False,
            "cell_filter": {
                "type": "frechet",
                "hydrostatic_strain": True,
            },
            "fmax": 10.0,
            "smax": 10.0,
            "max_steps": 0,
        },
    }
    config_path = tmp_path / "cell_run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = run_config(config_path)

    assert output_file.exists()
    assert summary["converged"] == [True]
    assert summary["max_stress"] is not None
