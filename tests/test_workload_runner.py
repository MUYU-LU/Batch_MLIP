from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read, write

from batch_mlip import AtomBitBatchCalculator
from batch_mlip.models.toy_models import QuadraticWellModel
from batch_mlip.workloads import (
    WorkloadJob,
    WorkloadManifest,
    WorkloadRunSpec,
    execute_workload,
    normalized_structure_sha256,
    write_workload_manifest,
)
from batch_mlip.workloads.cli import run_workload_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    tmp_path: Path,
    *,
    operation: str,
    jobs: int = 4,
) -> tuple[WorkloadManifest, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    source = dataset / "source.extxyz"
    write(
        source,
        Atoms("H2", positions=[[0.2, 0.1, 0.0], [-0.2, -0.1, 0.0]]),
    )
    atoms = read(source)
    normalized = normalized_structure_sha256(atoms)
    workload_id = "TEST-EVAL-v1" if operation == "force_evaluation" else "TEST-MD-v1"
    entries = tuple(
        WorkloadJob(
            system_id=f"{workload_id}:{index:04d}",
            group_id="source",
            duplicate_group=normalized,
            order=index,
            dataset_id="test",
            source_path=source.name,
            source_sha256=_sha256(source),
            normalized_structure_sha256=normalized,
            frame_index=0,
            atom_count=len(atoms),
            species=tuple(atoms.get_chemical_symbols()),
            chemical_formula=atoms.get_chemical_formula(mode="hill"),
            pbc=tuple(bool(value) for value in atoms.pbc),
            cell_A=tuple(float(value) for value in atoms.cell.array.reshape(-1)),
            volume_A3=0.0,
            constraints=(),
            topology_edge_counts={},
            random_seed=1000 + index if operation == "md_nve" else None,
        )
        for index in range(jobs)
    )
    metadata = (
        {"compute_stress": False}
        if operation == "force_evaluation"
        else {
            "ensemble": "nve",
            "initial_temperature_K": 300.0,
            "remove_initial_com": True,
            "force_exact_initial_temperature": True,
            "timestep_fs": 0.05,
            "warmup_steps": 1,
            "measured_steps": 2,
        }
    )
    manifest = WorkloadManifest(
        workload_id=workload_id,
        version=1,
        family="test",
        operation=operation,
        cell_mode="fixed",
        arrival_mode="closed",
        jobs=entries,
        metadata=metadata,
    ).seal()
    return manifest, dataset


def _calculator() -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        QuadraticWellModel(),
        cutoff=2.0,
        device="cpu",
        dtype=torch.float64,
    )


def _spec(batch_size: int, *, run_id: str) -> WorkloadRunSpec:
    return WorkloadRunSpec(
        run_id=run_id,
        study_id="unit-test",
        model_name="quadratic",
        model_checkpoint_sha256="a" * 64,
        code_commit="test",
        resident_batch_size=batch_size,
        equivalence_tier="K1",
        validation_pass=True,
    )


def test_static_workload_preserves_results_and_order_across_microbatches(tmp_path):
    manifest, dataset = _manifest(tmp_path, operation="force_evaluation")
    full = execute_workload(manifest, dataset, _calculator(), _spec(4, run_id="full"))
    chunked = execute_workload(
        manifest,
        dataset,
        _calculator(),
        _spec(2, run_id="chunked"),
        output_dir=tmp_path / "output",
        registry_path=tmp_path / "registry.csv",
    )

    assert chunked.summary["microbatch_count"] == 2
    assert chunked.summary["end_to_end_time_s"] >= chunked.summary["wall_time_s"]
    assert "calculator construction" in chunked.summary["timing_scope"]["end_to_end_time_s"]
    assert chunked.summary["output_system_ids"] == [job.system_id for job in manifest.jobs]
    assert chunked.telemetry.values["accepted_jobs"] == 4
    assert chunked.telemetry.values["total_model_calls"] == 2
    for expected, actual in zip(full.structures, chunked.structures, strict=True):
        assert expected.get_potential_energy() == actual.get_potential_energy()
        np.testing.assert_array_equal(expected.get_forces(), actual.get_forces())
    for name in ("telemetry.json", "runtime_profile.json", "summary.json", "final.extxyz"):
        assert (tmp_path / "output" / name).is_file()
    assert (tmp_path / "registry.csv").is_file()


def test_nve_workload_is_invariant_to_resident_batch_partitioning(tmp_path):
    manifest, dataset = _manifest(tmp_path, operation="md_nve")
    full = execute_workload(manifest, dataset, _calculator(), _spec(4, run_id="full"))
    chunked = execute_workload(manifest, dataset, _calculator(), _spec(2, run_id="chunked"))

    assert full.summary["useful_unit"] == "replica_steps"
    assert full.summary["useful_units"] == 8
    assert len(full.summary["md_energy"]["total_energy_drift_eV"]) == 4
    assert full.summary["md_energy"]["max_abs_energy_drift_eV_per_atom"] >= 0.0
    for expected, actual in zip(full.structures, chunked.structures, strict=True):
        np.testing.assert_allclose(expected.positions, actual.positions, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            expected.get_velocities(), actual.get_velocities(), rtol=0, atol=1e-15
        )


def test_yaml_runner_builds_a_generic_calculator(tmp_path):
    manifest, dataset = _manifest(tmp_path, operation="force_evaluation", jobs=2)
    manifest_path = tmp_path / "manifest.json"
    write_workload_manifest(manifest_path, manifest)
    config = {
        "schema_version": 1,
        "workload": {
            "manifest": str(manifest_path),
            "dataset_dir": str(dataset),
        },
        "calculator": {
            "factory": "batch_mlip.models.potential:load_atombit_batch",
            "kwargs": {
                "model_factory": "batch_mlip.models.toy_models:build_quadratic_model",
                "cutoff": 2.0,
                "dtype": "float64",
            },
        },
        "run": {
            "run_id": "yaml-run",
            "study_id": "unit-test",
            "model_name": "quadratic",
            "model_checkpoint_sha256": "b" * 64,
            "code_commit": "test",
            "resident_batch_size": 2,
            "equivalence_tier": "K0",
            "validation_pass": True,
        },
        "output": {"directory": str(tmp_path / "yaml-output")},
    }
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_workload_config(config_path)

    assert result.summary["jobs"] == 2
    assert json.loads((tmp_path / "yaml-output" / "summary.json").read_text())["jobs"] == 2


def test_run_spec_rejects_a_non_hexadecimal_checkpoint_digest():
    with np.testing.assert_raises_regex(ValueError, "SHA-256"):
        WorkloadRunSpec(
            run_id="invalid",
            study_id="unit-test",
            model_name="quadratic",
            model_checkpoint_sha256="z" * 64,
            code_commit="test",
            resident_batch_size=1,
            equivalence_tier="K0",
            validation_pass=False,
        )
