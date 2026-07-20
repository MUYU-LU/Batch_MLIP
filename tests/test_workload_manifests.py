from __future__ import annotations

import json

import pytest
from ase import Atoms

from batch_mlip.workloads import (
    TaskProfile,
    WorkloadJob,
    WorkloadManifest,
    normalized_structure_sha256,
    read_workload_manifest,
    topology_key,
    write_workload_jobs_csv,
    write_workload_manifest,
)


def _job(order: int, *, atoms: int = 2, duplicate: str = "a" * 64) -> WorkloadJob:
    return WorkloadJob(
        system_id=f"test:{order}",
        group_id=f"source-{order % 2}",
        duplicate_group=duplicate,
        order=order,
        dataset_id="test",
        source_path=f"source-{order % 2}.extxyz",
        source_sha256="b" * 64,
        normalized_structure_sha256=duplicate,
        frame_index=0,
        atom_count=atoms,
        species=("H",) * atoms,
        chemical_formula="H2",
        pbc=(False, False, False),
        cell_A=(0.0,) * 9,
        volume_A3=0.0,
        constraints=(),
        topology_edge_counts={
            topology_key(5.0, 0.0): 2,
            topology_key(5.0, 0.5): 4,
        },
        reference={"model": {"steps": 10 + order}},
    )


def _manifest() -> WorkloadManifest:
    return WorkloadManifest(
        workload_id="TEST-v1",
        version=1,
        family="test",
        operation="optimization",
        cell_mode="fixed",
        arrival_mode="closed",
        jobs=(_job(0), _job(1, atoms=4, duplicate="c" * 64)),
        metadata={"purpose": "unit-test"},
    ).seal()


def test_manifest_round_trip_and_csv_projection(tmp_path):
    manifest = _manifest()
    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "manifest.csv"

    write_workload_manifest(json_path, manifest)
    write_workload_jobs_csv(csv_path, manifest)

    assert read_workload_manifest(json_path) == manifest
    assert (
        csv_path.read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("workload_id,manifest_sha256,system_id")
    )
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3


def test_manifest_rejects_tampering_and_noncontiguous_order(tmp_path):
    manifest = _manifest()
    payload = manifest.to_dict()
    payload["metadata"]["purpose"] = "tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash"):
        read_workload_manifest(path)
    with pytest.raises(ValueError, match="job order"):
        WorkloadManifest(
            workload_id="bad",
            version=1,
            family="test",
            operation="optimization",
            cell_mode="fixed",
            arrival_mode="closed",
            jobs=(_job(1),),
            metadata={},
        )


def test_task_profile_summarizes_cost_and_reference_variation():
    manifest = _manifest()
    profile = TaskProfile.from_manifest(
        manifest,
        active_edge_key=topology_key(5.0, 0.0),
        candidate_edge_key=topology_key(5.0, 0.5),
        reference_model="model",
    )

    assert profile.pool_size == 2
    assert profile.unique_structure_count == 2
    assert profile.atom_count_mean == 3.0
    assert profile.atom_count_cv == pytest.approx(1.0 / 3.0)
    assert profile.total_atoms == 6
    assert profile.active_edges_mean == 2.0
    assert profile.active_edges_cv == 0.0
    assert profile.total_active_edges == 4
    assert profile.candidate_edges_mean == 4.0
    assert profile.candidate_edges_cv == 0.0
    assert profile.total_candidate_edges == 8
    assert profile.candidate_to_active_ratio == 2.0
    assert profile.reference_step_mean == 10.5
    assert profile.reference_step_cv == pytest.approx(0.5 / 10.5)


def test_normalized_structure_hash_tracks_geometry_not_metadata():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    original = normalized_structure_sha256(atoms)
    atoms.info["label"] = "ignored"
    assert normalized_structure_sha256(atoms) == original
    atoms.positions[1, 0] += 0.01
    assert normalized_structure_sha256(atoms) != original
