from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from batch_mlip import (
    OMCCSPWorkloadInputs,
    build_omc_csp_workloads,
    validate_omc_csp_workload_directory,
    write_omc_csp_workloads,
)
from batch_mlip.workloads import read_workload_manifest


def _write_family(
    root: Path,
    family: str,
    *,
    atom_count: int,
    candidate_count: int = 6,
) -> None:
    output = root / family / "structures"
    output.mkdir(parents=True)
    for index in range(candidate_count):
        positions = np.column_stack(
            (
                np.linspace(0.2, 2.0, atom_count),
                np.full(atom_count, 0.1 * index),
                np.linspace(0.3, 1.7, atom_count),
            )
        )
        atoms = Atoms(
            "H" * atom_count,
            positions=positions,
            cell=np.diag([6.0 + 0.1 * index, 6.5, 7.0]),
            pbc=True,
        )
        write(output / f"{family}_candidate_{index:03d}.cif", atoms)


def test_omc_csp_workloads_are_unique_nested_and_balanced(tmp_path):
    dataset = tmp_path / "test_set"
    _write_family(dataset, "FAMA", atom_count=2)
    _write_family(dataset, "FAMB", atom_count=2)
    _write_family(dataset, "FAMC", atom_count=10)
    (dataset / "EMPTY").mkdir(parents=True)
    reference = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    write(
        dataset / "FAMA" / "structures" / "FAMA_exp_reference.cif",
        reference,
    )

    workloads, catalog = build_omc_csp_workloads(
        OMCCSPWorkloadInputs(
            dataset_dir=dataset,
            pool_sizes=(2, 4),
            cutoff_A=2.0,
            skin_A=0.2,
        )
    )

    assert catalog["replication"] is False
    assert catalog["nonempty_family_count"] == 3
    assert catalog["empty_families"] == ["EMPTY"]
    assert catalog["experimental_reference_paths"] == [
        "FAMA/structures/FAMA_exp_reference.cif"
    ]
    assert len(catalog["construction_sha256"]) == 64
    assert any("MIX-ALL" in workload_id for workload_id in workloads)
    assert any("COSTBAND" in workload_id for workload_id in workloads)

    for manifest in workloads.values():
        manifest.verify()
        paths = [job.source_path for job in manifest.jobs]
        hashes = [
            job.normalized_structure_sha256 for job in manifest.jobs
        ]
        assert len(paths) == len(set(paths))
        assert len(hashes) == len(set(hashes))
        assert all("_exp_" not in path for path in paths)
        assert manifest.metadata["replication"] is False

    for manifest in workloads.values():
        parent_id = manifest.metadata["parent_pool_id"]
        if parent_id is None:
            continue
        parent = workloads[parent_id]
        assert [job.source_path for job in manifest.jobs] == [
            job.source_path for job in parent.jobs[: len(manifest.jobs)]
        ]

    mixed_parent = next(
        manifest
        for workload_id, manifest in workloads.items()
        if "MIX-ALL-P4" in workload_id
    )
    counts = mixed_parent.metadata["statistics"]["family_counts"]
    assert max(counts.values()) - min(counts.values()) <= 1

    output = tmp_path / "constructed"
    write_omc_csp_workloads(output, workloads, catalog)
    written_index = json.loads(
        (output / "index.json").read_text(encoding="utf-8")
    )
    assert written_index["construction_sha256"] == catalog[
        "construction_sha256"
    ]
    first_id = sorted(workloads)[0]
    restored = read_workload_manifest(
        output / "manifests" / f"{first_id}.json"
    )
    assert restored.manifest_sha256 == workloads[first_id].manifest_sha256
    validation = validate_omc_csp_workload_directory(output)
    assert validation["construction_sha256"] == catalog[
        "construction_sha256"
    ]
    assert validation["replication"] is False
