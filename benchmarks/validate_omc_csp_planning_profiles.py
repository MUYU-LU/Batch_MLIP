"""Validate layered OMC-CSP planning sidecars against frozen workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from batch_mlip import read_planning_profile, structure_workload_sha256
from batch_mlip.workloads import read_workload_manifest, topology_key


def _canonical_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads((args.profiles / "index.json").read_text())
    expected_index_hash = index.pop("index_sha256")
    if _canonical_sha256(index) != expected_index_hash:
        raise ValueError("planning profile index hash mismatch")
    records = index["profiles"]
    validated_systems = 0
    structure_hashes = set()
    for workload_id, record in records.items():
        profile = read_planning_profile(args.profiles / record["path"])
        manifest = read_workload_manifest(
            args.workloads / "manifests" / f"{workload_id}.json"
        )
        if profile.workload_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError(f"{workload_id}: workload manifest hash mismatch")
        expected_structure_hash = structure_workload_sha256(manifest)
        if profile.structure_workload_sha256 != expected_structure_hash:
            raise ValueError(f"{workload_id}: structure identity hash mismatch")
        if profile.profile_sha256 != record["planning_profile_sha256"]:
            raise ValueError(f"{workload_id}: profile index hash mismatch")
        if len(profile.systems) != len(manifest.jobs):
            raise ValueError(f"{workload_id}: system count mismatch")
        for system, job in zip(profile.systems, manifest.jobs, strict=True):
            graph = system.mlip_graph
            execution = system.graph_execution
            task = system.task_auxiliary
            active_key = topology_key(graph.cutoff_A, 0.0)
            candidate_key = topology_key(graph.cutoff_A, execution.skin_A)
            dimension = 3 * job.atom_count + 9
            if system.structure.atom_count != job.atom_count:
                raise ValueError(f"{workload_id}: atom count mismatch")
            if graph.active_edge_count != job.topology_edge_counts[active_key]:
                raise ValueError(f"{workload_id}: active edge mismatch")
            if (
                execution.candidate_edge_count
                != job.topology_edge_counts[candidate_key]
            ):
                raise ValueError(f"{workload_id}: candidate edge mismatch")
            if (
                task.generalized_dimension != dimension
                or task.dense_state_elements != dimension**2
                or task.dense_linear_algebra_work != dimension**3
            ):
                raise ValueError(f"{workload_id}: BFGS dimension mismatch")
            if (
                not task.variable_cell
                or not task.stress_required
                or task.cell_degrees_of_freedom != 9
                or task.state_dtype != "torch.float64"
                or "FrechetCellFilter" not in str(task.cell_method)
            ):
                raise ValueError(f"{workload_id}: task contract mismatch")
        validated_systems += len(profile.systems)
        structure_hashes.add(profile.structure_workload_sha256)
    print(
        json.dumps(
            {
                "index_sha256": expected_index_hash,
                "profile_count": len(records),
                "unique_structure_workloads": len(structure_hashes),
                "validated_system_references": validated_systems,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
