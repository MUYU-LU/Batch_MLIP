#!/usr/bin/env python3
"""Generate distinct held-out mixed-family optimization workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import TaskProfile  # noqa: E402
from batch_mlip.workloads import (  # noqa: E402
    RobustnessWorkloadInputs,
    WorkloadManifest,
    build_robustness_family_workload,
    topology_key,
    write_workload_jobs_csv,
    write_workload_manifest,
)

FAMILIES = (
    ("AXOSOW64", "AXOSOW", {64}, False),
    ("BOQQUT-MIX", "BOQQUT", {88, 176}, False),
    ("XAFPAY-MIX", "XAFPAY", {86, 172}, False),
    ("WICZUF132", "WICZUF", {132}, False),
)


def evenly_spaced_indices(available: int, selected: int) -> list[int]:
    """Include both density endpoints in a deterministic subset."""

    if available <= 0 or selected <= 0 or selected > available:
        raise ValueError("selected count must be within the available range")
    if selected == 1:
        return [0]
    return [
        round(index * (available - 1) / (selected - 1))
        for index in range(selected)
    ]


def _rank(system_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{system_id}".encode()).digest()


def build_mixed_manifest(
    family_manifests: list[WorkloadManifest],
    *,
    jobs_per_family: int,
    workload_id: str,
    seed: int,
) -> WorkloadManifest:
    """Select each family uniformly and round-robin the resulting jobs."""

    if not family_manifests:
        raise ValueError("at least one family manifest is required")
    selected_by_family = []
    for manifest in family_manifests:
        manifest.verify()
        indices = evenly_spaced_indices(
            len(manifest.jobs),
            jobs_per_family,
        )
        selected = [manifest.jobs[index] for index in indices]
        selected_by_family.append(
            sorted(selected, key=lambda job: (_rank(job.system_id, seed), job.system_id))
        )
    selected_jobs = [
        family_jobs[index]
        for index in range(jobs_per_family)
        for family_jobs in selected_by_family
    ]
    jobs = tuple(
        replace(
            job,
            system_id=f"{workload_id}:{order:04d}",
            order=order,
        )
        for order, job in enumerate(selected_jobs)
    )
    if len({job.normalized_structure_sha256 for job in jobs}) != len(jobs):
        raise ValueError("held-out validation jobs must be distinct")
    return WorkloadManifest(
        workload_id=workload_id,
        version=1,
        family="heldout_mixed_variable_horizon",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=jobs,
        metadata={
            "source_families": [
                manifest.metadata["source_family"]
                for manifest in family_manifests
            ],
            "family_manifest_sha256": {
                manifest.metadata["source_family"]: manifest.manifest_sha256
                for manifest in family_manifests
            },
            "selection_seed": seed,
            "jobs_per_family": jobs_per_family,
            "unique_structures": len(jobs),
            "synthetic_repetition": False,
            "selection": (
                "density-spanning family subsets, deterministic hash order, "
                "then family round-robin"
            ),
            "active_edge_key": topology_key(6.0, 0.0),
            "candidate_edge_key": topology_key(6.0, 0.5),
            "claim_role": "heldout_validation",
        },
    ).seal()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--family-selection-count", type=int, default=256)
    args = parser.parse_args()
    if args.candidate_count < args.family_selection_count:
        parser.error("candidate count must cover the family selection count")

    inputs = RobustnessWorkloadInputs(
        dataset_dir=args.dataset_dir,
        seed=args.seed,
        candidate_count=args.candidate_count,
        unique_structures=args.family_selection_count,
        pool_size=args.family_selection_count,
    )
    family_manifests = [
        build_robustness_family_workload(
            inputs,
            label=label,
            family=family,
            expected_atom_counts=atom_counts,
            balanced_atom_counts=balanced,
        )
        for label, family, atom_counts, balanced in FAMILIES
    ]
    manifest_dir = args.output_dir / "manifests"
    profile_dir = args.output_dir / "profiles"
    manifests = [
        build_mixed_manifest(
            family_manifests,
            jobs_per_family=jobs_per_family,
            workload_id=f"OPT-HOLDOUT-MIX-R{jobs_per_family * len(FAMILIES)}-v1",
            seed=args.seed,
        )
        for jobs_per_family in (64, 256)
    ]
    index = {
        "schema_version": 1,
        "dataset_dir": str(args.dataset_dir),
        "selection_seed": args.seed,
        "candidate_count_per_family": args.candidate_count,
        "selected_count_per_family": args.family_selection_count,
        "workloads": {},
    }
    for manifest in manifests:
        json_path = manifest_dir / f"{manifest.workload_id}.json"
        csv_path = manifest_dir / f"{manifest.workload_id}.csv"
        write_workload_manifest(json_path, manifest)
        write_workload_jobs_csv(csv_path, manifest)
        profile = TaskProfile.from_manifest(
            manifest,
            active_edge_key=topology_key(6.0, 0.0),
            candidate_edge_key=topology_key(6.0, 0.5),
        )
        profile_path = profile_dir / f"{manifest.workload_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index["workloads"][manifest.workload_id] = {
            "manifest_json": str(json_path),
            "manifest_csv": str(csv_path),
            "manifest_sha256": manifest.manifest_sha256,
            "profile_json": str(profile_path),
            "jobs": len(manifest.jobs),
            "unique_structures": len(
                {job.normalized_structure_sha256 for job in manifest.jobs}
            ),
        }
    index_path = args.output_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
