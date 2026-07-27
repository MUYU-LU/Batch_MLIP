#!/usr/bin/env python3
"""Generate signed cross-family robustness workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import TaskProfile  # noqa: E402
from batch_mlip.workloads import (  # noqa: E402
    RobustnessWorkloadInputs,
    build_robustness_family_workload,
    build_robustness_workloads,
    topology_key,
    write_workload_jobs_csv,
    write_workload_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/robustness/workloads"),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--skip-defaults",
        action="store_true",
        help="generate only workloads supplied with --additional-family",
    )
    parser.add_argument(
        "--additional-family",
        action="append",
        default=[],
        metavar="LABEL:FAMILY:ATOM_COUNT",
        help="add a homogeneous family using the same signed R256 protocol",
    )
    args = parser.parse_args()

    inputs = RobustnessWorkloadInputs(
        dataset_dir=args.dataset_dir,
        seed=args.seed,
    )
    manifests = {} if args.skip_defaults else build_robustness_workloads(inputs)
    for specification in args.additional_family:
        try:
            label, family, atom_count_text = specification.split(":")
            atom_count = int(atom_count_text)
        except ValueError as error:
            parser.error("--additional-family must be LABEL:FAMILY:ATOM_COUNT")
            raise AssertionError from error
        manifest = build_robustness_family_workload(
            inputs,
            label=label,
            family=family,
            expected_atom_counts={atom_count},
        )
        if manifest.workload_id in manifests:
            parser.error(f"duplicate workload {manifest.workload_id}")
        manifests[manifest.workload_id] = manifest
    manifest_dir = args.output_dir / "manifests"
    profile_dir = args.output_dir / "profiles"
    active_key = topology_key(6.0, 0.0)
    candidate_key = topology_key(6.0, 0.5)
    index = {
        "schema_version": 1,
        "dataset_dir": str(args.dataset_dir),
        "selection_seed": args.seed,
        "selection": (
            "32 unique structures selected from 256 deterministic candidates "
            "uniformly across 6 A directed-edge density"
        ),
        "workloads": {},
    }
    for workload_id, manifest in sorted(manifests.items()):
        json_path = manifest_dir / f"{workload_id}.json"
        csv_path = manifest_dir / f"{workload_id}.csv"
        write_workload_manifest(json_path, manifest)
        write_workload_jobs_csv(csv_path, manifest)
        profile = TaskProfile.from_manifest(
            manifest,
            active_edge_key=active_key,
            candidate_edge_key=candidate_key,
        )
        profile_path = profile_dir / f"{workload_id}.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        index["workloads"][workload_id] = {
            "manifest_json": str(json_path),
            "manifest_csv": str(csv_path),
            "manifest_sha256": manifest.manifest_sha256,
            "profile_json": str(profile_path),
            "jobs": len(manifest.jobs),
            "unique_structures": len(
                {
                    job.normalized_structure_sha256
                    for job in manifest.jobs
                }
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
