#!/usr/bin/env python3
"""Generate signed held-out workloads for task-aware policy validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import TaskProfile  # noqa: E402
from batch_mlip.workloads import (  # noqa: E402
    T2WorkloadInputs,
    build_task_aware_holdout_workloads,
    topology_key,
    write_workload_jobs_csv,
    write_workload_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("benchmarks/t2_fixed_samples.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/workloads"),
    )
    args = parser.parse_args()

    manifests = build_task_aware_holdout_workloads(
        T2WorkloadInputs(
            dataset_dir=args.dataset_dir,
            selection_manifest=args.selection,
            cutoffs_A=(4.5, 5.0, 6.0),
        )
    )
    manifest_dir = args.output_dir / "manifests"
    profile_dir = args.output_dir / "profiles"
    index_path = args.output_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for workload_id, manifest in sorted(manifests.items()):
        json_path = manifest_dir / f"{workload_id}.json"
        csv_path = manifest_dir / f"{workload_id}.csv"
        write_workload_manifest(json_path, manifest)
        write_workload_jobs_csv(csv_path, manifest)
        profiles = {}
        for model, cutoff in (("atombit", 6.0), ("mace_off_small", 5.0)):
            profile = TaskProfile.from_manifest(
                manifest,
                active_edge_key=topology_key(cutoff, 0.0),
                candidate_edge_key=topology_key(cutoff, 0.5),
            )
            profile_path = profile_dir / model / f"{workload_id}.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(
                json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            profiles[model] = str(profile_path)
        index["workloads"][workload_id] = {
            "manifest_json": str(json_path),
            "manifest_csv": str(csv_path),
            "manifest_sha256": manifest.manifest_sha256,
            "jobs": len(manifest.jobs),
            "unique_structures": len(
                {job.normalized_structure_sha256 for job in manifest.jobs}
            ),
            "profiles": profiles,
        }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generated": sorted(manifests),
                "index": str(index_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
