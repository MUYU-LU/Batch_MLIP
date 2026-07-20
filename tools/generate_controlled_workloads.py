#!/usr/bin/env python3
"""Generate signed controlled-workload manifests from the fixed T2 selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_mlip import TaskProfile
from batch_mlip.workloads import (
    T2WorkloadInputs,
    build_t2_workloads,
    topology_key,
    write_workload_jobs_csv,
    write_workload_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument("--selection", type=Path, default=Path("benchmarks/t2_fixed_samples.json"))
    parser.add_argument(
        "--atombit-reference",
        type=Path,
        default=Path("runs/bfgs_fire_scaling_raw/bfgs_ase_atoms276.json"),
    )
    parser.add_argument(
        "--atombit-model-artifact",
        type=Path,
        default=Path("../AtomBit-OMC-s/model_epoch_15.pt"),
    )
    parser.add_argument(
        "--mace-reference",
        type=Path,
        default=Path("runs/bfgs_active_refill_matrix_raw/mace_atoms276_ase.json"),
    )
    parser.add_argument(
        "--mace-model-artifact",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-OFF23_small.model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/workloads"))
    args = parser.parse_args()

    for path in (args.atombit_reference, args.mace_reference):
        if not path.is_file():
            raise FileNotFoundError(
                f"required B1 reference is missing: {path}; do not infer STEPVAR labels"
            )
    manifests = build_t2_workloads(
        T2WorkloadInputs(
            dataset_dir=args.dataset_dir,
            selection_manifest=args.selection,
        ),
        atombit_reference=args.atombit_reference,
        mace_reference=args.mace_reference,
        reference_model_artifacts={
            "atombit": args.atombit_model_artifact,
            "mace_off_small": args.mace_model_artifact,
        },
    )
    manifest_dir = args.output_dir / "manifests"
    profile_dir = args.output_dir / "profiles"
    index = {
        "schema_version": 1,
        "authoritative_format": "signed JSON",
        "csv_role": "human-auditable projection",
        "workloads": {},
        "deferred": {
            "EVAL-REPLAY50-v1": (
                "requires frozen reference trajectory frames; generation must not "
                "use fabricated or repeated initial frames"
            )
        },
    }
    for workload_id, manifest in sorted(manifests.items()):
        json_path = manifest_dir / f"{workload_id}.json"
        csv_path = manifest_dir / f"{workload_id}.csv"
        write_workload_manifest(json_path, manifest)
        write_workload_jobs_csv(csv_path, manifest)
        profiles = {}
        for model, cutoff, reference_model in (
            ("atombit", 6.0, "atombit"),
            ("mace_off_small", 4.5, "mace_off_small"),
        ):
            has_reference = all(reference_model in (job.reference or {}) for job in manifest.jobs)
            profile = TaskProfile.from_manifest(
                manifest,
                active_edge_key=topology_key(cutoff, 0.0),
                candidate_edge_key=topology_key(cutoff, 0.5),
                reference_model=reference_model if has_reference else None,
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
            "unique_structures": len({job.normalized_structure_sha256 for job in manifest.jobs}),
            "profiles": profiles,
        }
    index_path = args.output_dir / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
