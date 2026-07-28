#!/usr/bin/env python3
"""Generate signed 256-unique-structure refill calibration workloads."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import (  # noqa: E402
    RobustnessWorkloadInputs,
    build_robustness_family_workload,
    build_robustness_workloads,
    write_workload_jobs_csv,
    write_workload_manifest,
)


def _relabel(manifest, label: str):
    workload_id = f"OPT-RF-U256-{label}-R256-v1"
    jobs = tuple(
        replace(
            job,
            system_id=f"{workload_id}:{index:04d}",
            order=index,
        )
        for index, job in enumerate(manifest.jobs)
    )
    metadata = {
        **manifest.metadata,
        "claim_role": "refill_predictor_calibration",
        "unique_structures": 256,
        "repetitions": 1,
        "selection": (
            f"{manifest.metadata['selection']}; no repeated structures"
        ),
    }
    return replace(
        manifest,
        workload_id=workload_id,
        jobs=jobs,
        metadata=metadata,
        manifest_sha256="",
    ).seal()


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
        default=(
            ROOT
            / "experiments"
            / "refill-offline-predictor"
            / "workloads"
        ),
    )
    args = parser.parse_args()

    inputs = RobustnessWorkloadInputs(
        dataset_dir=args.dataset_dir,
        seed=20260725,
        candidate_count=256,
        unique_structures=256,
        pool_size=256,
    )
    generated = build_robustness_workloads(inputs)
    selected = {
        label: generated[f"OPT-RB-{label}-R256-v1"]
        for label in (
            "GUFJOG44",
            "SOXLEX48",
            "XATMOV88",
            "OBEQIX220",
            "ROFA-MIX",
            "ROFB296",
        )
    }
    for label, family, atom_count in (
        ("XAFPAY172", "XAFPAY", 172),
        ("AXOSOW64", "AXOSOW", 64),
        ("BOQWIN116", "BOQWIN", 116),
    ):
        selected[label] = build_robustness_family_workload(
            inputs,
            label=label,
            family=family,
            expected_atom_counts={atom_count},
        )

    manifest_dir = args.output_dir / "manifests"
    index = {"schema_version": 1, "workloads": {}}
    for label, source in sorted(selected.items()):
        manifest = _relabel(source, label)
        json_path = manifest_dir / f"{manifest.workload_id}.json"
        csv_path = manifest_dir / f"{manifest.workload_id}.csv"
        write_workload_manifest(json_path, manifest)
        write_workload_jobs_csv(csv_path, manifest)
        unique_count = len(
            {
                job.normalized_structure_sha256
                for job in manifest.jobs
            }
        )
        if unique_count != len(manifest.jobs):
            raise ValueError(
                f"{label} has {unique_count} unique structures for "
                f"{len(manifest.jobs)} jobs"
            )
        index["workloads"][label] = {
            "manifest": str(json_path),
            "manifest_sha256": manifest.manifest_sha256,
            "jobs": len(manifest.jobs),
            "unique_structures": unique_count,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
