#!/usr/bin/env python3
"""Generate nested, signed, unique-CIF refill transfer workloads."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import (  # noqa: E402
    RobustnessWorkloadInputs,
    build_robustness_family_workload,
    write_workload_jobs_csv,
    write_workload_manifest,
)

FAMILIES = {
    "XATMOV88": ("XATMOV", 88),
    "XAFPAY172": ("XAFPAY", 172),
    "SOXLEX48": ("SOXLEX", 48),
    "BOQWIN116": ("BOQWIN", 116),
}
POOL_SIZES = (128, 512, 1024)


def _nested_indices(total: int, count: int) -> tuple[int, ...]:
    if count == 1:
        return (0,)
    return tuple(round(index * (total - 1) / (count - 1)) for index in range(count))


def _subset(source, label: str, pool_size: int):
    workload_id = f"OPT-RFT-{label}-R{pool_size}-v1"
    selected = [source.jobs[index] for index in _nested_indices(len(source.jobs), pool_size)]
    jobs = tuple(
        replace(
            job,
            system_id=f"{workload_id}:{index:04d}",
            order=index,
        )
        for index, job in enumerate(selected)
    )
    return replace(
        source,
        workload_id=workload_id,
        jobs=jobs,
        metadata={
            **source.metadata,
            "claim_role": "refill_pool_multigpu_transfer",
            "master_pool_size": len(source.jobs),
            "unique_structures": pool_size,
            "repetitions": 1,
            "nested_density_subset": True,
        },
        manifest_sha256="",
    ).seal()


def _build_master(arguments):
    inputs, label, family, atom_count = arguments
    return label, build_robustness_family_workload(
        inputs,
        label=label,
        family=family,
        expected_atom_counts={atom_count},
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
        default=Path(__file__).with_name("workloads"),
    )
    args = parser.parse_args()

    inputs = RobustnessWorkloadInputs(
        dataset_dir=args.dataset_dir,
        seed=20260728,
        candidate_count=2048,
        unique_structures=1024,
        pool_size=1024,
    )
    manifest_dir = args.output_dir / "manifests"
    index = {"schema_version": 1, "workloads": {}}
    arguments = [
        (inputs, label, family, atom_count)
        for label, (family, atom_count) in FAMILIES.items()
    ]
    with ProcessPoolExecutor(max_workers=len(arguments)) as pool:
        masters = list(pool.map(_build_master, arguments))
    for label, master in masters:
        for pool_size in POOL_SIZES:
            manifest = _subset(master, label, pool_size)
            json_path = manifest_dir / f"{manifest.workload_id}.json"
            csv_path = manifest_dir / f"{manifest.workload_id}.csv"
            write_workload_manifest(json_path, manifest)
            write_workload_jobs_csv(csv_path, manifest)
            unique = len(
                {job.normalized_structure_sha256 for job in manifest.jobs}
            )
            if unique != pool_size:
                raise ValueError(
                    f"{manifest.workload_id} has {unique} unique structures"
                )
            index["workloads"][manifest.workload_id] = {
                "manifest": str(json_path),
                "manifest_sha256": manifest.manifest_sha256,
                "jobs": pool_size,
                "unique_structures": unique,
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
