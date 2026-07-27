#!/usr/bin/env python3
"""Generate a sealed unbiased subset of distinct T2 CIF source files."""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import (  # noqa: E402
    WorkloadManifest,
    write_workload_manifest,
)
from batch_mlip.workloads.generator import _job, _structure_record  # noqa: E402


def selection_key(path: Path, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{path.name}".encode()).hexdigest()
    return digest, path.name


def _read_selected(payload: tuple[Path, str]):
    path, relative_path = payload
    return _structure_record(
        path,
        relative_path=relative_path,
        cutoffs=(6.0,),
        skins=(0.0, 0.5),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.pool_size <= 0 or args.workers <= 0:
        parser.error("pool size and workers must be positive")

    paths = sorted(args.dataset_dir.glob("*.cif"))
    if args.pool_size > len(paths):
        parser.error("pool size exceeds the number of distinct CIF paths")
    selected = sorted(
        sorted(paths, key=lambda path: selection_key(path, args.seed))[
            : args.pool_size
        ],
        key=lambda path: path.name,
    )
    payloads = [
        (path, f"structures/{path.name}")
        for path in selected
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_read_selected, payloads, chunksize=8))

    workload_id = f"OPT-T2-UNIQUE-R{args.pool_size}-v1"
    jobs = tuple(
        _job(
            record,
            workload_id=workload_id,
            dataset_id="T2_test_tgz",
            order=order,
        )
        for order, record in enumerate(records)
    )
    manifest = WorkloadManifest(
        workload_id=workload_id,
        version=1,
        family="unique_t2_variable_cell",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=jobs,
        metadata={
            "selection": "lowest seeded SHA-256 filename ranks",
            "selection_seed": args.seed,
            "available_distinct_cif_paths": len(paths),
            "selected_distinct_cif_paths": len(selected),
            "cutoffs_A": [6.0],
            "skins_A": [0.0, 0.5],
            "synthetic_repetition": False,
        },
    ).seal()
    write_workload_manifest(args.output, manifest)
    print(
        f"{manifest.workload_id} jobs={len(jobs)} "
        f"sha256={manifest.manifest_sha256}"
    )


if __name__ == "__main__":
    main()
