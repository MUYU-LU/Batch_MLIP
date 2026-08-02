#!/usr/bin/env python3
"""Add model-specific topology counts and build a layered planning profile."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    BatchedBFGS,
    FrechetCellFilter,
    planning_profile_from_manifest,
    write_planning_profile,
)
from batch_mlip.core.neighbors import neighbor_list  # noqa: E402
from batch_mlip.workloads import (  # noqa: E402
    WorkloadManifest,
    read_workload_manifest,
    topology_key,
    write_workload_manifest,
)


def _topology_counts(
    item: tuple[str, str, int, float, float],
) -> tuple[str, int, int, int]:
    dataset_dir, source_path, frame_index, cutoff, skin = item
    atoms = read(Path(dataset_dir) / source_path, index=frame_index)
    active = len(neighbor_list("i", atoms, cutoff))
    candidate = len(neighbor_list("i", atoms, cutoff + skin))
    return source_path, frame_index, active, candidate


def enrich_manifest(
    manifest: WorkloadManifest,
    counts: dict[tuple[str, int], tuple[int, int]],
    *,
    workload_id: str,
    cutoff: float,
    skin: float,
    model_id: str,
) -> WorkloadManifest:
    """Return a sealed manifest with an additional model topology contract."""

    active_key = topology_key(cutoff, 0.0)
    candidate_key = topology_key(cutoff, skin)
    jobs = []
    for job in manifest.jobs:
        active, candidate = counts[(job.source_path, job.frame_index)]
        topology = {
            **job.topology_edge_counts,
            active_key: active,
            candidate_key: candidate,
        }
        jobs.append(replace(job, topology_edge_counts=topology))
    return replace(
        manifest,
        workload_id=workload_id,
        jobs=tuple(jobs),
        metadata={
            **manifest.metadata,
            "source_workload_id": manifest.workload_id,
            "source_workload_manifest_sha256": manifest.manifest_sha256,
            "topology_enrichment": {
                "model_id": model_id,
                "cutoff_A": cutoff,
                "skin_A": skin,
                "active_edge_key": active_key,
                "candidate_edge_key": candidate_key,
            },
        },
        manifest_sha256="",
    ).seal()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--workload-id")
    parser.add_argument(
        "--limit",
        type=int,
        help="Use a deterministic ordered prefix for a bounded validation run.",
    )
    parser.add_argument("--model-id", default="MACE-OFF23-Small")
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--model-dtype", default="torch.float64")
    parser.add_argument("--force-mode", default="native_mace")
    parser.add_argument("--neighbor-backend", default="auto")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.cutoff <= 0.0 or args.skin < 0.0 or args.workers <= 0:
        parser.error("cutoff and workers must be positive; skin must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    return args


def main() -> None:
    args = _parse_args()
    root_source = read_workload_manifest(args.manifest)
    source = root_source
    if args.limit is not None:
        if args.limit > len(root_source.jobs):
            raise ValueError("limit exceeds source workload size")
        source = replace(
            root_source,
            workload_id=f"{root_source.workload_id}-PREFIX-{args.limit}",
            jobs=root_source.jobs[: args.limit],
            metadata={
                **root_source.metadata,
                "parent_workload_id": root_source.workload_id,
                "parent_workload_manifest_sha256": root_source.manifest_sha256,
                "ordered_prefix_size": args.limit,
            },
            manifest_sha256="",
        ).seal()
    unique_sources = sorted(
        {(job.source_path, job.frame_index) for job in source.jobs}
    )
    work = [
        (
            str(args.dataset_dir),
            source_path,
            frame_index,
            args.cutoff,
            args.skin,
        )
        for source_path, frame_index in unique_sources
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        measured = list(executor.map(_topology_counts, work, chunksize=8))
    counts = {
        (source_path, frame_index): (active, candidate)
        for source_path, frame_index, active, candidate in measured
    }
    workload_id = args.workload_id or f"{source.workload_id}-MACE-OFF23-SMALL"
    manifest = enrich_manifest(
        source,
        counts,
        workload_id=workload_id,
        cutoff=args.cutoff,
        skin=args.skin,
        model_id=args.model_id,
    )
    active_key = topology_key(args.cutoff, 0.0)
    candidate_key = topology_key(args.cutoff, args.skin)
    profile = planning_profile_from_manifest(
        manifest,
        model_id=args.model_id,
        cutoff_A=args.cutoff,
        active_edge_key=active_key,
        candidate_edge_key=candidate_key,
        force_mode=args.force_mode,
        model_dtype=args.model_dtype,
        optimizer=BatchedBFGS(optimizer_dtype="float64"),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=args.skin,
        neighbor_backend=args.neighbor_backend,
    )
    write_workload_manifest(args.output_manifest, manifest)
    write_planning_profile(args.output_profile, profile)
    summary: dict[str, Any] = {
        "source_workload_id": source.workload_id,
        "source_workload_manifest_sha256": source.manifest_sha256,
        "root_source_workload_id": root_source.workload_id,
        "root_source_workload_manifest_sha256": root_source.manifest_sha256,
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "planning_profile_sha256": profile.profile_sha256,
        "structure_workload_sha256": profile.structure_workload_sha256,
        "system_count": len(manifest.jobs),
        "unique_source_count": len(unique_sources),
        "active_edge_key": active_key,
        "candidate_edge_key": candidate_key,
        "active_edges_total": sum(active for active, _ in counts.values()),
        "candidate_edges_total": sum(candidate for _, candidate in counts.values()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
