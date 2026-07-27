#!/usr/bin/env python3
"""Create four cost-balanced MPS manifests with four worker shards each."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import (  # noqa: E402
    WorkloadManifest,
    read_workload_manifest,
    write_workload_manifest,
)


def _costs(manifest: WorkloadManifest) -> dict[str, float]:
    edge_key = "cutoff=6.000_skin=0.000"
    edges = [job.topology_edge_counts[edge_key] for job in manifest.jobs]
    dense = [(3 * job.atom_count + 9) ** 3 for job in manifest.jobs]
    mean_edges = sum(edges) / len(edges)
    mean_dense = sum(dense) / len(dense)
    return {
        job.system_id: edge / mean_edges + dof_cost / mean_dense
        for job, edge, dof_cost in zip(
            manifest.jobs,
            edges,
            dense,
            strict=True,
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = read_workload_manifest(args.input)
    if len(source.jobs) != 3000:
        parser.error("the four-GPU MPS protocol requires exactly 3000 jobs")
    capacities = [188] * 8 + [187] * 8
    bins = [[] for _ in capacities]
    totals = [0.0] * len(capacities)
    costs = _costs(source)
    ordered = sorted(
        source.jobs,
        key=lambda job: (-costs[job.system_id], job.system_id),
    )
    for job in ordered:
        candidates = [
            index
            for index, capacity in enumerate(capacities)
            if len(bins[index]) < capacity
        ]
        selected = min(candidates, key=lambda index: (totals[index], index))
        bins[selected].append(job)
        totals[selected] += costs[job.system_id]

    gpu_bins = (
        (0, 1, 8, 9),
        (2, 3, 10, 11),
        (4, 5, 12, 13),
        (6, 7, 14, 15),
    )
    outputs = []
    for gpu_index, bin_indices in enumerate(gpu_bins):
        workload_id = f"{source.workload_id}-MPS-G{gpu_index}"
        selected_jobs = [
            job
            for bin_index in bin_indices
            for job in bins[bin_index]
        ]
        jobs = tuple(
            replace(
                job,
                system_id=f"{workload_id}:{order:04d}",
                order=order,
            )
            for order, job in enumerate(selected_jobs)
        )
        manifest = replace(
            source,
            workload_id=workload_id,
            jobs=jobs,
            metadata={
                **source.metadata,
                "base_workload_id": source.workload_id,
                "base_manifest_sha256": source.manifest_sha256,
                "mps_gpu_index": gpu_index,
                "mps_worker_job_counts": [188, 188, 187, 187],
                "mps_sharding": "capacity-constrained LPT on normalized edge and dense-BFGS costs",
            },
            manifest_sha256="",
        ).seal()
        output = args.output_dir / f"{workload_id}.json"
        write_workload_manifest(output, manifest)
        outputs.append(
            {
                "path": str(output),
                "jobs": len(jobs),
                "manifest_sha256": manifest.manifest_sha256,
                "worker_costs": [totals[index] for index in bin_indices],
            }
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
