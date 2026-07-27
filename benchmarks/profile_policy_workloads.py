#!/usr/bin/env python3
"""Generate model-specific edge profiles for policy workloads."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip.core.neighbors import neighbor_list  # noqa: E402
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _cv(values: list[int]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def _profile(manifest_path: Path, dataset_dir: Path, cutoff: float) -> dict[str, object]:
    manifest = read_workload_manifest(manifest_path)
    atom_counts = [job.atom_count for job in manifest.jobs]
    active = []
    candidate = []
    cache: dict[tuple[str, int], tuple[int, int]] = {}
    for job in manifest.jobs:
        key = (job.source_path, job.frame_index)
        counts = cache.get(key)
        if counts is None:
            atoms = read(dataset_dir / job.source_path, index=job.frame_index)
            counts = (
                len(neighbor_list("i", atoms, cutoff)),
                len(neighbor_list("i", atoms, cutoff + 0.5)),
            )
            cache[key] = counts
        active.append(counts[0])
        candidate.append(counts[1])
    unique = len({job.normalized_structure_sha256 for job in manifest.jobs})
    return {
        "schema_version": 1,
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "family": manifest.family,
        "operation": manifest.operation,
        "pool_size": len(manifest.jobs),
        "arrival_mode": manifest.arrival_mode,
        "cell_mode": manifest.cell_mode,
        "unique_structure_count": unique,
        "duplicate_job_count": len(manifest.jobs) - unique,
        "atom_count_min": min(atom_counts),
        "atom_count_max": max(atom_counts),
        "atom_count_mean": sum(atom_counts) / len(atom_counts),
        "atom_count_cv": _cv(atom_counts) if len(set(atom_counts)) > 1 else 0.0,
        "total_atoms": sum(atom_counts),
        "active_edge_key": f"cutoff={cutoff:.3f}_skin=0.000",
        "active_edges_min": min(active),
        "active_edges_max": max(active),
        "active_edges_mean": sum(active) / len(active),
        "active_edges_cv": _cv(active),
        "total_active_edges": sum(active),
        "candidate_edge_key": f"cutoff={cutoff:.3f}_skin=0.500",
        "candidate_edges_min": min(candidate),
        "candidate_edges_max": max(candidate),
        "candidate_edges_mean": sum(candidate) / len(candidate),
        "candidate_edges_cv": _cv(candidate),
        "total_candidate_edges": sum(candidate),
        "candidate_to_active_ratio": sum(candidate) / sum(active),
        "reference_step_mean": None,
        "reference_step_cv": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/matrix.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "benchmarks/workloads/profiles",
    )
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    for workload in matrix["workloads"]:
        manifest = ROOT / workload["manifest"]
        dataset_dir = Path(workload["dataset_dir"])
        for model, cutoff in (("atombit", 6.0), ("mace_off_small", 5.0)):
            profile = _profile(manifest, dataset_dir, cutoff)
            output = args.output_root / model / f"{workload['workload_id']}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(output)


if __name__ == "__main__":
    main()
