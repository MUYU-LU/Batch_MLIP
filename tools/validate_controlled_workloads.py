#!/usr/bin/env python3
"""Validate signed T2 workload manifests against their source structures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ase.io import read

from batch_mlip.core.neighbors import neighbor_list
from batch_mlip.workloads import (
    TaskProfile,
    normalized_structure_sha256,
    read_workload_manifest,
    topology_key,
)

EXPECTED = {
    "OPT-FIXED50-v1": (256, 32, {276: 256}, "variable"),
    "OPT-H276-FIXED-R256-v1": (256, 32, {276: 256}, "fixed"),
    "OPT-H276-R256-v1": (256, 32, {276: 256}, "variable"),
    "OPT-H46-R256-v1": (256, 32, {46: 256}, "variable"),
    "OPT-MIX-R256-v1": (256, 64, {46: 128, 276: 128}, "variable"),
    "OPT-MIX-R32-v1": (32, 32, {46: 16, 276: 16}, "variable"),
    "OPT-STEPVAR-ATOMBIT-R256-v1": (256, 16, {276: 256}, "variable"),
    "OPT-STEPVAR-MACE-OFF-SMALL-R256-v1": (
        256,
        16,
        {276: 256},
        "variable",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_manifest(
    workload_id: str,
    entry: dict[str, Any],
    *,
    dataset_dir: Path,
    source_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_jobs, expected_unique, expected_atoms, expected_cell_mode = EXPECTED[workload_id]
    manifest = read_workload_manifest(entry["manifest_json"])
    if manifest.workload_id != workload_id:
        raise ValueError(f"index key and manifest ID differ for {workload_id}")
    if manifest.manifest_sha256 != entry["manifest_sha256"]:
        raise ValueError(f"index digest differs for {workload_id}")
    if len(manifest.jobs) != expected_jobs:
        raise ValueError(f"unexpected job count for {workload_id}")
    unique = {job.normalized_structure_sha256 for job in manifest.jobs}
    if len(unique) != expected_unique:
        raise ValueError(f"unexpected unique-structure count for {workload_id}")
    if Counter(job.atom_count for job in manifest.jobs) != Counter(expected_atoms):
        raise ValueError(f"unexpected atom-count distribution for {workload_id}")
    if manifest.cell_mode != expected_cell_mode:
        raise ValueError(f"unexpected cell mode for {workload_id}")
    if not all(all(job.pbc) for job in manifest.jobs):
        raise ValueError(f"non-periodic job in T2 workload {workload_id}")

    workload_sources: set[str] = set()
    duplicate_sources: dict[str, set[str]] = {}
    for job in manifest.jobs:
        source = dataset_dir / job.source_path
        workload_sources.add(job.source_path)
        if job.source_path not in source_cache:
            atoms = read(source)
            source_cache[job.source_path] = {
                "source_sha256": _sha256_file(source),
                "normalized_structure_sha256": normalized_structure_sha256(atoms),
                "atom_count": len(atoms),
                "species": tuple(atoms.get_chemical_symbols()),
                "chemical_formula": atoms.get_chemical_formula(mode="hill"),
                "pbc": tuple(bool(value) for value in atoms.pbc),
                "cell_A": tuple(float(value) for value in atoms.cell.array.reshape(-1)),
                "volume_A3": float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0,
                "constraints": tuple(type(item).__name__ for item in atoms.constraints),
                "topology_edge_counts": {
                    topology_key(cutoff, skin): int(len(neighbor_list("i", atoms, cutoff + skin)))
                    for cutoff in (4.5, 6.0)
                    for skin in (0.0, 0.25, 0.5, 1.0)
                },
            }
        expected = source_cache[job.source_path]
        if job.source_sha256 != expected["source_sha256"]:
            raise ValueError(f"source digest differs for {job.source_path}")
        for field in (
            "normalized_structure_sha256",
            "atom_count",
            "species",
            "chemical_formula",
            "pbc",
            "cell_A",
            "volume_A3",
            "constraints",
            "topology_edge_counts",
        ):
            if getattr(job, field) != expected[field]:
                raise ValueError(f"source-derived {field} differs for {job.source_path}")
        duplicate_sources.setdefault(job.duplicate_group, set()).add(job.source_path)
        for cutoff in (4.5, 6.0):
            counts = [
                job.topology_edge_counts[topology_key(cutoff, skin)]
                for skin in (0.0, 0.25, 0.5, 1.0)
            ]
            if counts != sorted(counts):
                raise ValueError(f"edge counts are not monotonic for {job.system_id}")
    if any(len(sources) != 1 for sources in duplicate_sources.values()):
        raise ValueError(f"duplicate group aliases multiple sources in {workload_id}")

    if workload_id.startswith("OPT-STEPVAR-"):
        model = "atombit" if "ATOMBIT" in workload_id else "mace_off_small"
        references = [(job.reference or {}).get(model) for job in manifest.jobs]
        if any(reference is None for reference in references):
            raise ValueError(f"missing model reference for {workload_id}")
        strata = Counter(reference["stratum"] for reference in references if reference)
        if strata != Counter({"easy": 128, "hard": 128}):
            raise ValueError(f"unbalanced reference strata for {workload_id}")
        provenance = manifest.metadata.get("reference_provenance", {})
        if len(provenance.get("model_checkpoint_sha256", "")) != 64:
            raise ValueError(f"missing model hash for {workload_id}")

    with Path(entry["manifest_csv"]).open(newline="", encoding="utf-8") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != expected_jobs:
            raise ValueError(f"CSV row count differs for {workload_id}")
    for model, profile_path in entry["profiles"].items():
        profile = TaskProfile.from_dict(json.loads(Path(profile_path).read_text(encoding="utf-8")))
        if profile.workload_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError(f"{model} profile digest differs for {workload_id}")
        if profile.pool_size != expected_jobs:
            raise ValueError(f"{model} profile size differs for {workload_id}")

    return {
        "jobs": len(manifest.jobs),
        "unique_structures": len(unique),
        "atom_counts": dict(sorted(Counter(job.atom_count for job in manifest.jobs).items())),
        "manifest_sha256": manifest.manifest_sha256,
        "source_files_verified": len(workload_sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("benchmarks/workloads/index.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    index = json.loads(args.index.read_text(encoding="utf-8"))
    if set(index["workloads"]) != set(EXPECTED):
        raise ValueError("workload index does not contain the expected controlled suite")
    source_cache: dict[str, dict[str, Any]] = {}
    results = {
        workload_id: _validate_manifest(
            workload_id,
            entry,
            dataset_dir=args.dataset_dir,
            source_cache=source_cache,
        )
        for workload_id, entry in sorted(index["workloads"].items())
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "validated_workloads": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
