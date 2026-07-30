#!/usr/bin/env python3
"""Derive split-safe OMC-CSP scheduler workloads from signed v1 pools.

The source v1 construction selected unique CIFs from every available family.
This utility never rescans or changes that construction.  It derives immutable
scheduler-epoch manifests from the v1 P2048 family pools, keeping all nested
P64/P512/P2048 members of a family in exactly one development, validation, or
test split.  The mixed pools are constructed independently within each split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from batch_mlip.workloads import (
    TaskProfile,
    WorkloadJob,
    WorkloadManifest,
    read_workload_manifest,
    write_workload_manifest,
)

SPLITS = {
    "development": (
        "BOQWIN",
        "GUFJOG",
        "HAMTIZ",
        "KONTIQ",
        "NACJAF",
        "OBEQUJ",
        "PAHYON",
        "SOXLEX",
        "UJIRIO",
        "XAFPAY",
        "XATMOV",
        "rof-b",
    ),
    "validation": (
        "AXOSOW",
        "BOQQUT",
        "WICZUF",
        "WIDBAO",
        "XAFQIH",
    ),
    "test": (
        "JAYDUI",
        "OBEQIX",
        "XULDUD",
        "rof-a",
        "rof-c",
    ),
}
EXCLUDED_FAMILIES = {
    "XIFZOF": "contains Si, absent from AtomBit smooth-rms checkpoint",
}
POOL_SIZES = (64, 512, 2048)
DEVELOPMENT_SELECTED_EXTREME_POOL_FAMILIES = (
    "GUFJOG",
    "KONTIQ",
    "BOQWIN",
    "XAFPAY",
    "rof-b",
)
VALIDATION_SELECTED_EXTREME_POOL_FAMILIES = ("BOQQUT", "WIDBAO")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_family(family: str) -> str:
    return family.upper()


def _workload_id(
    split: str,
    label: str,
    *,
    pool_size: int,
    scope: str,
    cost_class: str,
) -> str:
    return "OPT-OMC-SCHED-E1-" f"{split.upper()}-{label}-P{pool_size}-{scope}-{cost_class}-v2"


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def _statistics(
    jobs: Sequence[WorkloadJob],
    *,
    candidate_edge_key: str,
) -> dict[str, Any]:
    atoms = [job.atom_count for job in jobs]
    candidates = [job.topology_edge_counts[candidate_edge_key] for job in jobs]
    costs = [
        16.0 * (3 * atom_count + 9) ** 2 + 256.0 * atom_count + 64.0 * edge_count
        for atom_count, edge_count in zip(atoms, candidates, strict=True)
    ]
    return {
        "atom_count_min": min(atoms),
        "atom_count_max": max(atoms),
        "candidate_edges_min": min(candidates),
        "candidate_edges_max": max(candidates),
        "family_counts": dict(sorted(Counter(job.group_id for job in jobs).items())),
        "planning_cost_min": min(costs),
        "planning_cost_max": max(costs),
        "planning_cost_ratio": max(costs) / min(costs),
        "planning_cost_cv": _coefficient_of_variation(costs),
    }


def _cost_class(statistics: dict[str, Any], *, scope: str) -> str:
    if scope == "INTER":
        return "WIDE"
    return "NARROW" if statistics["planning_cost_ratio"] <= 2.0 else "WIDE"


def _derive_manifest(
    source_jobs: Sequence[WorkloadJob],
    *,
    workload_id: str,
    split: str,
    scope: str,
    source_families: Sequence[str],
    parent_pool_id: str | None,
    source_manifest_ids: Sequence[str],
    source_manifest_hashes: Sequence[str],
    selection: str,
    active_edge_key: str,
    candidate_edge_key: str,
    cutoff_A: float,
    skin_A: float,
) -> WorkloadManifest:
    jobs = tuple(
        replace(
            job,
            system_id=f"{workload_id}:{order:05d}",
            order=order,
        )
        for order, job in enumerate(source_jobs)
    )
    statistics = _statistics(jobs, candidate_edge_key=candidate_edge_key)
    return WorkloadManifest(
        workload_id=workload_id,
        version=2,
        family="variable_horizon_closed",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=jobs,
        metadata={
            "application": "omc_csp",
            "scheduler_epoch": 1,
            "scheduler_split": split,
            "family_scope": scope.lower(),
            "source_families": list(source_families),
            "source_manifest_ids": list(source_manifest_ids),
            "source_manifest_sha256": list(source_manifest_hashes),
            "parent_pool_id": parent_pool_id,
            "selection": selection,
            "selection_seed": 20260729,
            "pool_notation": "P denotes specific unique CIF candidates",
            "replication": False,
            "active_edge_key": active_edge_key,
            "candidate_edge_key": candidate_edge_key,
            "cutoff_A": cutoff_A,
            "skin_A": skin_A,
            "cost_ratio_max": 2.0,
            "planning_cost_formula": ("16*D2 + 256*N + 64*E_candidate; D=3*N+9"),
            "statistics": statistics,
        },
    ).seal()


def _load_source_family_pools(
    source_dir: Path,
    *,
    families: Iterable[str],
) -> tuple[dict[str, WorkloadManifest], dict[str, Any]]:
    catalog = json.loads((source_dir / "index.json").read_text(encoding="utf-8"))
    sources = {}
    for family in families:
        prefix = f"OPT-OMC-{_display_family(family)}-P2048-INTRA-"
        matches = [
            workload_id for workload_id in catalog["workloads"] if workload_id.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(f"could not resolve one v1 P2048 pool for {family}")
        manifest = read_workload_manifest(source_dir / "manifests" / f"{matches[0]}.json")
        if len(manifest.jobs) != 2048:
            raise ValueError(f"{family} source pool is not P2048")
        sources[family] = manifest
    return sources, catalog


def _balanced_round_robin(
    sources: dict[str, WorkloadManifest],
    *,
    count: int,
) -> tuple[WorkloadJob, ...]:
    selected: list[WorkloadJob] = []
    seen_structures: set[str] = set()
    positions = {family: 0 for family in sources}
    families = tuple(sorted(sources))
    while len(selected) < count:
        progressed = False
        for family in families:
            jobs = sources[family].jobs
            while positions[family] < len(jobs):
                job = jobs[positions[family]]
                positions[family] += 1
                if job.normalized_structure_sha256 in seen_structures:
                    continue
                selected.append(job)
                seen_structures.add(job.normalized_structure_sha256)
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError("insufficient unique structures for balanced pool")
    return tuple(selected)


def build_scheduler_epoch_workloads(
    source_dir: str | Path,
) -> tuple[dict[str, WorkloadManifest], dict[str, Any]]:
    """Return the immutable scheduler epoch 1 workload collection."""

    source_dir = Path(source_dir)
    all_families = tuple(family for values in SPLITS.values() for family in values)
    if len(all_families) != len(set(all_families)):
        raise ValueError("scheduler split overlaps families")
    source_pools, source_catalog = _load_source_family_pools(
        source_dir,
        families=all_families,
    )
    source_family_names = set(source_catalog["families"])
    expected = source_family_names - {"OBEQET", *EXCLUDED_FAMILIES}
    if set(all_families) != expected:
        missing = sorted(expected - set(all_families))
        extra = sorted(set(all_families) - expected)
        raise ValueError(
            f"scheduler split must cover compatible v1 families; missing={missing}, extra={extra}"
        )

    first_source = next(iter(source_pools.values()))
    active_edge_key = first_source.metadata["active_edge_key"]
    candidate_edge_key = first_source.metadata["candidate_edge_key"]
    cutoff_A = float(first_source.metadata["cutoff_A"])
    skin_A = float(first_source.metadata["skin_A"])
    if any(
        source.metadata["active_edge_key"] != active_edge_key
        or source.metadata["candidate_edge_key"] != candidate_edge_key
        or source.metadata["cutoff_A"] != cutoff_A
        or source.metadata["skin_A"] != skin_A
        for source in source_pools.values()
    ):
        raise ValueError("v1 source pools have inconsistent graph contracts")

    workloads: dict[str, WorkloadManifest] = {}
    for split, families in SPLITS.items():
        for family in families:
            source = source_pools[family]
            parent_id = None
            derived = {}
            for size in reversed(POOL_SIZES):
                jobs = source.jobs[:size]
                statistics = _statistics(jobs, candidate_edge_key=candidate_edge_key)
                workload_id = _workload_id(
                    split,
                    _display_family(family),
                    pool_size=size,
                    scope="INTRA",
                    cost_class=_cost_class(statistics, scope="INTRA"),
                )
                derived[size] = _derive_manifest(
                    jobs,
                    workload_id=workload_id,
                    split=split,
                    scope="INTRA",
                    source_families=(family,),
                    parent_pool_id=parent_id,
                    source_manifest_ids=(source.workload_id,),
                    source_manifest_hashes=(source.manifest_sha256,),
                    selection=(
                        "v1 P2048 ordered prefix; family is bound to one " "scheduler split"
                    ),
                    active_edge_key=active_edge_key,
                    candidate_edge_key=candidate_edge_key,
                    cutoff_A=cutoff_A,
                    skin_A=skin_A,
                )
                parent_id = workload_id
            workloads.update(
                (manifest.workload_id, manifest) for _, manifest in sorted(derived.items())
            )

        mixed_jobs = _balanced_round_robin(
            {family: source_pools[family] for family in families},
            count=512,
        )
        statistics = _statistics(mixed_jobs, candidate_edge_key=candidate_edge_key)
        mixed_id = _workload_id(
            split,
            "MIX-ALL",
            pool_size=512,
            scope="INTER",
            cost_class=_cost_class(statistics, scope="INTER"),
        )
        source_manifests = tuple(source_pools[family] for family in families)
        workloads[mixed_id] = _derive_manifest(
            mixed_jobs,
            workload_id=mixed_id,
            split=split,
            scope="INTER",
            source_families=families,
            parent_pool_id=None,
            source_manifest_ids=tuple(manifest.workload_id for manifest in source_manifests),
            source_manifest_hashes=tuple(manifest.manifest_sha256 for manifest in source_manifests),
            selection=(
                "balanced round-robin over v1 P2048 candidates from one "
                "scheduler split; no cross-split family admission"
            ),
            active_edge_key=active_edge_key,
            candidate_edge_key=candidate_edge_key,
            cutoff_A=cutoff_A,
            skin_A=skin_A,
        )

    scheduler_matrix = {
        "development": {
            "all_family_p512": list(SPLITS["development"]),
            "selected_intra_family_extreme_pool_families": list(
                DEVELOPMENT_SELECTED_EXTREME_POOL_FAMILIES
            ),
            "mixed_p512": True,
        },
        "validation": {
            "all_family_p512": list(SPLITS["validation"]),
            "selected_intra_family_extreme_pool_families": list(
                VALIDATION_SELECTED_EXTREME_POOL_FAMILIES
            ),
            "mixed_p512": True,
        },
        "test": {
            "all_family_p64_p512_p2048": list(SPLITS["test"]),
            "mixed_p512": True,
        },
    }
    catalog = {
        "schema_version": 1,
        "construction_id": "omc-csp-scheduler-epoch1-v2",
        "source_construction_id": source_catalog["construction_id"],
        "source_construction_sha256": source_catalog["construction_sha256"],
        "source_directory": str(source_dir),
        "selection_seed": 20260729,
        "pool_sizes": list(POOL_SIZES),
        "replication": False,
        "excluded_families": EXCLUDED_FAMILIES,
        "scheduler_splits": {key: list(value) for key, value in SPLITS.items()},
        "scheduler_matrix": scheduler_matrix,
        "family_count": len(all_families),
        "workload_count": len(workloads),
        "workloads": {
            workload_id: {
                "jobs": len(manifest.jobs),
                "manifest_sha256": manifest.manifest_sha256,
                "scheduler_split": manifest.metadata["scheduler_split"],
                "family_scope": manifest.metadata["family_scope"],
                "source_families": manifest.metadata["source_families"],
                "parent_pool_id": manifest.metadata["parent_pool_id"],
            }
            for workload_id, manifest in sorted(workloads.items())
        },
    }
    catalog["construction_sha256"] = _canonical_sha256(catalog)
    return workloads, catalog


def write_scheduler_epoch_workloads(
    output_dir: str | Path,
    workloads: dict[str, WorkloadManifest],
    catalog: dict[str, Any],
) -> None:
    """Write signed manifests and legacy aggregate profile sidecars."""

    output = Path(output_dir)
    manifest_dir = output / "manifests"
    profile_dir = output / "profiles"
    for workload_id, manifest in workloads.items():
        write_workload_manifest(manifest_dir / f"{workload_id}.json", manifest)
        profile = TaskProfile.from_manifest(
            manifest,
            active_edge_key=manifest.metadata["active_edge_key"],
            candidate_edge_key=manifest.metadata["candidate_edge_key"],
        )
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / f"{workload_id}.json").write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_scheduler_epoch_workloads(output_dir: str | Path) -> dict[str, Any]:
    """Verify split isolation, hashes, nesting, and mixed-pool membership."""

    output = Path(output_dir)
    catalog = json.loads((output / "index.json").read_text(encoding="utf-8"))
    expected_hash = catalog.pop("construction_sha256")
    if _canonical_sha256(catalog) != expected_hash:
        raise ValueError("scheduler workload construction hash mismatch")
    catalog["construction_sha256"] = expected_hash
    splits = {split: set(families) for split, families in catalog["scheduler_splits"].items()}
    if sum(len(families) for families in splits.values()) != len(set().union(*splits.values())):
        raise ValueError("scheduler family splits overlap")

    manifests = {}
    for workload_id, entry in catalog["workloads"].items():
        manifest = read_workload_manifest(output / "manifests" / f"{workload_id}.json")
        manifest.verify()
        if manifest.manifest_sha256 != entry["manifest_sha256"]:
            raise ValueError(f"{workload_id}: manifest hash differs from index")
        split = manifest.metadata["scheduler_split"]
        families = set(manifest.metadata["source_families"])
        if not families or not families <= splits[split]:
            raise ValueError(f"{workload_id}: cross-split family admission")
        if families & set(catalog["excluded_families"]):
            raise ValueError(f"{workload_id}: excluded family admitted")
        paths = [job.source_path for job in manifest.jobs]
        structures = [job.normalized_structure_sha256 for job in manifest.jobs]
        if len(paths) != len(set(paths)) or len(structures) != len(set(structures)):
            raise ValueError(f"{workload_id}: duplicate job identity")
        manifests[workload_id] = manifest

    for manifest in manifests.values():
        parent_id = manifest.metadata["parent_pool_id"]
        if parent_id is None:
            continue
        parent = manifests[parent_id]
        if [job.source_path for job in manifest.jobs] != [
            job.source_path for job in parent.jobs[: len(manifest.jobs)]
        ]:
            raise ValueError(f"{manifest.workload_id}: nested prefix mismatch")

    intra = defaultdict(set)
    for manifest in manifests.values():
        if manifest.metadata["family_scope"] != "intra":
            continue
        family = manifest.metadata["source_families"][0]
        intra[family].add(len(manifest.jobs))
    expected_families = set().union(*splits.values())
    if set(intra) != expected_families:
        raise ValueError("intra-family coverage differs from scheduler split")
    if any(sizes != set(POOL_SIZES) for sizes in intra.values()):
        raise ValueError("intra-family pool sizes differ from epoch contract")
    return {
        "construction_sha256": expected_hash,
        "family_count": len(expected_families),
        "workload_count": len(manifests),
        "split_workloads": {
            split: sum(
                manifest.metadata["scheduler_split"] == split for manifest in manifests.values()
            )
            for split in sorted(splits)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workloads", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_scheduler_epoch_workloads(args.output), sort_keys=True))
        return
    if args.source_workloads is None:
        parser.error("--source-workloads is required unless --validate-only is used")
    workloads, catalog = build_scheduler_epoch_workloads(args.source_workloads)
    write_scheduler_epoch_workloads(args.output, workloads, catalog)
    print(json.dumps(validate_scheduler_epoch_workloads(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
