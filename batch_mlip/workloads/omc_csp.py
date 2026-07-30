"""Deterministic unique-candidate workloads for OMC crystal prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

from ..core.neighbors import neighbor_list
from .generator import normalized_structure_sha256, topology_key
from .schema import (
    TaskProfile,
    WorkloadJob,
    WorkloadManifest,
    read_workload_manifest,
    write_workload_jobs_csv,
    write_workload_manifest,
)

_COST_RATIO_MAX = 2.0
_LEGACY_OMC_BFGS_COST_PROXY_FORMULA = (
    "16*D2 + 256*N + 64*E_candidate; D=3*N+9"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _coefficient_of_variation(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass(frozen=True)
class OMCCSPWorkloadInputs:
    """Frozen rules for unique OMC-CSP candidate selection."""

    dataset_dir: Path
    dataset_id: str = "omc_csp_test_set"
    pool_sizes: tuple[int, ...] = (64, 512, 2048)
    selection_seed: int = 20260729
    cutoff_A: float = 6.0
    skin_A: float = 0.5
    cost_ratio_max: float = _COST_RATIO_MAX
    scan_growth_factor: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(self.dataset_dir)
        if (
            not self.pool_sizes
            or tuple(sorted(set(self.pool_sizes))) != self.pool_sizes
            or any(size <= 0 for size in self.pool_sizes)
        ):
            raise ValueError("pool_sizes must be unique positive ascending values")
        if self.cutoff_A <= 0.0 or self.skin_A < 0.0:
            raise ValueError("cutoff must be positive and skin non-negative")
        if self.cost_ratio_max < 1.0:
            raise ValueError("cost_ratio_max must be at least one")
        if self.scan_growth_factor < 1:
            raise ValueError("scan_growth_factor must be positive")


@dataclass(frozen=True)
class OMCCSPCandidate:
    """One immutable profiled CIF candidate."""

    family: str
    source_path: str
    source_sha256: str
    normalized_structure_sha256: str
    atom_count: int
    species: tuple[str, ...]
    chemical_formula: str
    pbc: tuple[bool, bool, bool]
    cell_A: tuple[float, ...]
    volume_A3: float
    active_edge_count: int
    candidate_edge_count: int
    dof_squared: int
    omc_variable_cell_bfgs_cost_proxy: float

    def to_job(
        self,
        *,
        workload_id: str,
        dataset_id: str,
        order: int,
        active_edge_key: str,
        candidate_edge_key: str,
    ) -> WorkloadJob:
        return WorkloadJob(
            system_id=f"{workload_id}:{order:05d}",
            group_id=self.family,
            duplicate_group=self.normalized_structure_sha256,
            order=order,
            dataset_id=dataset_id,
            source_path=self.source_path,
            source_sha256=self.source_sha256,
            normalized_structure_sha256=self.normalized_structure_sha256,
            frame_index=0,
            atom_count=self.atom_count,
            species=self.species,
            chemical_formula=self.chemical_formula,
            pbc=self.pbc,
            cell_A=self.cell_A,
            volume_A3=self.volume_A3,
            constraints=(),
            topology_edge_counts={
                active_edge_key: self.active_edge_count,
                candidate_edge_key: self.candidate_edge_count,
            },
        )


def _profile_candidate(
    arguments: tuple[str, str, float, float],
) -> OMCCSPCandidate:
    root_text, relative_path, cutoff, skin = arguments
    root = Path(root_text)
    path = root / relative_path
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="crystal system .* is not interpreted for space group.*",
            category=UserWarning,
        )
        atoms = read(path)
    if not bool(atoms.pbc.all()) or atoms.cell.rank != 3:
        raise ValueError(f"OMC-CSP candidate is not fully periodic: {relative_path}")
    distances = neighbor_list("d", atoms, cutoff + skin)
    candidate_edges = int(len(distances))
    active_edges = int(np.count_nonzero(distances < cutoff))
    atom_count = len(atoms)
    dof = 3 * atom_count + 9
    dof_squared = dof * dof
    omc_bfgs_cost_proxy = (
        16.0 * dof_squared
        + 256.0 * atom_count
        + 64.0 * candidate_edges
    )
    return OMCCSPCandidate(
        family=Path(relative_path).parts[0],
        source_path=relative_path,
        source_sha256=_sha256_file(path),
        normalized_structure_sha256=normalized_structure_sha256(atoms),
        atom_count=atom_count,
        species=tuple(atoms.get_chemical_symbols()),
        chemical_formula=atoms.get_chemical_formula(mode="hill"),
        pbc=tuple(bool(value) for value in atoms.pbc),
        cell_A=tuple(float(value) for value in atoms.cell.array.reshape(-1)),
        volume_A3=float(atoms.get_volume()),
        active_edge_count=active_edges,
        candidate_edge_count=candidate_edges,
        dof_squared=dof_squared,
        omc_variable_cell_bfgs_cost_proxy=omc_bfgs_cost_proxy,
    )


def _ranked_paths(
    root: Path,
    family_dir: Path,
    *,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    all_cifs = sorted(family_dir.rglob("*.cif"))
    references = [
        path for path in all_cifs if "_exp_" in path.name.lower()
    ]
    candidates = [
        path for path in all_cifs if "_exp_" not in path.name.lower()
    ]

    def rank(path: Path) -> tuple[bytes, str]:
        relative = path.relative_to(root).as_posix()
        payload = f"{seed}:{relative}".encode()
        return hashlib.sha256(payload).digest(), relative

    return sorted(candidates, key=rank), references


def _profile_paths(
    root: Path,
    paths: Sequence[Path],
    *,
    cutoff: float,
    skin: float,
    workers: int,
) -> list[OMCCSPCandidate]:
    arguments = [
        (
            str(root),
            path.relative_to(root).as_posix(),
            cutoff,
            skin,
        )
        for path in paths
    ]
    if workers == 1:
        return [_profile_candidate(item) for item in arguments]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_profile_candidate, arguments, chunksize=8))


def _select_unique_family(
    inputs: OMCCSPWorkloadInputs,
    family_dir: Path,
    *,
    workers: int,
) -> tuple[list[OMCCSPCandidate], list[str], int]:
    ranked, references = _ranked_paths(
        inputs.dataset_dir,
        family_dir,
        seed=inputs.selection_seed,
    )
    target = max(inputs.pool_sizes)
    if len(ranked) < target:
        raise ValueError(
            f"{family_dir.name} has {len(ranked)} candidates, fewer than P{target}"
        )
    selected: list[OMCCSPCandidate] = []
    seen_source: set[str] = set()
    seen_structure: set[str] = set()
    cursor = 0
    while len(selected) < target and cursor < len(ranked):
        remaining = target - len(selected)
        batch_size = max(64, remaining * inputs.scan_growth_factor)
        paths = ranked[cursor : min(len(ranked), cursor + batch_size)]
        cursor += len(paths)
        for record in _profile_paths(
            inputs.dataset_dir,
            paths,
            cutoff=inputs.cutoff_A,
            skin=inputs.skin_A,
            workers=workers,
        ):
            if (
                record.source_sha256 in seen_source
                or record.normalized_structure_sha256 in seen_structure
            ):
                continue
            seen_source.add(record.source_sha256)
            seen_structure.add(record.normalized_structure_sha256)
            selected.append(record)
            if len(selected) == target:
                break
    if len(selected) < target:
        raise ValueError(
            f"{family_dir.name} has only {len(selected)} unique candidates"
        )
    return (
        selected,
        [
            path.relative_to(inputs.dataset_dir).as_posix()
            for path in references
        ],
        len(ranked),
    )


def _cost_class(records: Sequence[OMCCSPCandidate], ratio_max: float) -> str:
    costs = [
        record.omc_variable_cell_bfgs_cost_proxy for record in records
    ]
    return "NARROW" if max(costs) / min(costs) <= ratio_max else "WIDE"


def _family_counts(records: Sequence[OMCCSPCandidate]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.family] += 1
    return dict(sorted(counts.items()))


def _pool_statistics(records: Sequence[OMCCSPCandidate]) -> dict[str, Any]:
    costs = [
        record.omc_variable_cell_bfgs_cost_proxy for record in records
    ]
    return {
        "atom_count_min": min(record.atom_count for record in records),
        "atom_count_max": max(record.atom_count for record in records),
        "candidate_edges_min": min(
            record.candidate_edge_count for record in records
        ),
        "candidate_edges_max": max(
            record.candidate_edge_count for record in records
        ),
        "planning_cost_min": min(costs),
        "planning_cost_max": max(costs),
        "planning_cost_ratio": max(costs) / min(costs),
        "planning_cost_cv": _coefficient_of_variation(costs),
        "family_counts": _family_counts(records),
    }


def _workload_id(
    scope: str,
    label: str,
    pool_size: int,
    cost_class: str,
) -> str:
    normalized = label.upper().replace("_", "-")
    return f"OPT-OMC-{normalized}-P{pool_size}-{scope}-{cost_class}-v1"


def _manifest(
    inputs: OMCCSPWorkloadInputs,
    records: Sequence[OMCCSPCandidate],
    *,
    workload_id: str,
    scope: str,
    selection: str,
    parent_pool_id: str | None,
) -> WorkloadManifest:
    active_key = topology_key(inputs.cutoff_A, 0.0)
    candidate_key = topology_key(inputs.cutoff_A, inputs.skin_A)
    jobs = tuple(
        record.to_job(
            workload_id=workload_id,
            dataset_id=inputs.dataset_id,
            order=order,
            active_edge_key=active_key,
            candidate_edge_key=candidate_key,
        )
        for order, record in enumerate(records)
    )
    return WorkloadManifest(
        workload_id=workload_id,
        version=1,
        family="variable_horizon_closed",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=jobs,
        metadata={
            "application": "omc_csp",
            "pool_notation": "P denotes specific unique CIF candidates",
            "family_scope": scope.lower(),
            "source_families": sorted({record.family for record in records}),
            "selection_seed": inputs.selection_seed,
            "selection": selection,
            "replication": False,
            "parent_pool_id": parent_pool_id,
            "active_edge_key": active_key,
            "candidate_edge_key": candidate_key,
            "cutoff_A": inputs.cutoff_A,
            "skin_A": inputs.skin_A,
            "cost_ratio_max": inputs.cost_ratio_max,
            # Frozen v1 key retained for construction-hash compatibility.
            "planning_cost_formula": (
                _LEGACY_OMC_BFGS_COST_PROXY_FORMULA
            ),
            "statistics": _pool_statistics(records),
        },
    ).seal()


def _balanced_round_robin(
    records_by_family: dict[str, Sequence[OMCCSPCandidate]],
    *,
    count: int,
) -> list[OMCCSPCandidate]:
    families = sorted(records_by_family)
    cursors = {family: 0 for family in families}
    selected: list[OMCCSPCandidate] = []
    seen: set[str] = set()
    while len(selected) < count:
        progressed = False
        for family in families:
            records = records_by_family[family]
            cursor = cursors[family]
            while (
                cursor < len(records)
                and records[cursor].normalized_structure_sha256 in seen
            ):
                cursor += 1
            cursors[family] = cursor
            if cursor >= len(records):
                continue
            record = records[cursor]
            cursors[family] += 1
            seen.add(record.normalized_structure_sha256)
            selected.append(record)
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(
                f"cross-family selection provides fewer than {count} unique candidates"
            )
    return selected


def _cost_buckets(
    records: Iterable[OMCCSPCandidate],
    *,
    ratio_max: float,
) -> list[list[OMCCSPCandidate]]:
    ordered = sorted(
        records,
        key=lambda record: (
            -record.omc_variable_cell_bfgs_cost_proxy,
            record.source_path,
        ),
    )
    groups: list[list[OMCCSPCandidate]] = []
    current: list[OMCCSPCandidate] = []
    largest = 0.0
    for record in ordered:
        if (
            current
            and largest / record.omc_variable_cell_bfgs_cost_proxy
            > ratio_max
        ):
            groups.append(current)
            current = []
        if not current:
            largest = record.omc_variable_cell_bfgs_cost_proxy
        current.append(record)
    if current:
        groups.append(current)
    return groups


def build_omc_csp_workloads(
    inputs: OMCCSPWorkloadInputs,
    *,
    workers: int = 1,
) -> tuple[dict[str, WorkloadManifest], dict[str, Any]]:
    """Build unique nested family and cross-family OMC-CSP pools."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    selected_by_family: dict[str, list[OMCCSPCandidate]] = {}
    family_catalog: dict[str, Any] = {}
    reference_paths: list[str] = []
    empty_families: list[str] = []
    for family_dir in sorted(
        path for path in inputs.dataset_dir.iterdir() if path.is_dir()
    ):
        candidate_paths = list(family_dir.rglob("*.cif"))
        if not candidate_paths:
            empty_families.append(family_dir.name)
            family_catalog[family_dir.name] = {
                "candidate_files": 0,
                "reference_files": 0,
                "selected_unique_candidates": 0,
            }
            continue
        selected, references, candidate_count = _select_unique_family(
            inputs,
            family_dir,
            workers=workers,
        )
        selected_by_family[family_dir.name] = selected
        reference_paths.extend(references)
        family_catalog[family_dir.name] = {
            "candidate_files": candidate_count,
            "reference_files": len(references),
            "selected_unique_candidates": len(selected),
            "selected_statistics": _pool_statistics(selected),
        }

    if not selected_by_family:
        raise ValueError("dataset contains no nonempty CSP candidate families")

    workloads: dict[str, WorkloadManifest] = {}

    def add_nested(
        records: Sequence[OMCCSPCandidate],
        *,
        scope: str,
        label: str,
        selection: str,
    ) -> None:
        parent_id = None
        manifests: dict[int, WorkloadManifest] = {}
        for size in reversed(inputs.pool_sizes):
            pool = list(records[:size])
            cost_class = _cost_class(pool, inputs.cost_ratio_max)
            workload_id = _workload_id(scope, label, size, cost_class)
            manifests[size] = _manifest(
                inputs,
                pool,
                workload_id=workload_id,
                scope=scope,
                selection=selection,
                parent_pool_id=parent_id,
            )
            parent_id = workload_id
        workloads.update(
            (manifest.workload_id, manifest)
            for _, manifest in sorted(manifests.items())
        )

    for family, records in selected_by_family.items():
        add_nested(
            records,
            scope="INTRA",
            label=family,
            selection=(
                "seeded SHA-256 relative-path rank; first unique normalized "
                "structures; smaller pool is ordered prefix"
            ),
        )

    maximum_pool = max(inputs.pool_sizes)
    inter_wide = _balanced_round_robin(
        selected_by_family,
        count=maximum_pool,
    )
    add_nested(
        inter_wide,
        scope="INTER",
        label="MIX-ALL",
        selection=(
            "balanced family round-robin over frozen intra-family candidates; "
            "smaller pool is ordered prefix"
        ),
    )

    all_selected = [
        record
        for records in selected_by_family.values()
        for record in records
    ]
    eligible_bands = []
    for bucket in _cost_buckets(
        all_selected,
        ratio_max=inputs.cost_ratio_max,
    ):
        by_family: dict[str, list[OMCCSPCandidate]] = defaultdict(list)
        for record in bucket:
            by_family[record.family].append(record)
        if len(bucket) < maximum_pool or len(by_family) < 2:
            continue
        eligible_bands.append((bucket, dict(by_family)))
    for band_index, (_, by_family) in enumerate(eligible_bands, start=1):
        records = _balanced_round_robin(
            by_family,
            count=maximum_pool,
        )
        add_nested(
            records,
            scope="INTER",
            label=f"COSTBAND-{band_index:02d}",
            selection=(
                "scheduler-compatible cost band followed by balanced family "
                "round-robin; smaller pool is ordered prefix"
            ),
        )

    catalog = {
        "schema_version": 1,
        "construction_id": "omc-csp-unique-pools-v1",
        "dataset_id": inputs.dataset_id,
        "dataset_root": str(inputs.dataset_dir),
        "selection_seed": inputs.selection_seed,
        "pool_sizes": list(inputs.pool_sizes),
        "cutoff_A": inputs.cutoff_A,
        "skin_A": inputs.skin_A,
        "cost_ratio_max": inputs.cost_ratio_max,
        # Frozen v1 key retained for construction-hash compatibility.
        "planning_cost_formula": _LEGACY_OMC_BFGS_COST_PROXY_FORMULA,
        "replication": False,
        "family_count": len(family_catalog),
        "nonempty_family_count": len(selected_by_family),
        "empty_families": empty_families,
        "experimental_reference_paths": sorted(reference_paths),
        "families": family_catalog,
        "workload_count": len(workloads),
        "workloads": {
            workload_id: {
                "jobs": len(manifest.jobs),
                "unique_structures": len(
                    {
                        job.normalized_structure_sha256
                        for job in manifest.jobs
                    }
                ),
                "manifest_sha256": manifest.manifest_sha256,
                "family_scope": manifest.metadata["family_scope"],
                "source_families": manifest.metadata["source_families"],
                "parent_pool_id": manifest.metadata["parent_pool_id"],
                "statistics": manifest.metadata["statistics"],
            }
            for workload_id, manifest in sorted(workloads.items())
        },
    }
    catalog["construction_sha256"] = _canonical_sha256(catalog)
    return workloads, catalog


def write_omc_csp_workloads(
    output_dir: str | Path,
    workloads: dict[str, WorkloadManifest],
    catalog: dict[str, Any],
    *,
    write_csv: bool = True,
) -> None:
    """Write signed manifests, task profiles, and the construction index."""

    output = Path(output_dir)
    manifests_dir = output / "manifests"
    profiles_dir = output / "profiles"
    rows = []
    for workload_id, manifest in sorted(workloads.items()):
        write_workload_manifest(
            manifests_dir / f"{workload_id}.json",
            manifest,
        )
        if write_csv:
            write_workload_jobs_csv(
                manifests_dir / f"{workload_id}.csv",
                manifest,
            )
        profile = TaskProfile.from_manifest(
            manifest,
            active_edge_key=manifest.metadata["active_edge_key"],
            candidate_edge_key=manifest.metadata["candidate_edge_key"],
        )
        profiles_dir.mkdir(parents=True, exist_ok=True)
        (profiles_dir / f"{workload_id}.json").write_text(
            json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        statistics = manifest.metadata["statistics"]
        rows.append(
            {
                "workload_id": workload_id,
                "pool_size": len(manifest.jobs),
                "family_scope": manifest.metadata["family_scope"],
                "source_family_count": len(manifest.metadata["source_families"]),
                "atom_count_min": statistics["atom_count_min"],
                "atom_count_max": statistics["atom_count_max"],
                "planning_cost_ratio": statistics["planning_cost_ratio"],
                "planning_cost_cv": statistics["planning_cost_cv"],
                "manifest_sha256": manifest.manifest_sha256,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_omc_csp_workload_directory(
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate hashes, uniqueness, nesting, and family coverage on disk."""

    output = Path(output_dir)
    catalog = json.loads((output / "index.json").read_text(encoding="utf-8"))
    expected_construction_hash = catalog.pop("construction_sha256")
    if _canonical_sha256(catalog) != expected_construction_hash:
        raise ValueError("construction index hash mismatch")
    catalog["construction_sha256"] = expected_construction_hash

    manifests: dict[str, WorkloadManifest] = {}
    for workload_id, entry in catalog["workloads"].items():
        manifest = read_workload_manifest(
            output / "manifests" / f"{workload_id}.json"
        )
        manifest.verify()
        if manifest.manifest_sha256 != entry["manifest_sha256"]:
            raise ValueError(f"{workload_id}: catalog manifest hash mismatch")
        if len(manifest.jobs) != entry["jobs"]:
            raise ValueError(f"{workload_id}: catalog job count mismatch")
        paths = [job.source_path for job in manifest.jobs]
        structures = [
            job.normalized_structure_sha256 for job in manifest.jobs
        ]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{workload_id}: repeated source path")
        if len(structures) != len(set(structures)):
            raise ValueError(f"{workload_id}: repeated normalized structure")
        if any("_exp_" in path.lower() for path in paths):
            raise ValueError(f"{workload_id}: experimental reference included")
        manifests[workload_id] = manifest

    for workload_id, manifest in manifests.items():
        parent_id = manifest.metadata["parent_pool_id"]
        if parent_id is None:
            continue
        parent = manifests[parent_id]
        paths = [job.source_path for job in manifest.jobs]
        parent_prefix = [
            job.source_path for job in parent.jobs[: len(manifest.jobs)]
        ]
        if paths != parent_prefix:
            raise ValueError(f"{workload_id}: not an ordered parent prefix")

    expected_sizes = set(catalog["pool_sizes"])
    intra_sizes: dict[str, set[int]] = defaultdict(set)
    for manifest in manifests.values():
        if manifest.metadata["family_scope"] != "intra":
            continue
        source_families = manifest.metadata["source_families"]
        if len(source_families) != 1:
            raise ValueError(
                f"{manifest.workload_id}: intra pool spans multiple families"
            )
        intra_sizes[source_families[0]].add(len(manifest.jobs))
    expected_families = {
        family
        for family, entry in catalog["families"].items()
        if entry["candidate_files"] > 0
    }
    if set(intra_sizes) != expected_families:
        raise ValueError("intra-family workload coverage mismatch")
    if any(sizes != expected_sizes for sizes in intra_sizes.values()):
        raise ValueError("intra-family pool-size coverage mismatch")

    return {
        "construction_sha256": expected_construction_hash,
        "family_count": catalog["nonempty_family_count"],
        "workload_count": len(manifests),
        "validated_job_references": sum(
            len(manifest.jobs) for manifest in manifests.values()
        ),
        "pool_sizes": catalog["pool_sizes"],
        "replication": catalog["replication"],
    }
