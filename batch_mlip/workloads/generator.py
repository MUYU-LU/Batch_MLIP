"""Deterministic builders for the controlled T2 workload suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from ..core.neighbors import neighbor_list
from .schema import WorkloadJob, WorkloadManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_structure_sha256(atoms: Atoms) -> str:
    """Hash only structure-defining values in a platform-stable representation."""

    digest = hashlib.sha256()
    arrays = (
        np.asarray(atoms.numbers, dtype="<i8"),
        np.asarray(atoms.positions, dtype="<f8"),
        np.asarray(atoms.cell.array, dtype="<f8"),
        np.asarray(atoms.pbc, dtype="u1"),
    )
    for values in arrays:
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def topology_key(cutoff: float, skin: float) -> str:
    if cutoff <= 0.0 or skin < 0.0:
        raise ValueError("cutoff must be positive and skin non-negative")
    return f"cutoff={cutoff:.3f}_skin={skin:.3f}"


@dataclass(frozen=True)
class T2WorkloadInputs:
    dataset_dir: Path
    selection_manifest: Path
    dataset_id: str = "T2_test"
    cutoffs_A: tuple[float, ...] = (4.5, 6.0)
    skins_A: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class RobustnessWorkloadInputs:
    """Inputs for deterministic cross-family robustness workloads."""

    dataset_dir: Path
    dataset_id: str = "cross_family_test_set"
    seed: int = 20260725
    candidate_count: int = 256
    unique_structures: int = 32
    pool_size: int = 256
    cutoffs_A: tuple[float, ...] = (6.0,)
    skins_A: tuple[float, ...] = (0.0, 0.5)


@dataclass(frozen=True)
class _StructureRecord:
    source_path: str
    source_sha256: str
    normalized_sha256: str
    atoms: Atoms
    topology_edge_counts: dict[str, int]


def _structure_record(
    path: Path,
    *,
    relative_path: str,
    cutoffs: tuple[float, ...],
    skins: tuple[float, ...],
) -> _StructureRecord:
    atoms = read(path)
    topology = {
        topology_key(cutoff, skin): int(len(neighbor_list("i", atoms, cutoff + skin)))
        for cutoff in cutoffs
        for skin in skins
    }
    return _StructureRecord(
        source_path=relative_path,
        source_sha256=_sha256_file(path),
        normalized_sha256=normalized_structure_sha256(atoms),
        atoms=atoms,
        topology_edge_counts=topology,
    )


def _constraint_names(atoms: Atoms) -> tuple[str, ...]:
    return tuple(type(constraint).__name__ for constraint in atoms.constraints)


def _job(
    record: _StructureRecord,
    *,
    workload_id: str,
    dataset_id: str,
    order: int,
    reference: dict[str, Any] | None = None,
    random_seed: int | None = None,
) -> WorkloadJob:
    atoms = record.atoms
    return WorkloadJob(
        system_id=f"{workload_id}:{order:04d}",
        group_id=Path(record.source_path).stem,
        duplicate_group=record.normalized_sha256,
        order=order,
        dataset_id=dataset_id,
        source_path=record.source_path,
        source_sha256=record.source_sha256,
        normalized_structure_sha256=record.normalized_sha256,
        frame_index=0,
        atom_count=len(atoms),
        species=tuple(atoms.get_chemical_symbols()),
        chemical_formula=atoms.get_chemical_formula(mode="hill"),
        pbc=tuple(bool(value) for value in atoms.pbc),
        cell_A=tuple(float(value) for value in atoms.cell.array.reshape(-1)),
        volume_A3=float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0,
        constraints=_constraint_names(atoms),
        topology_edge_counts=dict(record.topology_edge_counts),
        random_seed=random_seed,
        reference=reference,
    )


def _manifest(
    workload_id: str,
    records: list[_StructureRecord],
    *,
    inputs: T2WorkloadInputs | RobustnessWorkloadInputs,
    family: str = "variable_horizon_closed",
    operation: str = "optimization",
    cell_mode: str = "variable",
    metadata: dict[str, Any],
    references: dict[str, dict[str, Any]] | None = None,
    random_seeds: list[int | None] | None = None,
) -> WorkloadManifest:
    if random_seeds is not None and len(random_seeds) != len(records):
        raise ValueError("random_seeds must contain one value per workload job")
    jobs = tuple(
        _job(
            record,
            workload_id=workload_id,
            dataset_id=inputs.dataset_id,
            order=index,
            reference=(references or {}).get(record.source_path),
            random_seed=None if random_seeds is None else random_seeds[index],
        )
        for index, record in enumerate(records)
    )
    return WorkloadManifest(
        workload_id=workload_id,
        version=1,
        family=family,
        operation=operation,
        cell_mode=cell_mode,
        arrival_mode="closed",
        jobs=jobs,
        metadata=metadata,
    ).seal()


def _deterministic_candidates(
    paths: list[Path],
    *,
    count: int,
    seed: int,
    scope: str,
) -> list[Path]:
    if count <= 0:
        raise ValueError("candidate count must be positive")
    if len(paths) < count:
        raise ValueError(
            f"{scope} provides {len(paths)} structures, fewer than requested {count}"
        )

    def rank(path: Path) -> bytes:
        payload = f"{seed}:{scope}:{path.as_posix()}".encode()
        return hashlib.sha256(payload).digest()

    return sorted(paths, key=lambda path: (rank(path), path.as_posix()))[:count]


def _density_stratified_records(
    records: list[_StructureRecord],
    *,
    count: int,
    edge_key: str,
) -> list[_StructureRecord]:
    if count <= 0 or len(records) < count:
        raise ValueError("density stratum count is invalid")
    ranked = sorted(
        records,
        key=lambda record: (
            record.topology_edge_counts[edge_key] / len(record.atoms),
            record.source_path,
        ),
    )
    indices = np.linspace(0, len(ranked) - 1, count, dtype=int)
    return [ranked[int(index)] for index in indices]


def _repeat_records(
    records: list[_StructureRecord],
    *,
    pool_size: int,
) -> list[_StructureRecord]:
    if not records or pool_size <= 0 or pool_size % len(records):
        raise ValueError("pool size must be a positive multiple of unique structures")
    return [records[index % len(records)] for index in range(pool_size)]


def _robustness_records(
    inputs: RobustnessWorkloadInputs,
    *,
    family: str,
    expected_atom_counts: set[int],
    balanced_atom_counts: bool = False,
    filter_mixed_atom_counts: bool = False,
) -> list[_StructureRecord]:
    family_dir = inputs.dataset_dir / family
    paths = sorted(
        path
        for path in family_dir.rglob("*.cif")
        if "_exp_" not in path.name.lower()
    )
    if not paths:
        raise FileNotFoundError(f"no CIF structures found under {family_dir}")
    edge_key = topology_key(inputs.cutoffs_A[0], inputs.skins_A[0])

    def load(selected_paths: list[Path]) -> list[_StructureRecord]:
        records = [
            _structure_record(
                path,
                relative_path=path.relative_to(inputs.dataset_dir).as_posix(),
                cutoffs=inputs.cutoffs_A,
                skins=inputs.skins_A,
            )
            for path in selected_paths
        ]
        unexpected = sorted(
            {len(record.atoms) for record in records} - expected_atom_counts
        )
        if unexpected:
            raise ValueError(
                f"{family} contains unexpected atom counts in selection: {unexpected}"
            )
        return records

    if not balanced_atom_counts:
        if filter_mixed_atom_counts:
            records = []
            ranked_paths = _deterministic_candidates(
                paths,
                count=len(paths),
                seed=inputs.seed,
                scope=family,
            )
            for path in ranked_paths:
                record = _structure_record(
                    path,
                    relative_path=path.relative_to(inputs.dataset_dir).as_posix(),
                    cutoffs=inputs.cutoffs_A,
                    skins=inputs.skins_A,
                )
                if len(record.atoms) not in expected_atom_counts:
                    continue
                records.append(record)
                if len(records) == inputs.candidate_count:
                    break
            if len(records) < inputs.candidate_count:
                raise ValueError(
                    f"{family} provides {len(records)} matching structures, "
                    f"fewer than requested {inputs.candidate_count}"
                )
            return _density_stratified_records(
                records,
                count=inputs.unique_structures,
                edge_key=edge_key,
            )
        candidates = _deterministic_candidates(
            paths,
            count=inputs.candidate_count,
            seed=inputs.seed,
            scope=family,
        )
        return _density_stratified_records(
            load(candidates),
            count=inputs.unique_structures,
            edge_key=edge_key,
        )

    per_stratum = inputs.unique_structures // len(expected_atom_counts)
    candidate_per_stratum = inputs.candidate_count // len(expected_atom_counts)
    if (
        per_stratum * len(expected_atom_counts) != inputs.unique_structures
        or candidate_per_stratum * len(expected_atom_counts)
        != inputs.candidate_count
    ):
        raise ValueError(
            "candidate and unique counts must divide balanced atom-count strata"
        )
    selected: list[_StructureRecord] = []
    for atom_count in sorted(expected_atom_counts):
        suffix = f"_{atom_count}.cif"
        stratum_paths = [path for path in paths if path.name.endswith(suffix)]
        candidates = _deterministic_candidates(
            stratum_paths,
            count=candidate_per_stratum,
            seed=inputs.seed,
            scope=f"{family}-N{atom_count}",
        )
        selected.extend(
            _density_stratified_records(
                load(candidates),
                count=per_stratum,
                edge_key=edge_key,
            )
        )
    return selected


def build_robustness_family_workload(
    inputs: RobustnessWorkloadInputs,
    *,
    label: str,
    family: str,
    expected_atom_counts: set[int],
    balanced_atom_counts: bool = False,
) -> WorkloadManifest:
    """Build one signed R256-style family workload with the suite protocol."""

    if not label or not family or not expected_atom_counts:
        raise ValueError("label, family, and expected_atom_counts are required")
    if inputs.pool_size % inputs.unique_structures:
        raise ValueError("pool size must divide evenly by unique structures")
    selected = _robustness_records(
        inputs,
        family=family,
        expected_atom_counts=expected_atom_counts,
        balanced_atom_counts=balanced_atom_counts,
        filter_mixed_atom_counts=True,
    )
    workload_id = f"OPT-RB-{label}-R{inputs.pool_size}-v1"
    return _manifest(
        workload_id,
        _repeat_records(selected, pool_size=inputs.pool_size),
        inputs=inputs,
        metadata={
            "source_family": family,
            "selection_seed": inputs.seed,
            "candidate_count": inputs.candidate_count,
            "unique_structures": inputs.unique_structures,
            "repetitions": inputs.pool_size // inputs.unique_structures,
            "selection": (
                "balanced by atom count then uniform across 6 A edge-density rank"
                if balanced_atom_counts
                else "uniform across 6 A edge-density rank"
            ),
            "atom_count_strata": sorted(expected_atom_counts),
            "active_edge_key": topology_key(
                inputs.cutoffs_A[0],
                inputs.skins_A[0],
            ),
            "cutoffs_A": list(inputs.cutoffs_A),
            "skins_A": list(inputs.skins_A),
            "claim_role": "positive_control",
        },
    )


def build_robustness_workloads(
    inputs: RobustnessWorkloadInputs,
) -> dict[str, WorkloadManifest]:
    """Build fixed cross-family pools spanning chemistry, size, and density."""

    if inputs.pool_size % inputs.unique_structures:
        raise ValueError("pool size must divide evenly by unique structures")
    specifications = (
        ("GUFJOG44", "GUFJOG", {44}, False),
        ("SOXLEX48", "SOXLEX", {48}, False),
        ("XATMOV88", "XATMOV", {88}, False),
        ("OBEQIX220", "OBEQIX", {220}, False),
        ("ROFA-MIX", "rof-a", {74, 148, 222, 296}, True),
        ("ROFB296", "rof-b", {296}, False),
    )
    edge_key = topology_key(inputs.cutoffs_A[0], inputs.skins_A[0])
    workloads: dict[str, WorkloadManifest] = {}
    selected_by_name: dict[str, list[_StructureRecord]] = {}
    for label, family, atom_counts, balanced in specifications:
        selected = _robustness_records(
            inputs,
            family=family,
            expected_atom_counts=atom_counts,
            balanced_atom_counts=balanced,
        )
        selected_by_name[label] = selected
        workload_id = f"OPT-RB-{label}-R{inputs.pool_size}-v1"
        workloads[workload_id] = _manifest(
            workload_id,
            _repeat_records(selected, pool_size=inputs.pool_size),
            inputs=inputs,
            metadata={
                "source_family": family,
                "selection_seed": inputs.seed,
                "candidate_count": inputs.candidate_count,
                "unique_structures": inputs.unique_structures,
                "repetitions": inputs.pool_size // inputs.unique_structures,
                "selection": (
                    "balanced by atom count then uniform across 6 A edge-density rank"
                    if balanced
                    else "uniform across 6 A edge-density rank"
                ),
                "atom_count_strata": sorted(atom_counts),
                "active_edge_key": edge_key,
                "cutoffs_A": list(inputs.cutoffs_A),
                "skins_A": list(inputs.skins_A),
                "claim_role": "positive_control",
            },
        )

    mixed_unique = [
        records[index]
        for index in range(inputs.unique_structures)
        for records in selected_by_name.values()
        if index < len(records)
    ]
    mixed_pool_size = len(mixed_unique)
    mixed_id = f"OPT-RB-CROSS-MIX-R{mixed_pool_size}-v1"
    workloads[mixed_id] = _manifest(
        mixed_id,
        mixed_unique,
        inputs=inputs,
        metadata={
            "source_families": [item[1] for item in specifications],
            "selection_seed": inputs.seed,
            "unique_structures": mixed_pool_size,
            "selection": "round-robin interleaving of positive-control families",
            "active_edge_key": edge_key,
            "claim_role": "heterogeneous_positive_control",
        },
    )
    return workloads


def _load_reference(
    path: Path,
    *,
    model_key: str,
    model_artifact: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise ValueError(f"reference {path} must contain exactly one point")
    records = points[0].get("records")
    if not isinstance(records, list) or len(records) < 4:
        raise ValueError(f"reference {path} has insufficient records")
    embedded_checkpoint = data.get("checkpoint", {})
    model_sha256 = embedded_checkpoint.get("sha256")
    if model_artifact is not None:
        if not model_artifact.is_file():
            raise FileNotFoundError(model_artifact)
        artifact_sha256 = _sha256_file(model_artifact)
        if model_sha256 is not None and model_sha256 != artifact_sha256:
            raise ValueError(f"model artifact hash differs from reference {path}")
        model_sha256 = artifact_sha256
    if model_sha256 is None:
        raise ValueError(f"reference {path} has no model hash; provide the exact model artifact")
    reference_artifact_sha256 = _sha256_file(path)
    references = {
        record["source"]: {
            model_key: {
                "steps": int(record["steps"]),
                "converged": bool(record["converged"]),
                "energy_eV": float(record["energy_eV"]),
                "reference_artifact_sha256": reference_artifact_sha256,
            }
        }
        for record in records
    }
    provenance = {
        "reference_artifact": str(path),
        "reference_artifact_sha256": reference_artifact_sha256,
        "model_checkpoint_sha256": model_sha256,
        "model_artifact_name": (
            model_artifact.name
            if model_artifact is not None
            else Path(embedded_checkpoint.get("path", "embedded-checkpoint")).name
        ),
        "model_name": data.get("mlip", model_key),
        "model_variant": data.get("model"),
        "model_dtype": data.get("parameters", {}).get("dtype"),
        "model_cutoff_A": data.get("parameters", {}).get("cutoff_A"),
        "optimizer": data.get("optimizer"),
        "reference_method": data.get("method"),
        "selection_manifest_sha256": data.get("manifest", {}).get("sha256"),
    }
    return references, provenance


def _stepvar_records(
    base: dict[str, _StructureRecord],
    reference: dict[str, dict[str, Any]],
    *,
    model_key: str,
) -> tuple[list[_StructureRecord], dict[str, dict[str, Any]], dict[str, Any]]:
    ranked = []
    for source, record in base.items():
        if source not in reference:
            raise KeyError(f"reference has no B1 result for {source}")
        result = reference[source][model_key]
        if not result["converged"]:
            raise ValueError(f"reference B1 job did not converge: {source}")
        ranked.append((result["steps"], source, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    quartile = len(ranked) // 4
    easy = ranked[:quartile]
    hard = ranked[-quartile:]
    selected = []
    for index in range(128):
        selected.append(easy[index % quartile][2])
        selected.append(hard[index % quartile][2])
    strata = {
        source: {**values, model_key: {**values[model_key], "stratum": stratum}}
        for stratum, group in (("easy", easy), ("hard", hard))
        for _, source, _ in group
        for values in (reference[source],)
    }
    return (
        selected,
        strata,
        {
            "reference_model": model_key,
            "easy_sources": [source for _, source, _ in easy],
            "hard_sources": [source for _, source, _ in hard],
            "selection": "lowest/highest B1 step-count quartiles; ties by source path",
        },
    )


def build_t2_workloads(
    inputs: T2WorkloadInputs,
    *,
    atombit_reference: Path | None = None,
    mace_reference: Path | None = None,
    reference_model_artifacts: dict[str, Path] | None = None,
) -> dict[str, WorkloadManifest]:
    """Build the initial controlled manifests from the fixed T2 selection."""

    selection = json.loads(
        inputs.selection_manifest.read_text(encoding="utf-8")
    )
    selected_names = {
        atom_count: selection["samples"][str(atom_count)][:32] for atom_count in (46, 276)
    }
    cache: dict[str, _StructureRecord] = {}
    for expected_atom_count, names in selected_names.items():
        for name in names:
            path = inputs.dataset_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            record = _structure_record(
                path,
                relative_path=name,
                cutoffs=inputs.cutoffs_A,
                skins=inputs.skins_A,
            )
            if len(record.atoms) != expected_atom_count:
                raise ValueError(
                    f"selection labels {name} as {expected_atom_count} atoms, "
                    f"but the structure contains {len(record.atoms)}"
                )
            cache[name] = record

    base_metadata = {
        "selection_manifest": str(inputs.selection_manifest),
        "selection_manifest_sha256": _sha256_file(inputs.selection_manifest),
        "cutoffs_A": list(inputs.cutoffs_A),
        "skins_A": list(inputs.skins_A),
        "statistics_role": "technical replicates grouped by duplicate_group",
    }
    small = [cache[name] for name in selected_names[46]]
    large = [cache[name] for name in selected_names[276]]
    workloads = {
        "OPT-H46-R256-v1": _manifest(
            "OPT-H46-R256-v1",
            [small[index % 32] for index in range(256)],
            inputs=inputs,
            metadata={**base_metadata, "unique_structures": 32, "repetitions": 8},
        ),
        "OPT-H276-R256-v1": _manifest(
            "OPT-H276-R256-v1",
            [large[index % 32] for index in range(256)],
            inputs=inputs,
            metadata={**base_metadata, "unique_structures": 32, "repetitions": 8},
        ),
        "OPT-H276-FIXED-R256-v1": _manifest(
            "OPT-H276-FIXED-R256-v1",
            [large[index % 32] for index in range(256)],
            inputs=inputs,
            cell_mode="fixed",
            metadata={
                **base_metadata,
                "unique_structures": 32,
                "repetitions": 8,
                "source_workload": "OPT-H276-R256-v1",
            },
        ),
        "OPT-MIX-R256-v1": _manifest(
            "OPT-MIX-R256-v1",
            [record for index in range(128) for record in (small[index % 32], large[index % 32])],
            inputs=inputs,
            metadata={
                **base_metadata,
                "unique_structures": 64,
                "atom_46_jobs": 128,
                "atom_276_jobs": 128,
            },
        ),
        "OPT-MIX-R32-v1": _manifest(
            "OPT-MIX-R32-v1",
            [record for index in range(16) for record in (small[index], large[index])],
            inputs=inputs,
            metadata={
                **base_metadata,
                "unique_structures": 32,
                "atom_46_jobs": 16,
                "atom_276_jobs": 16,
            },
        ),
    }
    fixed = _manifest(
        "OPT-FIXED50-v1",
        [large[index % 32] for index in range(256)],
        inputs=inputs,
        family="fixed_horizon_persistent",
        cell_mode="variable",
        metadata={
            **base_metadata,
            "source_workload": "OPT-H276-R256-v1",
            "steps": 50,
            "convergence_stopping": False,
            "status": "definition_frozen; fixed-step execution support pending",
        },
    )
    workloads[fixed.workload_id] = fixed

    pool_records = {
        ("H46", 32): small,
        ("H46", 256): [small[index % 32] for index in range(256)],
        ("H276", 32): large,
        ("H276", 256): [large[index % 32] for index in range(256)],
        ("MIX", 32): [record for index in range(16) for record in (small[index], large[index])],
        ("MIX", 256): [
            record for index in range(128) for record in (small[index % 32], large[index % 32])
        ],
    }
    for (distribution, pool_size), records in pool_records.items():
        common = {
            **base_metadata,
            "source_distribution": distribution,
            "pool_size": pool_size,
            "unique_structures": len({record.normalized_sha256 for record in records}),
        }
        evaluation_id = f"EVAL-{distribution}-R{pool_size}-v1"
        workloads[evaluation_id] = _manifest(
            evaluation_id,
            records,
            inputs=inputs,
            family="static_one_shot",
            operation="force_evaluation",
            cell_mode="fixed",
            metadata={
                **common,
                "requested_properties": ["energy", "forces"],
                "compute_stress": False,
            },
        )
        md_id = f"MD-NVE-{distribution}-R{pool_size}-v1"
        workloads[md_id] = _manifest(
            md_id,
            records,
            inputs=inputs,
            family="fixed_horizon_persistent",
            operation="md_nve",
            cell_mode="fixed",
            metadata={
                **common,
                "ensemble": "nve",
                "initial_temperature_K": 300.0,
                "remove_initial_com": True,
                "force_exact_initial_temperature": True,
                "timestep_fs": 0.5,
                "warmup_steps": 100,
                "measured_steps": 1000,
                "seed_rule": "2026072000 + output order",
            },
            random_seeds=[2026072000 + index for index in range(pool_size)],
        )

    for model_key, path in (
        ("atombit", atombit_reference),
        ("mace_off_small", mace_reference),
    ):
        if path is None:
            continue
        reference, provenance = _load_reference(
            path,
            model_key=model_key,
            model_artifact=(reference_model_artifacts or {}).get(model_key),
        )
        selected, strata, metadata = _stepvar_records(
            {name: cache[name] for name in selected_names[276]},
            reference,
            model_key=model_key,
        )
        workload_id = f"OPT-STEPVAR-{model_key.upper().replace('_', '-')}-R256-v1"
        workloads[workload_id] = _manifest(
            workload_id,
            selected,
            inputs=inputs,
            metadata={**base_metadata, **metadata, "reference_provenance": provenance},
            references=strata,
        )

    return workloads


def build_task_aware_holdout_workloads(
    inputs: T2WorkloadInputs,
    *,
    pool_sizes: tuple[int, ...] = (32, 64, 256),
) -> dict[str, WorkloadManifest]:
    """Build signed size-interpolation and four-size mixed validation pools."""

    if any(size <= 0 or size % 32 for size in pool_sizes):
        raise ValueError("holdout pool sizes must be positive multiples of 32")
    selection = json.loads(inputs.selection_manifest.read_text(encoding="utf-8"))
    atom_counts = (46, 92, 184, 276)
    selected_names = {
        atom_count: selection["samples"][str(atom_count)][:32]
        for atom_count in atom_counts
    }
    records_by_count: dict[int, list[_StructureRecord]] = {}
    for atom_count, names in selected_names.items():
        records = []
        for name in names:
            path = inputs.dataset_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            record = _structure_record(
                path,
                relative_path=name,
                cutoffs=inputs.cutoffs_A,
                skins=inputs.skins_A,
            )
            if len(record.atoms) != atom_count:
                raise ValueError(
                    f"selection labels {name} as {atom_count} atoms, "
                    f"but the structure contains {len(record.atoms)}"
                )
            records.append(record)
        records_by_count[atom_count] = records

    common = {
        "selection_manifest": str(inputs.selection_manifest),
        "selection_manifest_sha256": _sha256_file(inputs.selection_manifest),
        "cutoffs_A": list(inputs.cutoffs_A),
        "skins_A": list(inputs.skins_A),
        "claim_role": "task_aware_policy_holdout",
        "selection": "first 32 pre-frozen T2 samples in each atom-count stratum",
        "statistics_role": "technical replicates grouped by duplicate_group",
    }
    workloads: dict[str, WorkloadManifest] = {}
    for atom_count in (92, 184):
        unique = records_by_count[atom_count]
        for pool_size in pool_sizes:
            workload_id = f"OPT-H{atom_count}-R{pool_size}-v1"
            workloads[workload_id] = _manifest(
                workload_id,
                _repeat_records(unique, pool_size=pool_size),
                inputs=inputs,
                metadata={
                    **common,
                    "atom_count": atom_count,
                    "pool_size": pool_size,
                    "unique_structures": 32,
                    "repetitions": pool_size // 32,
                },
            )

    for pool_size in pool_sizes:
        per_stratum = pool_size // len(atom_counts)
        mixed = [
            records_by_count[atom_count][index % 32]
            for index in range(per_stratum)
            for atom_count in atom_counts
        ]
        workload_id = f"OPT-MIX4-R{pool_size}-v1"
        workloads[workload_id] = _manifest(
            workload_id,
            mixed,
            inputs=inputs,
            metadata={
                **common,
                "pool_size": pool_size,
                "atom_count_strata": list(atom_counts),
                "jobs_per_stratum": per_stratum,
                "unique_structures": len(
                    {record.normalized_sha256 for record in mixed}
                ),
                "selection": (
                    "round-robin H46/H92/H184/H276 from pre-frozen samples"
                ),
            },
        )
    return workloads
