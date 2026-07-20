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
        reference=reference,
    )


def _manifest(
    workload_id: str,
    records: list[_StructureRecord],
    *,
    inputs: T2WorkloadInputs,
    family: str = "variable_horizon_closed",
    operation: str = "optimization",
    cell_mode: str = "variable",
    metadata: dict[str, Any],
    references: dict[str, dict[str, Any]] | None = None,
) -> WorkloadManifest:
    jobs = tuple(
        _job(
            record,
            workload_id=workload_id,
            dataset_id=inputs.dataset_id,
            order=index,
            reference=(references or {}).get(record.source_path),
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

    selection = json.loads(inputs.selection_manifest.read_text(encoding="utf-8"))
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
