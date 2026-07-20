"""Materialize signed workload jobs as input-ordered ASE structures."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ase import Atoms
from ase.io import read

from .generator import normalized_structure_sha256
from .schema import WorkloadManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_workload(
    manifest: WorkloadManifest,
    dataset_dir: str | Path,
) -> list[Atoms]:
    """Load and verify every unique source, then reproduce manifest job order."""

    manifest.verify()
    root = Path(dataset_dir)
    source_hashes: dict[str, str] = {}
    frames: dict[tuple[str, int], tuple[Atoms, str, tuple[str, ...]]] = {}
    output: list[Atoms] = []
    for job in manifest.jobs:
        source = root / job.source_path
        if job.source_path not in source_hashes:
            if not source.is_file():
                raise FileNotFoundError(source)
            source_sha256 = _sha256_file(source)
            if source_sha256 != job.source_sha256:
                raise ValueError(f"source hash differs for {job.source_path}")
            source_hashes[job.source_path] = source_sha256
        elif source_hashes[job.source_path] != job.source_sha256:
            raise ValueError(f"replicated source hashes differ for {job.system_id}")

        frame_key = (job.source_path, job.frame_index)
        if frame_key not in frames:
            atoms = read(source, index=job.frame_index)
            if not isinstance(atoms, Atoms):
                raise TypeError(f"source frame is not an Atoms object: {job.source_path}")
            normalized_sha256 = normalized_structure_sha256(atoms)
            if normalized_sha256 != job.normalized_structure_sha256:
                raise ValueError(f"normalized structure differs for {job.source_path}")
            species = tuple(atoms.get_chemical_symbols())
            if species != job.species:
                raise ValueError(f"species ordering differs for {job.source_path}")
            frames[frame_key] = (atoms, normalized_sha256, species)
        source_atoms, normalized_sha256, species = frames[frame_key]
        if job.normalized_structure_sha256 != normalized_sha256 or job.species != species:
            raise ValueError(f"replicated job descriptors differ for {job.system_id}")
        atoms = source_atoms.copy()
        atoms.info["workload_id"] = manifest.workload_id
        atoms.info["workload_system_id"] = job.system_id
        atoms.info["workload_order"] = job.order
        output.append(atoms)
    return output
