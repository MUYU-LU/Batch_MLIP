"""Versioned schemas for immutable atomistic workload manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mean(values: list[int | float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _coefficient_of_variation(values: list[int | float]) -> float | None:
    mean = _mean(values)
    if mean is None or mean == 0.0:
        return None
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


@dataclass(frozen=True)
class WorkloadJob:
    """One immutable job identity and its structure-derived descriptors."""

    system_id: str
    group_id: str
    duplicate_group: str
    order: int
    dataset_id: str
    source_path: str
    source_sha256: str
    normalized_structure_sha256: str
    frame_index: int
    atom_count: int
    species: tuple[str, ...]
    chemical_formula: str
    pbc: tuple[bool, bool, bool]
    cell_A: tuple[float, ...]
    volume_A3: float
    constraints: tuple[str, ...]
    topology_edge_counts: dict[str, int]
    arrival_time: float = 0.0
    random_seed: int | None = None
    reference: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.system_id or not self.group_id or not self.duplicate_group:
            raise ValueError("job identifiers must not be empty")
        if self.order < 0 or self.frame_index < 0 or self.atom_count <= 0:
            raise ValueError("job order, frame index, and atom count are invalid")
        if len(self.pbc) != 3 or len(self.cell_A) != 9:
            raise ValueError("pbc and flattened cell must have lengths 3 and 9")
        if len(self.species) != self.atom_count:
            raise ValueError("species must preserve the per-atom chemical ordering")
        if not math.isfinite(self.volume_A3) or self.volume_A3 < 0.0:
            raise ValueError("job volume must be finite and non-negative")
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0.0:
            raise ValueError("job arrival time must be finite and non-negative")
        if any(count < 0 for count in self.topology_edge_counts.values()):
            raise ValueError("topology edge counts must be non-negative")
        for name, value in (
            ("source_sha256", self.source_sha256),
            ("normalized_structure_sha256", self.normalized_structure_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkloadJob:
        values = dict(payload)
        values["pbc"] = tuple(values["pbc"])
        values["cell_A"] = tuple(values["cell_A"])
        values["species"] = tuple(values["species"])
        values["constraints"] = tuple(values.get("constraints", ()))
        return cls(**values)


@dataclass(frozen=True)
class WorkloadManifest:
    """Self-verifying ordered collection of atomistic jobs."""

    workload_id: str
    version: int
    family: str
    operation: str
    cell_mode: str
    arrival_mode: str
    jobs: tuple[WorkloadJob, ...]
    metadata: dict[str, Any]
    manifest_sha256: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported workload manifest schema")
        if not self.workload_id or self.version <= 0 or not self.jobs:
            raise ValueError("workload identity, version, and jobs are required")
        if self.cell_mode not in {"fixed", "variable"}:
            raise ValueError("cell_mode must be 'fixed' or 'variable'")
        if self.arrival_mode not in {"closed", "streaming", "open"}:
            raise ValueError("unsupported arrival_mode")
        identifiers = [job.system_id for job in self.jobs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("system IDs must be unique")
        if [job.order for job in self.jobs] != list(range(len(self.jobs))):
            raise ValueError("job order must be contiguous and match tuple order")

    def unsigned_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("manifest_sha256")
        return payload

    def calculate_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.unsigned_dict())).hexdigest()

    def seal(self) -> WorkloadManifest:
        return replace(self, manifest_sha256=self.calculate_sha256())

    def verify(self) -> None:
        if not self.manifest_sha256:
            raise ValueError("workload manifest is not sealed")
        if self.manifest_sha256 != self.calculate_sha256():
            raise ValueError("workload manifest content hash does not match")

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["manifest_sha256"] = self.manifest_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, verify: bool = True) -> WorkloadManifest:
        values = dict(payload)
        values["jobs"] = tuple(WorkloadJob.from_dict(item) for item in values["jobs"])
        manifest = cls(**values)
        if verify:
            manifest.verify()
        return manifest


@dataclass(frozen=True)
class TaskProfile:
    """Compact planner input derived from one frozen workload manifest."""

    workload_id: str
    workload_manifest_sha256: str
    family: str
    operation: str
    pool_size: int
    arrival_mode: str
    cell_mode: str
    unique_structure_count: int
    duplicate_job_count: int
    atom_count_min: int
    atom_count_max: int
    atom_count_mean: float
    atom_count_cv: float
    total_atoms: int
    active_edge_key: str | None
    active_edges_min: int | None
    active_edges_max: int | None
    active_edges_mean: float | None
    active_edges_cv: float | None
    total_active_edges: int | None
    candidate_edge_key: str | None
    candidate_edges_min: int | None
    candidate_edges_max: int | None
    candidate_edges_mean: float | None
    candidate_edges_cv: float | None
    total_candidate_edges: int | None
    candidate_to_active_ratio: float | None
    reference_step_mean: float | None
    reference_step_cv: float | None
    schema_version: int = 1

    @classmethod
    def from_manifest(
        cls,
        manifest: WorkloadManifest,
        *,
        active_edge_key: str | None = None,
        candidate_edge_key: str | None = None,
        reference_model: str | None = None,
    ) -> TaskProfile:
        manifest.verify()
        atom_counts = [job.atom_count for job in manifest.jobs]

        def edge_values(key: str | None) -> list[int]:
            if key is None:
                return []
            values = []
            for job in manifest.jobs:
                if key not in job.topology_edge_counts:
                    raise KeyError(f"job {job.system_id} has no edge profile {key!r}")
                values.append(job.topology_edge_counts[key])
            return values

        active_edges = edge_values(active_edge_key)
        candidate_edges = edge_values(candidate_edge_key)
        steps = []
        if reference_model is not None:
            for job in manifest.jobs:
                reference = (job.reference or {}).get(reference_model)
                if reference is None or "steps" not in reference:
                    raise KeyError(f"job {job.system_id} has no {reference_model!r} step reference")
                steps.append(float(reference["steps"]))
        step_mean = _mean(steps)
        step_cv = None
        if step_mean is not None and step_mean > 0.0:
            variance = sum((value - step_mean) ** 2 for value in steps) / len(steps)
            step_cv = math.sqrt(variance) / step_mean
        return cls(
            workload_id=manifest.workload_id,
            workload_manifest_sha256=manifest.manifest_sha256,
            family=manifest.family,
            operation=manifest.operation,
            pool_size=len(manifest.jobs),
            arrival_mode=manifest.arrival_mode,
            cell_mode=manifest.cell_mode,
            unique_structure_count=len({job.normalized_structure_sha256 for job in manifest.jobs}),
            duplicate_job_count=len(manifest.jobs)
            - len({job.normalized_structure_sha256 for job in manifest.jobs}),
            atom_count_min=min(atom_counts),
            atom_count_max=max(atom_counts),
            atom_count_mean=float(sum(atom_counts) / len(atom_counts)),
            atom_count_cv=_coefficient_of_variation(atom_counts) or 0.0,
            total_atoms=sum(atom_counts),
            active_edge_key=active_edge_key,
            active_edges_min=min(active_edges) if active_edges else None,
            active_edges_max=max(active_edges) if active_edges else None,
            active_edges_mean=_mean(active_edges),
            active_edges_cv=_coefficient_of_variation(active_edges),
            total_active_edges=sum(active_edges) if active_edges else None,
            candidate_edge_key=candidate_edge_key,
            candidate_edges_min=min(candidate_edges) if candidate_edges else None,
            candidate_edges_max=max(candidate_edges) if candidate_edges else None,
            candidate_edges_mean=_mean(candidate_edges),
            candidate_edges_cv=_coefficient_of_variation(candidate_edges),
            total_candidate_edges=sum(candidate_edges) if candidate_edges else None,
            candidate_to_active_ratio=(
                None
                if not active_edges or not candidate_edges or sum(active_edges) == 0
                else sum(candidate_edges) / sum(active_edges)
            ),
            reference_step_mean=step_mean,
            reference_step_cv=step_cv,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskProfile:
        return cls(**payload)


def write_workload_manifest(path: str | Path, manifest: WorkloadManifest) -> None:
    manifest.verify()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_workload_manifest(path: str | Path) -> WorkloadManifest:
    return WorkloadManifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_workload_jobs_csv(path: str | Path, manifest: WorkloadManifest) -> None:
    """Write a flat, human-auditable projection of a signed JSON manifest."""

    manifest.verify()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "workload_id",
        "manifest_sha256",
        "system_id",
        "group_id",
        "duplicate_group",
        "order",
        "dataset_id",
        "source_path",
        "source_sha256",
        "normalized_structure_sha256",
        "frame_index",
        "atom_count",
        "species",
        "chemical_formula",
        "pbc",
        "cell_A",
        "volume_A3",
        "constraints",
        "topology_edge_counts",
        "arrival_time",
        "random_seed",
        "reference",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for job in manifest.jobs:
            values = job.to_dict()
            writer.writerow(
                {
                    "workload_id": manifest.workload_id,
                    "manifest_sha256": manifest.manifest_sha256,
                    **{
                        name: (
                            json.dumps(values[name], sort_keys=True)
                            if isinstance(values.get(name), (dict, list, tuple))
                            else values.get(name)
                        )
                        for name in fields
                        if name not in {"workload_id", "manifest_sha256"}
                    },
                }
            )
