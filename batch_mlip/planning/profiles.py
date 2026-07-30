"""Layered cost profiles separating structures, models, tasks, and policies."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..workloads.schema import WorkloadManifest

HorizonKind = Literal["fixed", "variable", "streaming"]
OptimizerStateKind = Literal["none", "linear", "dense"]


def _qualified_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StructureCostProfile:
    """General structure feature independent of MLIP and numerical task."""

    index: int
    atom_count: int

    def __post_init__(self) -> None:
        if self.index < 0 or self.atom_count <= 0:
            raise ValueError("structure index and atom count are invalid")


@dataclass(frozen=True)
class MLIPGraphCostProfile:
    """MLIP-bound node/active-edge work for one structure."""

    index: int
    model_id: str
    cutoff_A: float
    active_edge_count: int
    force_mode: str
    model_dtype: str

    def __post_init__(self) -> None:
        if self.index < 0 or not self.model_id:
            raise ValueError("MLIP graph profile identity is invalid")
        if not math.isfinite(self.cutoff_A) or self.cutoff_A <= 0.0:
            raise ValueError("MLIP cutoff must be finite and positive")
        if self.active_edge_count < 0:
            raise ValueError("active edge count must be non-negative")
        if not self.force_mode or not self.model_dtype:
            raise ValueError("force mode and model dtype must not be empty")


@dataclass(frozen=True)
class TaskAuxiliaryCostProfile:
    """Numerical-method state and work beyond MLIP node/edge evaluation."""

    index: int
    operation: str
    algorithm: str
    variable_cell: bool
    stress_required: bool
    cell_method: str | None
    cell_degrees_of_freedom: int
    generalized_dimension: int
    optimizer_state_kind: OptimizerStateKind
    state_dtype: str
    linear_state_elements: int
    dense_state_elements: int
    dense_linear_algebra_work: int
    horizon_kind: HorizonKind

    def __post_init__(self) -> None:
        if self.index < 0 or not self.operation or not self.algorithm:
            raise ValueError("task auxiliary profile identity is invalid")
        if not self.state_dtype:
            raise ValueError("task state dtype must not be empty")
        if self.generalized_dimension <= 0:
            raise ValueError("generalized dimension must be positive")
        if any(
            value < 0
            for value in (
                self.linear_state_elements,
                self.dense_state_elements,
                self.dense_linear_algebra_work,
                self.cell_degrees_of_freedom,
            )
        ):
            raise ValueError("task state and work counts must be non-negative")
        if self.variable_cell and not self.stress_required:
            raise ValueError("variable-cell relaxation requires stress")
        if self.variable_cell != (self.cell_degrees_of_freedom > 0):
            raise ValueError(
                "variable-cell mode must agree with positive cell degrees of freedom"
            )
        if self.variable_cell != (self.cell_method is not None):
            raise ValueError(
                "variable-cell mode must agree with an explicit cell method"
            )
        if self.optimizer_state_kind == "dense":
            expected = self.generalized_dimension**2
            if self.dense_state_elements != expected:
                raise ValueError(
                    "dense optimizer state must contain D squared elements"
                )
        elif self.dense_state_elements != 0:
            raise ValueError("non-dense task cannot declare dense state")

    @classmethod
    def relaxation(
        cls,
        *,
        index: int,
        atom_count: int,
        optimizer: object,
        variable_cell: bool,
        cell_method: object | None = None,
    ) -> TaskAuxiliaryCostProfile:
        """Bind the current relaxation representation to one structure."""

        algorithm = _qualified_name(optimizer)
        name = algorithm.lower()
        options = getattr(optimizer, "options", {})
        requested_dtype = (
            options.get("optimizer_dtype")
            if isinstance(options, Mapping)
            else None
        )
        state_dtype = (
            "calculator_state_dtype"
            if requested_dtype is None
            else (
                str(requested_dtype)
                if str(requested_dtype).startswith("torch.")
                else f"torch.{requested_dtype}"
            )
        )
        if variable_cell and cell_method is None:
            raise ValueError(
                "variable-cell relaxation requires an explicit cell method"
            )
        cell_dof = 9 if variable_cell else 0
        dimension = 3 * atom_count + cell_dof
        dense = "bfgs" in name or "quasinewton" in name
        if dense:
            state_kind: OptimizerStateKind = "dense"
            dense_elements = dimension**2
            linear_elements = 0
            dense_work = dimension**3
        else:
            state_kind = "linear"
            dense_elements = 0
            linear_elements = dimension
            dense_work = 0
        return cls(
            index=index,
            operation="optimization",
            algorithm=algorithm,
            variable_cell=variable_cell,
            stress_required=variable_cell,
            cell_method=(
                None if cell_method is None else _qualified_name(cell_method)
            ),
            cell_degrees_of_freedom=cell_dof,
            generalized_dimension=dimension,
            optimizer_state_kind=state_kind,
            state_dtype=state_dtype,
            linear_state_elements=linear_elements,
            dense_state_elements=dense_elements,
            dense_linear_algebra_work=dense_work,
            horizon_kind="variable",
        )

    @classmethod
    def static_evaluation(
        cls,
        *,
        index: int,
        atom_count: int,
        stress_required: bool,
    ) -> TaskAuxiliaryCostProfile:
        """Describe one force/energy/stress evaluation without method state."""

        return cls(
            index=index,
            operation="evaluation",
            algorithm="single_point",
            variable_cell=False,
            stress_required=stress_required,
            cell_method=None,
            cell_degrees_of_freedom=0,
            generalized_dimension=3 * atom_count,
            optimizer_state_kind="none",
            state_dtype="none",
            linear_state_elements=0,
            dense_state_elements=0,
            dense_linear_algebra_work=0,
            horizon_kind="fixed",
        )

    @classmethod
    def molecular_dynamics(
        cls,
        *,
        index: int,
        atom_count: int,
        ensemble: str,
        variable_cell: bool,
        extended_state_elements: int = 0,
        state_dtype: str = "calculator_state_dtype",
    ) -> TaskAuxiliaryCostProfile:
        """Describe persistent velocity and thermostat/barostat state."""

        if extended_state_elements < 0:
            raise ValueError("extended_state_elements must be non-negative")
        dimension = 3 * atom_count + (9 if variable_cell else 0)
        return cls(
            index=index,
            operation="molecular_dynamics",
            algorithm=ensemble,
            variable_cell=variable_cell,
            stress_required=variable_cell,
            cell_method=ensemble if variable_cell else None,
            cell_degrees_of_freedom=9 if variable_cell else 0,
            generalized_dimension=dimension,
            optimizer_state_kind="linear",
            state_dtype=state_dtype,
            linear_state_elements=3 * atom_count + extended_state_elements,
            dense_state_elements=0,
            dense_linear_algebra_work=0,
            horizon_kind="fixed",
        )


@dataclass(frozen=True)
class GraphExecutionCostProfile:
    """Graph-policy memory/work that is neither structure nor MLIP identity."""

    index: int
    skin_A: float
    candidate_edge_count: int
    cache_enabled: bool
    neighbor_backend: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("execution graph index is invalid")
        if not math.isfinite(self.skin_A) or self.skin_A < 0.0:
            raise ValueError("skin must be finite and non-negative")
        if self.candidate_edge_count < 0:
            raise ValueError("candidate edge count must be non-negative")
        if not self.neighbor_backend:
            raise ValueError("neighbor backend must not be empty")
        if self.cache_enabled != (self.skin_A > 0.0):
            raise ValueError("cache_enabled must agree with positive skin")


@dataclass(frozen=True)
class BoundSystemCostProfile:
    """One index-aligned composition of all pre-hardware cost layers."""

    structure: StructureCostProfile
    mlip_graph: MLIPGraphCostProfile
    task_auxiliary: TaskAuxiliaryCostProfile
    graph_execution: GraphExecutionCostProfile

    def __post_init__(self) -> None:
        indices = {
            self.structure.index,
            self.mlip_graph.index,
            self.task_auxiliary.index,
            self.graph_execution.index,
        }
        if len(indices) != 1:
            raise ValueError("all bound cost layers must use the same index")
        if (
            self.graph_execution.candidate_edge_count
            < self.mlip_graph.active_edge_count
        ):
            raise ValueError("candidate edges cannot be fewer than active edges")

    @property
    def index(self) -> int:
        return self.structure.index

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def base_compute_features(self) -> dict[str, int]:
        """Return the general node/active-edge MLIP work interface."""

        return {
            "atom_count": self.structure.atom_count,
            "active_edge_count": self.mlip_graph.active_edge_count,
        }


@dataclass(frozen=True)
class HardwareCostProfile:
    """Hardware binding applied after structure/model/task profiling."""

    device_type: str
    device_name: str
    total_memory_bytes: int
    memory_safety_fraction: float
    device_count: int

    def __post_init__(self) -> None:
        if not self.device_type or not self.device_name:
            raise ValueError("hardware identity must not be empty")
        if self.total_memory_bytes <= 0 or self.device_count <= 0:
            raise ValueError("hardware memory and device count must be positive")
        if not 0.0 < self.memory_safety_fraction < 1.0:
            raise ValueError("memory safety fraction must be between zero and one")


@dataclass(frozen=True)
class LayeredCostFeatures:
    """Additive features for one resident batch or execution chunk."""

    system_count: int
    atom_count: int
    active_edge_count: int
    candidate_edge_count: int
    linear_state_elements: int
    dense_state_elements: int
    dense_linear_algebra_work: int

    def __post_init__(self) -> None:
        if self.system_count <= 0:
            raise ValueError("layered feature system_count must be positive")
        if any(
            value < 0
            for name, value in asdict(self).items()
            if name != "system_count"
        ):
            raise ValueError("layered feature counts must be non-negative")

    @classmethod
    def from_profiles(
        cls,
        profiles: Iterable[BoundSystemCostProfile],
    ) -> LayeredCostFeatures:
        """Sum system features while retaining one batch-level intercept."""

        items = tuple(profiles)
        if not items:
            raise ValueError("at least one bound system profile is required")
        return cls(
            system_count=len(items),
            atom_count=sum(item.structure.atom_count for item in items),
            active_edge_count=sum(
                item.mlip_graph.active_edge_count for item in items
            ),
            candidate_edge_count=sum(
                item.graph_execution.candidate_edge_count for item in items
            ),
            linear_state_elements=sum(
                item.task_auxiliary.linear_state_elements for item in items
            ),
            dense_state_elements=sum(
                item.task_auxiliary.dense_state_elements for item in items
            ),
            dense_linear_algebra_work=sum(
                item.task_auxiliary.dense_linear_algebra_work
                for item in items
            ),
        )


@dataclass(frozen=True)
class LayeredCostCoefficients:
    """Hardware/task-bound coefficients; never a universal workload formula."""

    fixed: float = 0.0
    per_atom: float = 0.0
    per_active_edge: float = 0.0
    per_candidate_edge: float = 0.0
    per_linear_state_element: float = 0.0
    per_dense_state_element: float = 0.0
    per_dense_linear_algebra_work: float = 0.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in asdict(self).values()
        ):
            raise ValueError("layered cost coefficients must be finite and non-negative")

    def estimate_features(self, features: LayeredCostFeatures) -> float:
        """Estimate one batch; the fixed term is charged exactly once."""

        return (
            self.fixed
            + self.per_atom * features.atom_count
            + self.per_active_edge * features.active_edge_count
            + self.per_candidate_edge
            * features.candidate_edge_count
            + self.per_linear_state_element
            * features.linear_state_elements
            + self.per_dense_state_element
            * features.dense_state_elements
            + self.per_dense_linear_algebra_work
            * features.dense_linear_algebra_work
        )

    def estimate(self, profile: BoundSystemCostProfile) -> float:
        """Estimate a singleton batch for compatibility."""

        return self.estimate_features(LayeredCostFeatures.from_profiles((profile,)))


@dataclass(frozen=True)
class HardwareBoundCostModel:
    """One calibrated time or memory model for an exact execution contract."""

    contract_id: str
    metric: Literal["seconds", "bytes"]
    hardware: HardwareCostProfile
    coefficients: LayeredCostCoefficients
    safety_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("cost-model contract_id must not be empty")
        if self.metric not in ("seconds", "bytes"):
            raise ValueError("cost-model metric must be 'seconds' or 'bytes'")
        if not math.isfinite(self.safety_factor) or self.safety_factor < 1.0:
            raise ValueError("cost-model safety_factor must be at least one")

    def estimate(self, profile: BoundSystemCostProfile) -> float:
        return self.safety_factor * self.coefficients.estimate(profile)

    def estimate_batch(
        self,
        profiles: Iterable[BoundSystemCostProfile],
    ) -> float:
        """Estimate one resident batch with one launch-level fixed cost."""

        features = LayeredCostFeatures.from_profiles(profiles)
        return self.safety_factor * self.coefficients.estimate_features(features)


def structure_workload_sha256(manifest: WorkloadManifest) -> str:
    """Hash ordered structure identities without graph/task profile fields."""

    manifest.verify()
    return _canonical_sha256(
        {
            "workload_id": manifest.workload_id,
            "jobs": [
                {
                    "order": job.order,
                    "dataset_id": job.dataset_id,
                    "source_path": job.source_path,
                    "source_sha256": job.source_sha256,
                    "normalized_structure_sha256": (
                        job.normalized_structure_sha256
                    ),
                    "frame_index": job.frame_index,
                    "atom_count": job.atom_count,
                    "species": job.species,
                    "pbc": job.pbc,
                    "cell_A": job.cell_A,
                    "constraints": job.constraints,
                }
                for job in manifest.jobs
            ],
        }
    )


@dataclass(frozen=True)
class PlanningProfileBundle:
    """Sealed model/task/policy sidecar for an immutable workload manifest."""

    workload_id: str
    workload_manifest_sha256: str
    structure_workload_sha256: str
    systems: tuple[BoundSystemCostProfile, ...]
    profile_sha256: str = ""
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported planning profile schema")
        if (
            not self.workload_id
            or len(self.workload_manifest_sha256) != 64
            or len(self.structure_workload_sha256) != 64
        ):
            raise ValueError("workload identity and hash are required")
        if not self.systems:
            raise ValueError("planning profile systems must not be empty")
        if [system.index for system in self.systems] != list(
            range(len(self.systems))
        ):
            raise ValueError("planning profile indices must be contiguous")

    def unsigned_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("profile_sha256")
        return payload

    def calculate_sha256(self) -> str:
        return _canonical_sha256(self.unsigned_dict())

    def seal(self) -> PlanningProfileBundle:
        return replace(self, profile_sha256=self.calculate_sha256())

    def verify(self) -> None:
        if not self.profile_sha256:
            raise ValueError("planning profile is not sealed")
        if self.profile_sha256 != self.calculate_sha256():
            raise ValueError("planning profile content hash does not match")

    def to_dict(self) -> dict[str, Any]:
        payload = self.unsigned_dict()
        payload["profile_sha256"] = self.profile_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanningProfileBundle:
        values = dict(payload)
        values["systems"] = tuple(
            BoundSystemCostProfile(
                structure=StructureCostProfile(**item["structure"]),
                mlip_graph=MLIPGraphCostProfile(**item["mlip_graph"]),
                task_auxiliary=TaskAuxiliaryCostProfile(
                    **item["task_auxiliary"]
                ),
                graph_execution=GraphExecutionCostProfile(
                    **item["graph_execution"]
                ),
            )
            for item in values["systems"]
        )
        profile = cls(**values)
        profile.verify()
        return profile


def planning_profile_from_manifest(
    manifest: WorkloadManifest,
    *,
    model_id: str,
    cutoff_A: float,
    active_edge_key: str,
    candidate_edge_key: str,
    force_mode: str,
    model_dtype: str,
    optimizer: object,
    variable_cell: bool,
    cell_method: object | None,
    skin_A: float,
    neighbor_backend: str,
) -> PlanningProfileBundle:
    """Build a task-bound sidecar without changing workload identity."""

    manifest.verify()
    systems = []
    for index, job in enumerate(manifest.jobs):
        try:
            active_edges = job.topology_edge_counts[active_edge_key]
            candidate_edges = job.topology_edge_counts[candidate_edge_key]
        except KeyError as error:
            raise KeyError(
                f"job {job.system_id} lacks requested graph profile {error.args[0]!r}"
            ) from error
        systems.append(
            BoundSystemCostProfile(
                structure=StructureCostProfile(
                    index=index,
                    atom_count=job.atom_count,
                ),
                mlip_graph=MLIPGraphCostProfile(
                    index=index,
                    model_id=model_id,
                    cutoff_A=cutoff_A,
                    active_edge_count=active_edges,
                    force_mode=force_mode,
                    model_dtype=model_dtype,
                ),
                task_auxiliary=TaskAuxiliaryCostProfile.relaxation(
                    index=index,
                    atom_count=job.atom_count,
                    optimizer=optimizer,
                    variable_cell=variable_cell,
                    cell_method=cell_method,
                ),
                graph_execution=GraphExecutionCostProfile(
                    index=index,
                    skin_A=skin_A,
                    candidate_edge_count=candidate_edges,
                    cache_enabled=skin_A > 0.0,
                    neighbor_backend=neighbor_backend,
                ),
            )
        )
    return PlanningProfileBundle(
        workload_id=manifest.workload_id,
        workload_manifest_sha256=manifest.manifest_sha256,
        structure_workload_sha256=structure_workload_sha256(manifest),
        systems=tuple(systems),
    ).seal()


def write_planning_profile(
    path: str | Path,
    profile: PlanningProfileBundle,
) -> None:
    """Write one sealed model/task/policy sidecar."""

    profile.verify()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_planning_profile(path: str | Path) -> PlanningProfileBundle:
    """Read and verify one model/task/policy sidecar."""

    return PlanningProfileBundle.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
