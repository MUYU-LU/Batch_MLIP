"""Pickle-safe structure providers for eager and source-backed execution."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ase import Atoms
from ase.io import read

if TYPE_CHECKING:
    from ..workloads.schema import WorkloadManifest


class StructureProvider(Protocol):
    """Materialize ordered systems without exposing their storage format."""

    @property
    def system_count(self) -> int: ...

    @property
    def fully_periodic(self) -> bool: ...

    @property
    def mode(self) -> str: ...

    def materialize(self, indices: Sequence[int]) -> list[Atoms]: ...


_AUTO_LOADER_MEDIUM_PROCESS_COUNT = 2
_AUTO_LOADER_MEDIUM_MIN_POOL_SIZE = 512
_AUTO_LOADER_MEDIUM_MIN_ATOMS_PER_WORKER = 3_000.0
_AUTO_LOADER_LARGE_PROCESS_COUNT = 4
_AUTO_LOADER_LARGE_MIN_POOL_SIZE = 2048
_AUTO_LOADER_LARGE_MIN_ATOMS_PER_WORKER = 32_000.0


@dataclass(frozen=True)
class ManifestLoaderDecision:
    """Static worker-local structure-loading decision."""

    requested: int | Literal["auto"]
    process_count: int
    reason: str
    pool_size: int
    total_atoms: int
    atoms_per_worker: float
    active_worker_count: int
    available_cpu_count: int
    required_cpu_count: int
    policy_id: str = "manifest-loader-process-policy-v2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "requested": self.requested,
            "process_count": self.process_count,
            "reason": self.reason,
            "pool_size": self.pool_size,
            "total_atoms": self.total_atoms,
            "atoms_per_worker": self.atoms_per_worker,
            "active_worker_count": self.active_worker_count,
            "available_cpu_count": self.available_cpu_count,
            "required_cpu_count": self.required_cpu_count,
            "automatic_thresholds": {
                "medium": {
                    "process_count": _AUTO_LOADER_MEDIUM_PROCESS_COUNT,
                    "minimum_pool_size": _AUTO_LOADER_MEDIUM_MIN_POOL_SIZE,
                    "minimum_atoms_per_worker": (
                        _AUTO_LOADER_MEDIUM_MIN_ATOMS_PER_WORKER
                    ),
                },
                "large": {
                    "process_count": _AUTO_LOADER_LARGE_PROCESS_COUNT,
                    "minimum_pool_size": _AUTO_LOADER_LARGE_MIN_POOL_SIZE,
                    "minimum_atoms_per_worker": (
                        _AUTO_LOADER_LARGE_MIN_ATOMS_PER_WORKER
                    ),
                },
            },
        }


def select_manifest_loader_processes(
    atom_counts: Sequence[int],
    *,
    active_worker_count: int,
    requested: int | Literal["auto"] = "auto",
    available_cpu_count: int | None = None,
    compute_threads_per_worker: int = 1,
    manifest_backed: bool = True,
) -> ManifestLoaderDecision:
    """Select bounded CPU parsing parallelism without a timing pilot."""

    counts = tuple(int(count) for count in atom_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("atom_counts must contain positive integers")
    if active_worker_count <= 0:
        raise ValueError("active_worker_count must be positive")
    if compute_threads_per_worker <= 0:
        raise ValueError("compute_threads_per_worker must be positive")
    cpu_count = (
        _available_cpu_count()
        if available_cpu_count is None
        else available_cpu_count
    )
    if cpu_count <= 0:
        raise ValueError("available_cpu_count must be positive")
    total_atoms = sum(counts)
    atoms_per_worker = total_atoms / active_worker_count
    medium_required_cpu_count = active_worker_count * (
        _AUTO_LOADER_MEDIUM_PROCESS_COUNT + compute_threads_per_worker
    )
    large_required_cpu_count = active_worker_count * (
        _AUTO_LOADER_LARGE_PROCESS_COUNT + compute_threads_per_worker
    )

    if requested != "auto":
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested <= 0
        ):
            raise ValueError("requested loader processes must be positive or 'auto'")
        selected = requested if manifest_backed else 1
        reason = (
            "explicit manifest-loader process count"
            if manifest_backed
            else "non-manifest providers do not use process materialization"
        )
    elif not manifest_backed:
        selected = 1
        reason = "non-manifest providers do not use process materialization"
    elif (
        len(counts) >= _AUTO_LOADER_LARGE_MIN_POOL_SIZE
        and atoms_per_worker >= _AUTO_LOADER_LARGE_MIN_ATOMS_PER_WORKER
        and cpu_count >= large_required_cpu_count
    ):
        selected = _AUTO_LOADER_LARGE_PROCESS_COUNT
        reason = "large-pool atom-record and host-CPU gates passed"
    elif (
        len(counts) >= _AUTO_LOADER_MEDIUM_MIN_POOL_SIZE
        and atoms_per_worker >= _AUTO_LOADER_MEDIUM_MIN_ATOMS_PER_WORKER
        and cpu_count >= medium_required_cpu_count
    ):
        selected = _AUTO_LOADER_MEDIUM_PROCESS_COUNT
        reason = "medium-pool atom-record and host-CPU gates passed"
    elif len(counts) < _AUTO_LOADER_MEDIUM_MIN_POOL_SIZE:
        selected = 1
        reason = "pool is below the validated two-process loading regime"
    elif atoms_per_worker < _AUTO_LOADER_MEDIUM_MIN_ATOMS_PER_WORKER:
        selected = 1
        reason = "atom-record pressure per worker is below the two-process gate"
    elif cpu_count < medium_required_cpu_count:
        selected = 1
        reason = "host CPU capacity is insufficient for two loader processes"
    else:
        raise RuntimeError("manifest loader policy did not select a tier")
    required_cpu_count = active_worker_count * (
        selected + compute_threads_per_worker
    )

    return ManifestLoaderDecision(
        requested=requested,
        process_count=selected,
        reason=reason,
        pool_size=len(counts),
        total_atoms=total_atoms,
        atoms_per_worker=atoms_per_worker,
        active_worker_count=active_worker_count,
        available_cpu_count=cpu_count,
        required_cpu_count=required_cpu_count,
    )


def _available_cpu_count() -> int:
    """Return a conservative usable CPU count for loader planning."""

    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        count = os.cpu_count() or 1
    return max(1, int(count))


@dataclass(frozen=True)
class EagerStructureProvider:
    """Adapter preserving the existing in-memory ``Atoms`` execution path."""

    systems: tuple[Atoms, ...]

    def __post_init__(self) -> None:
        if not self.systems:
            raise ValueError("an eager structure provider cannot be empty")

    @property
    def system_count(self) -> int:
        return len(self.systems)

    @property
    def fully_periodic(self) -> bool:
        return all(bool(atoms.pbc.all()) for atoms in self.systems)

    @property
    def mode(self) -> str:
        return "eager_in_memory"

    def materialize(self, indices: Sequence[int]) -> list[Atoms]:
        return [self.systems[index] for index in indices]


@dataclass(frozen=True)
class AseStructureReference:
    """One immutable ASE-readable structure location."""

    system_id: str
    source_path: str
    frame_index: int

    def __post_init__(self) -> None:
        path = PurePosixPath(self.source_path)
        if (
            not self.system_id
            or not self.source_path
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError("structure references require safe relative paths")
        if self.frame_index < 0:
            raise ValueError("structure frame_index must be non-negative")


def _read_ase_reference(
    request: tuple[str, AseStructureReference],
) -> Atoms:
    dataset_dir, reference = request
    atoms = read(
        Path(dataset_dir) / reference.source_path,
        index=reference.frame_index,
    )
    if not isinstance(atoms, Atoms):
        raise TypeError(
            f"{reference.source_path} frame {reference.frame_index} "
            "did not produce one ASE Atoms object"
        )
    atoms.info["batch_mlip_system_id"] = reference.system_id
    return atoms


@dataclass(frozen=True)
class AseManifestStructureProvider:
    """Load signed workload jobs lazily through ASE inside execution workers."""

    dataset_dir: str
    workload_id: str
    workload_manifest_sha256: str
    references: tuple[AseStructureReference, ...]
    periodic: bool

    def __post_init__(self) -> None:
        if not self.dataset_dir or not self.workload_id or not self.references:
            raise ValueError("manifest structure provider fields cannot be empty")
        if len(self.workload_manifest_sha256) != 64:
            raise ValueError("manifest structure provider requires a SHA-256")

    @classmethod
    def from_manifest(
        cls,
        manifest: WorkloadManifest,
        dataset_dir: str | Path,
    ) -> AseManifestStructureProvider:
        manifest.verify()
        root = Path(dataset_dir).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {root}")
        return cls(
            dataset_dir=str(root),
            workload_id=manifest.workload_id,
            workload_manifest_sha256=manifest.manifest_sha256,
            references=tuple(
                AseStructureReference(
                    system_id=job.system_id,
                    source_path=job.source_path,
                    frame_index=job.frame_index,
                )
                for job in manifest.jobs
            ),
            periodic=all(all(job.pbc) for job in manifest.jobs),
        )

    @property
    def system_count(self) -> int:
        return len(self.references)

    @property
    def fully_periodic(self) -> bool:
        return self.periodic

    @property
    def mode(self) -> str:
        return "manifest_lazy_worker"

    def materialize(self, indices: Sequence[int]) -> list[Atoms]:
        return [
            _read_ase_reference((self.dataset_dir, self.references[index]))
            for index in indices
        ]


@dataclass(frozen=True)
class AsyncMaterializationHandle:
    """Ordered futures for one globally unassigned execution chunk."""

    indices: tuple[int, ...]
    futures: tuple[Future[Atoms], ...]
    submitted_at: float


@dataclass
class AsyncStructureMaterializer:
    """Reusable process pool for bounded parent-side chunk prefetch."""

    process_count: int
    start_method: str = "spawn"
    _executor: ProcessPoolExecutor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.process_count, bool)
            or not isinstance(self.process_count, int)
            or self.process_count <= 0
        ):
            raise ValueError("process_count must be a positive integer")
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"unsupported multiprocessing start method {self.start_method!r}"
            )

    def submit(
        self,
        provider: AseManifestStructureProvider,
        indices: Sequence[int],
    ) -> AsyncMaterializationHandle:
        if self._closed:
            raise RuntimeError("asynchronous materializer is closed")
        normalized = tuple(int(index) for index in indices)
        if not normalized:
            raise ValueError("materialization indices must not be empty")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.process_count,
                mp_context=mp.get_context(self.start_method),
            )
        futures = tuple(
            self._executor.submit(
                _read_ase_reference,
                (provider.dataset_dir, provider.references[index]),
            )
            for index in normalized
        )
        return AsyncMaterializationHandle(
            indices=normalized,
            futures=futures,
            submitted_at=time.perf_counter(),
        )

    def resolve(
        self,
        handle: AsyncMaterializationHandle,
    ) -> tuple[list[Atoms], dict[str, Any]]:
        ready_on_dispatch = all(future.done() for future in handle.futures)
        wait_started = time.perf_counter()
        systems = [future.result() for future in handle.futures]
        resolved_at = time.perf_counter()
        return systems, {
            "system_count": len(systems),
            "process_count": self.process_count,
            "elapsed_seconds": resolved_at - handle.submitted_at,
            "dispatch_wait_seconds": resolved_at - wait_started,
            "ready_on_dispatch": ready_on_dispatch,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is None:
            return
        executor = self._executor
        self._executor = None
        executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> AsyncStructureMaterializer:
        if self._closed:
            raise RuntimeError("asynchronous materializer is closed")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass
class StructureMaterializer:
    """Worker-owned deterministic structure loader with optional processes."""

    provider: StructureProvider
    process_count: int = 1
    start_method: str = "spawn"
    _executor: ProcessPoolExecutor | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.process_count, bool)
            or not isinstance(self.process_count, int)
            or self.process_count <= 0
        ):
            raise ValueError("process_count must be a positive integer")
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(
                f"unsupported multiprocessing start method {self.start_method!r}"
            )

    @property
    def parallel(self) -> bool:
        return (
            self.process_count > 1
            and isinstance(self.provider, AseManifestStructureProvider)
        )

    def materialize(self, indices: Sequence[int]) -> list[Atoms]:
        normalized = tuple(int(index) for index in indices)
        if not normalized:
            return []
        if not self.parallel:
            return self.provider.materialize(normalized)
        provider = self.provider
        if not isinstance(provider, AseManifestStructureProvider):
            raise RuntimeError("parallel materializer provider type changed")
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.process_count,
                mp_context=mp.get_context(self.start_method),
            )
        requests = [
            (provider.dataset_dir, provider.references[index])
            for index in normalized
        ]
        chunksize = max(
            1,
            math.ceil(len(requests) / (4 * self.process_count)),
        )
        return list(
            self._executor.map(
                _read_ase_reference,
                requests,
                chunksize=chunksize,
            )
        )

    def close(self) -> None:
        if self._executor is None:
            return
        executor = self._executor
        self._executor = None
        executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> StructureMaterializer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
