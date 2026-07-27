"""Reusable process workers for independent batched relaxation calls."""

from __future__ import annotations

import copy
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ase import Atoms

from ..core.calculator import BatchCalculator
from ..core.types import RelaxationResult
from ..execution import (
    CudaAllocatorPlan,
    PersistentTaskExecution,
    PersistentTaskPool,
    TaskWorker,
    select_cuda_allocator,
)
from ..optimization.registry import (
    BatchOptimizer,
    _validate_capabilities,
    create_optimizer,
)
from ..planning.auto import AutoSchedulerConfig, profile_auto_workload
from ..planning.deterministic import plan_deterministic_relaxation
from .api import (
    _combine_relaxation_results,
    _measure_representative_memory,
    _normalize_devices,
    _normalize_systems,
    _offload_relaxation_result,
    _parallel_deterministic_chunk_policy,
    _parallel_deterministic_chunks,
    relax,
)


@dataclass(frozen=True)
class _OptimizerSpec:
    factory: type[Any] | None
    options: dict[str, Any] | None
    template: BatchOptimizer | None

    @classmethod
    def from_optimizer(cls, optimizer: BatchOptimizer) -> _OptimizerSpec:
        options = getattr(optimizer, "options", None)
        if isinstance(options, Mapping):
            return cls(type(optimizer), dict(options), None)
        return cls(None, None, optimizer)

    def create(self) -> BatchOptimizer:
        if self.factory is not None:
            return self.factory(**(self.options or {}))
        if self.template is None:  # pragma: no cover - construction guarantees one
            raise RuntimeError("optimizer representation is missing")
        return copy.deepcopy(self.template)


@dataclass(frozen=True)
class _ExecutorRelaxTask:
    systems: tuple[Atoms, ...]
    optimizer: _OptimizerSpec
    optimizer_kwargs: dict[str, Any]


@dataclass
class _ExecutorWorkerRunner:
    calculator: BatchCalculator
    allocator_metadata: dict[str, Any]

    def __call__(self, task: _ExecutorRelaxTask) -> RelaxationResult:
        device = self.calculator.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        result = relax(
            task.systems,
            self.calculator,
            optimizer=task.optimizer.create(),
            scheduling="single_batch",
            **task.optimizer_kwargs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            peak_allocated = None
            peak_reserved = None
        if result.state.device.type == "cuda":
            torch.cuda.synchronize(result.state.device)
            result = _offload_relaxation_result(result)
        result.metadata["executor_worker"] = {
            **self.allocator_metadata,
            "pid": os.getpid(),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
        return result


@dataclass
class _ExecutorWorkerPreparer:
    calculator_template: BatchCalculator
    warmup_system: Atoms
    compute_stress: bool
    cpu_threads: int
    allocator_plan: CudaAllocatorPlan

    def __call__(self, worker: TaskWorker) -> _ExecutorWorkerRunner:
        torch.set_num_threads(self.cpu_threads)
        device = torch.device(worker.device)
        cuda_initialized_before_prepare = torch.cuda.is_initialized()
        if device.type == "cuda" and cuda_initialized_before_prepare:
            raise RuntimeError(
                "CUDA initialized before the worker allocator environment "
                "could take effect"
            )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        calculator = self.calculator_template.clone_to(device)
        warm_state = calculator.create_state([self.warmup_system])
        calculator(warm_state, compute_stress=self.compute_stress)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        return _ExecutorWorkerRunner(
            calculator=calculator,
            allocator_metadata={
                **self.allocator_plan.metadata(),
                "applied": device.type == "cuda",
                "cuda_initialized_before_prepare": (
                    cuda_initialized_before_prepare
                ),
                "effective_environment": {
                    name: os.environ.get(name)
                    for name in self.allocator_plan.environment()
                },
                "reported_backend": (
                    torch.cuda.memory.get_allocator_backend()
                    if device.type == "cuda"
                    else None
                ),
            },
        )


@dataclass(frozen=True)
class _ExecutorChunk:
    indices: tuple[int, ...]
    cost: float
    bucket_index: int


def _worker_records(
    execution: PersistentTaskExecution | None,
    chunks: Sequence[_ExecutorChunk],
) -> list[dict[str, Any]]:
    if execution is None:
        return []
    task_results = {
        result.task_index: result for result in execution.task_results
    }
    records = []
    for worker_result in execution.worker_results:
        completed = []
        allocator = None
        for task_index in worker_result.task_indices:
            chunk = chunks[task_index]
            task_result = task_results[task_index]
            worker_metadata = task_result.payload.metadata.get(
                "executor_worker",
                {},
            )
            allocator = worker_metadata
            completed.append(
                {
                    "bucket_index": chunk.bucket_index,
                    "system_count": len(chunk.indices),
                    "estimated_cost": chunk.cost,
                    "wall_seconds": task_result.run_seconds,
                    "peak_allocated_bytes": worker_metadata.get(
                        "peak_allocated_bytes"
                    ),
                    "peak_reserved_bytes": worker_metadata.get(
                        "peak_reserved_bytes"
                    ),
                    "schedule": task_result.payload.metadata.get(
                        "scheduling",
                        {"decision": "single_batch"},
                    ),
                }
            )
        records.append(
            {
                "worker_id": worker_result.worker.worker_id,
                "device": worker_result.worker.device,
                "startup_seconds": worker_result.startup_seconds,
                "task_seconds": worker_result.run_seconds,
                "allocator": allocator,
                "chunks": completed,
            }
        )
    return records


class BatchExecutor:
    """Reuse child-owned calculators across independent relaxation pools.

    Workers start lazily because the optimizer and cell mode determine the CUDA
    allocator policy. Compatible calls retain the same worker generation;
    incompatible allocator or CPU-thread settings trigger a clean restart.
    """

    def __init__(
        self,
        calculator: BatchCalculator,
        *,
        devices: Sequence[str | torch.device],
        auto_config: AutoSchedulerConfig | None = None,
        start_method: str = "spawn",
        startup_timeout_seconds: float = 1800.0,
        run_timeout_seconds: float = 7200.0,
    ) -> None:
        resolved_devices = _normalize_devices(devices)
        if calculator.cutoff is None:
            raise ValueError("BatchExecutor requires a calculator cutoff")
        self.calculator = (
            calculator
            if resolved_devices[0] == calculator.device
            else calculator.clone_to(resolved_devices[0])
        )
        self.devices = resolved_devices
        self.auto_config = auto_config or AutoSchedulerConfig()
        self.start_method = start_method
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.run_timeout_seconds = float(run_timeout_seconds)
        self._pool: PersistentTaskPool | None = None
        self._pool_key: tuple[Any, ...] | None = None
        self._closed = False
        self._generation = 0
        self._relaxation_calls = 0

    @property
    def worker_generation(self) -> int:
        return self._generation

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return () if self._pool is None else self._pool.worker_pids

    @property
    def started(self) -> bool:
        return self._pool is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def _desired_pool_key(
        self,
        allocator_plan: CudaAllocatorPlan,
        config: AutoSchedulerConfig,
    ) -> tuple[Any, ...]:
        device_type = self.devices[0].type
        allocator = (
            allocator_plan.selected_policy
            if device_type == "cuda"
            else "not_applicable"
        )
        return (
            device_type,
            allocator,
            config.multi_gpu_process_cpu_threads,
            self.start_method,
        )

    def _ensure_pool(
        self,
        *,
        allocator_plan: CudaAllocatorPlan,
        config: AutoSchedulerConfig,
        warmup_system: Atoms,
        compute_stress: bool,
    ) -> tuple[bool, float]:
        desired_key = self._desired_pool_key(allocator_plan, config)
        restarted = self._pool is not None and self._pool_key != desired_key
        if self._pool is not None and (
            restarted or self._pool.broken or self._pool.closed
        ):
            self._pool.close()
            self._pool = None
            self._pool_key = None
            restarted = True
        if self._pool is not None:
            return False, 0.0

        main_module = sys.modules.get("__main__")
        if not getattr(main_module, "__file__", None):
            raise RuntimeError(
                "BatchExecutor with spawn requires a file-backed __main__ module"
            )
        calculator_template = self.calculator.clone_to("cpu")
        model = getattr(calculator_template, "model", None)
        if isinstance(model, torch.nn.Module):
            model.share_memory()
        preparer = _ExecutorWorkerPreparer(
            calculator_template=calculator_template,
            warmup_system=warmup_system,
            compute_stress=compute_stress,
            cpu_threads=config.multi_gpu_process_cpu_threads,
            allocator_plan=allocator_plan,
        )
        try:
            pickle.dumps(preparer)
        except Exception as error:
            raise TypeError(
                "BatchExecutor requires a serializable calculator: "
                f"{type(error).__name__}: {error}"
            ) from error
        has_cuda = self.devices[0].type == "cuda"
        self._pool = PersistentTaskPool(
            [str(device) for device in self.devices],
            preparer,
            worker_environment=(
                allocator_plan.environment() if has_cuda else None
            ),
            start_method=self.start_method,
            startup_timeout_seconds=self.startup_timeout_seconds,
            run_timeout_seconds=self.run_timeout_seconds,
        )
        self._pool_key = desired_key
        self._generation += 1
        return restarted, self._pool.startup_wall_seconds

    def relax(
        self,
        systems: Atoms | Sequence[Atoms],
        *,
        optimizer: str | BatchOptimizer = "fire",
        auto_config: AutoSchedulerConfig | None = None,
        **optimizer_kwargs: Any,
    ) -> RelaxationResult:
        if self._closed:
            raise RuntimeError("BatchExecutor is closed")
        total_started = time.perf_counter()
        normalized = _normalize_systems(systems)
        resolved = (
            create_optimizer(optimizer)
            if isinstance(optimizer, str)
            else optimizer
        )
        if not isinstance(resolved, BatchOptimizer):
            raise TypeError(
                "optimizer must be a registered name or implement BatchOptimizer"
            )
        config = auto_config or self.auto_config
        if config.multi_gpu_worker_backend == "thread":
            raise ValueError(
                "BatchExecutor owns process workers and does not accept "
                "multi_gpu_worker_backend='thread'"
            )
        capabilities = resolved.capabilities()
        options = dict(optimizer_kwargs)
        if options.get("refill_batch_size") is not None:
            raise ValueError(
                "BatchExecutor controls refill_batch_size; do not set it"
            )
        if getattr(capabilities, "active_compaction", False):
            options.setdefault("active_compaction", True)
        _validate_capabilities(resolved, options)
        allocator_plan = select_cuda_allocator(
            self.calculator,
            resolved,
            variable_cell=options.get("cell_filter") is not None,
            policy=config.cuda_allocator_policy,
        )

        planning_started = time.perf_counter()
        workload = profile_auto_workload(
            normalized,
            self.calculator,
            resolved,
            options,
            config,
        )
        probe = _measure_representative_memory(
            normalized,
            self.calculator,
            options,
            workload,
            config,
        )
        plan = plan_deterministic_relaxation(
            workload,
            probe,
            resolved,
            options,
            self.calculator.dtype,
            config,
        )
        pending_chunks = _parallel_deterministic_chunks(
            plan,
            device_count=len(self.devices),
            target_chunks_per_device=(
                config.multi_gpu_target_chunks_per_device
            ),
        )
        planning_seconds = time.perf_counter() - planning_started

        restarted, startup_seconds = self._ensure_pool(
            allocator_plan=allocator_plan,
            config=config,
            warmup_system=normalized[0],
            compute_stress=options.get("cell_filter") is not None,
        )
        if self._pool is None:  # pragma: no cover - narrowed by _ensure_pool
            raise RuntimeError("persistent worker pool did not start")
        self._relaxation_calls += 1

        optimizer_spec = _OptimizerSpec.from_optimizer(resolved)
        indexed_results: list[
            tuple[tuple[int, ...], RelaxationResult]
        ] = []
        production_chunks = [
            _ExecutorChunk(
                indices=chunk.indices,
                cost=chunk.estimated_cost,
                bucket_index=chunk.bucket_index,
            )
            for chunk in pending_chunks
        ]
        production_tasks = [
            _ExecutorRelaxTask(
                systems=tuple(normalized[index] for index in chunk.indices),
                optimizer=optimizer_spec,
                optimizer_kwargs=options,
            )
            for chunk in production_chunks
        ]
        production_execution = (
            self._pool.execute(
                production_tasks,
                [chunk.cost for chunk in production_chunks],
            )
            if production_tasks
            else None
        )
        if production_execution is not None:
            for chunk, task_result in zip(
                production_chunks,
                production_execution.task_results,
                strict=True,
            ):
                indexed_results.append((chunk.indices, task_result.payload))

        reassembly_started = time.perf_counter()
        executed = [
            index for indices, _ in indexed_results for index in indices
        ]
        if sorted(executed) != list(range(len(normalized))):
            raise RuntimeError(
                "persistent scheduling duplicated or omitted input systems"
            )
        result = _combine_relaxation_results(
            indexed_results,
            workload_size=len(normalized),
            calculator=self.calculator,
        )
        reassembly_seconds = time.perf_counter() - reassembly_started
        result.metadata["scheduling"] = {
            "policy": "auto",
            "decision": "persistent_deterministic_memory_plan",
            "devices": [str(device) for device in self.devices],
            "gpu_count": len(self.devices),
            "fingerprint": workload.fingerprint,
            "memory_fraction": plan.memory_fraction,
            "memory_growth_margin": plan.memory_growth_margin,
            "memory_budget_bytes_per_gpu": probe.memory_budget_bytes,
            "profiling_seconds": workload.profiling_seconds,
            "planning_seconds": planning_seconds,
            "probe": {
                "device": str(self.calculator.device),
                "system_count": len(probe.probe_indices),
                "system_indices": list(probe.probe_indices),
                "model_forward_count": 1 if probe.probe_indices else 0,
                "baseline_allocated_bytes": probe.baseline_allocated_bytes,
                "peak_allocated_bytes": probe.peak_allocated_bytes,
                "peak_reserved_bytes": probe.peak_reserved_bytes,
                "model_bytes_per_work": probe.model_bytes_per_work,
            },
            "parallel_chunk_policy": _parallel_deterministic_chunk_policy(
                plan,
                device_count=len(self.devices),
                target_chunks_per_device=(
                    config.multi_gpu_target_chunks_per_device
                ),
            ),
            "target_chunks_per_device": (
                config.multi_gpu_target_chunks_per_device
            ),
            "resident_plan_chunk_count": len(plan.chunks),
            "execution_chunk_count": len(pending_chunks),
            "resident_plan_chunks": [
                {
                    "bucket_index": chunk.bucket_index,
                    "system_count": len(chunk.system_indices),
                    "predicted_peak_bytes": chunk.predicted_peak_bytes,
                    "estimated_cost": chunk.estimated_cost,
                }
                for chunk in plan.chunks
            ],
            "planned_chunks": [
                {
                    "bucket_index": chunk.bucket_index,
                    "system_count": len(chunk.indices),
                    "predicted_peak_bytes": chunk.predicted_peak_bytes,
                    "estimated_cost": chunk.estimated_cost,
                }
                for chunk in pending_chunks
            ],
            "reassembly_seconds": reassembly_seconds,
            "executor_call": self._relaxation_calls,
            "worker_generation": self._generation,
            "worker_generation_restarted": restarted,
            "worker_pids": list(self._pool.worker_pids),
            "worker_startup_seconds_this_call": startup_seconds,
            "worker_startup_seconds_generation": (
                self._pool.startup_wall_seconds
            ),
            "allocator": {
                **allocator_plan.metadata(),
                "applied_to_workers": self.devices[0].type == "cuda",
            },
            "optimization_pilot_runs": 0,
            "production_run_seconds": (
                0.0
                if production_execution is None
                else production_execution.run_wall_seconds
            ),
            "workers": _worker_records(
                production_execution,
                production_chunks,
            ),
            "pending_work_stealing": True,
            "total_seconds": time.perf_counter() - total_started,
        }
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            self._pool_key = None

    def __enter__(self) -> BatchExecutor:
        if self._closed:
            raise RuntimeError("BatchExecutor is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_: Any) -> None:
        del exc_type, exc, traceback_
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        if hasattr(self, "_closed") and not self._closed:
            try:
                self.close()
            except Exception:
                pass
