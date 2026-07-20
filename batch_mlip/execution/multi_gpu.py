"""One independent process per GPU with deterministic workload sharding."""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class WorkerShard:
    """Input indices assigned to one process and device."""

    worker_id: int
    device: str
    system_indices: tuple[int, ...]
    estimated_cost: float


@dataclass(frozen=True)
class WorkerResult:
    """Serializable payload and timings returned by one worker."""

    shard: WorkerShard
    startup_seconds: float
    run_seconds: float
    payload: Any


@dataclass(frozen=True)
class MultiGPUExecution:
    """Coordinated worker outputs with startup and timed wall durations."""

    worker_results: tuple[WorkerResult, ...]
    startup_wall_seconds: float
    run_wall_seconds: float
    end_to_end_wall_seconds: float


class PreparedWorker(Protocol):
    """Callable produced after model loading and warm-up in a child process."""

    def __call__(self) -> Any: ...


WorkerPreparer = Callable[[WorkerShard], PreparedWorker]


class ParallelWorkerError(RuntimeError):
    """Raised when a child process fails during preparation or execution."""


def balance_work(
    costs: Sequence[float], devices: Sequence[str]
) -> tuple[WorkerShard, ...]:
    """Assign largest jobs first to the currently lightest worker.

    This is deterministic LPT scheduling. Indices within each shard are sorted
    so that each worker sees the same relative input order as the caller.
    """

    normalized_costs = [float(cost) for cost in costs]
    normalized_devices = [str(device) for device in devices]
    if not normalized_costs:
        raise ValueError("costs must not be empty")
    if not normalized_devices:
        raise ValueError("devices must not be empty")
    if len(normalized_devices) > len(normalized_costs):
        raise ValueError("the number of devices cannot exceed the number of jobs")
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError("devices must be unique")
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in normalized_costs):
        raise ValueError("costs must be finite and positive")

    assignments: list[list[int]] = [[] for _ in normalized_devices]
    loads = [0.0] * len(normalized_devices)
    heap = [(0.0, worker_id) for worker_id in range(len(normalized_devices))]
    heapq.heapify(heap)
    for index in sorted(
        range(len(normalized_costs)),
        key=lambda item: (-normalized_costs[item], item),
    ):
        load, worker_id = heapq.heappop(heap)
        assignments[worker_id].append(index)
        loads[worker_id] = load + normalized_costs[index]
        heapq.heappush(heap, (loads[worker_id], worker_id))

    return tuple(
        WorkerShard(
            worker_id=worker_id,
            device=device,
            system_indices=tuple(sorted(assignments[worker_id])),
            estimated_cost=loads[worker_id],
        )
        for worker_id, device in enumerate(normalized_devices)
    )


def _worker_entry(
    shard: WorkerShard,
    prepare: WorkerPreparer,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
) -> None:
    startup_started = time.perf_counter()
    try:
        runner = prepare(shard)
        startup_seconds = time.perf_counter() - startup_started
        ready_queue.put((shard.worker_id, startup_seconds, None))
        start_event.wait()
        run_started = time.perf_counter()
        payload = runner()
        run_seconds = time.perf_counter() - run_started
        result_queue.put(
            (shard.worker_id, startup_seconds, run_seconds, payload, None)
        )
    except Exception:
        error = traceback.format_exc()
        ready_queue.put((shard.worker_id, None, error))
        result_queue.put((shard.worker_id, None, None, None, error))


def _terminate(processes: Sequence[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5.0)


def _get_message(
    message_queue: Any,
    processes: Sequence[mp.Process],
    *,
    deadline: float,
    phase: str,
) -> Any:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(f"parallel worker {phase} timed out")
        try:
            return message_queue.get(timeout=min(1.0, remaining))
        except queue.Empty as error:
            failed = [
                process for process in processes if process.exitcode not in (None, 0)
            ]
            if failed:
                codes = ", ".join(
                    f"pid={process.pid}:exit={process.exitcode}"
                    for process in failed
                )
                raise ParallelWorkerError(
                    f"parallel worker exited during {phase}: {codes}"
                ) from error


def run_parallel_workers(
    shards: Sequence[WorkerShard],
    prepare: WorkerPreparer,
    *,
    start_method: str = "spawn",
    startup_timeout_seconds: float = 1800.0,
    run_timeout_seconds: float = 7200.0,
) -> MultiGPUExecution:
    """Prepare workers independently, release them together, and collect outputs.

    ``prepare`` and everything it captures must be pickleable with ``spawn``.
    Preparation should load the model, warm up its device, and synchronize it.
    The returned callable performs the timed workload and must return a
    pickleable CPU payload.
    """

    normalized = tuple(shards)
    if not normalized:
        raise ValueError("shards must not be empty")
    if startup_timeout_seconds <= 0.0 or run_timeout_seconds <= 0.0:
        raise ValueError("worker timeouts must be positive")
    worker_ids = [shard.worker_id for shard in normalized]
    if sorted(worker_ids) != list(range(len(normalized))):
        raise ValueError("worker ids must be contiguous from zero")
    assigned = [index for shard in normalized for index in shard.system_indices]
    if len(set(assigned)) != len(assigned):
        raise ValueError("system indices must be assigned exactly once")

    context = mp.get_context(start_method)
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_worker_entry,
            args=(shard, prepare, ready_queue, result_queue, start_event),
            name=f"batch-mlip-worker-{shard.worker_id}",
        )
        for shard in normalized
    ]
    total_started = time.perf_counter()
    try:
        for process in processes:
            process.start()
        startup_deadline = time.perf_counter() + startup_timeout_seconds
        ready: dict[int, float] = {}
        while len(ready) < len(processes):
            worker_id, startup_seconds, error = _get_message(
                ready_queue,
                processes,
                deadline=startup_deadline,
                phase="startup",
            )
            if error is not None:
                raise ParallelWorkerError(
                    f"worker {worker_id} failed during startup:\n{error}"
                )
            ready[int(worker_id)] = float(startup_seconds)
        startup_wall_seconds = time.perf_counter() - total_started

        run_started = time.perf_counter()
        start_event.set()
        run_deadline = time.perf_counter() + run_timeout_seconds
        outputs: dict[int, WorkerResult] = {}
        while len(outputs) < len(processes):
            worker_id, startup_seconds, run_seconds, payload, error = _get_message(
                result_queue,
                processes,
                deadline=run_deadline,
                phase="execution",
            )
            if error is not None:
                raise ParallelWorkerError(
                    f"worker {worker_id} failed during execution:\n{error}"
                )
            outputs[int(worker_id)] = WorkerResult(
                shard=normalized[int(worker_id)],
                startup_seconds=float(startup_seconds),
                run_seconds=float(run_seconds),
                payload=payload,
            )
        run_wall_seconds = time.perf_counter() - run_started
        for process in processes:
            process.join(timeout=30.0)
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise ParallelWorkerError("a parallel worker exited unsuccessfully")
        return MultiGPUExecution(
            worker_results=tuple(outputs[index] for index in range(len(outputs))),
            startup_wall_seconds=startup_wall_seconds,
            run_wall_seconds=run_wall_seconds,
            end_to_end_wall_seconds=time.perf_counter() - total_started,
        )
    except BaseException:
        _terminate(processes)
        raise
    finally:
        ready_queue.close()
        result_queue.close()
