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


@dataclass(frozen=True)
class TaskWorker:
    """Identity and device of one persistent task worker."""

    worker_id: int
    device: str


@dataclass(frozen=True)
class TaskResult:
    """One indexed task result returned by a persistent worker."""

    task_index: int
    worker_id: int
    run_seconds: float
    payload: Any


@dataclass(frozen=True)
class TaskWorkerResult:
    """Lifecycle and task assignment recorded for one persistent worker."""

    worker: TaskWorker
    startup_seconds: float
    run_seconds: float
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class MultiGPUTaskExecution:
    """Dynamic task-queue outputs and persistent-worker timings."""

    task_results: tuple[TaskResult, ...]
    worker_results: tuple[TaskWorkerResult, ...]
    startup_wall_seconds: float
    run_wall_seconds: float
    end_to_end_wall_seconds: float


class PreparedWorker(Protocol):
    """Callable produced after model loading and warm-up in a child process."""

    def __call__(self) -> Any: ...


WorkerPreparer = Callable[[WorkerShard], PreparedWorker]


class PreparedTaskWorker(Protocol):
    """Callable that executes one task using persistent worker state."""

    def __call__(self, task: Any) -> Any: ...


TaskWorkerPreparer = Callable[[TaskWorker], PreparedTaskWorker]


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


def _task_worker_entry(
    worker: TaskWorker,
    prepare: TaskWorkerPreparer,
    initial_item: tuple[int, Any],
    task_queue: Any,
    ready_queue: Any,
    result_queue: Any,
    start_event: Any,
    finish_event: Any,
) -> None:
    startup_started = time.perf_counter()
    try:
        runner = prepare(worker)
        startup_seconds = time.perf_counter() - startup_started
        ready_queue.put((worker.worker_id, startup_seconds, None))
        start_event.wait()
        run_started = time.perf_counter()
        task_indices = []
        pending_item: tuple[int, Any] | None = initial_item
        while True:
            item = pending_item if pending_item is not None else task_queue.get()
            pending_item = None
            if item is None:
                break
            task_index, task = item
            task_started = time.perf_counter()
            payload = runner(task)
            result_queue.put(
                (
                    "task",
                    worker.worker_id,
                    int(task_index),
                    time.perf_counter() - task_started,
                    payload,
                    None,
                )
            )
            task_indices.append(int(task_index))
        result_queue.put(
            (
                "done",
                worker.worker_id,
                startup_seconds,
                time.perf_counter() - run_started,
                tuple(task_indices),
                None,
            )
        )
        # Torch CPU tensors use a child-owned resource sharer during queue
        # deserialization. Keep the process alive until the parent confirms
        # that every task payload has been rebuilt.
        finish_event.wait()
    except Exception:
        error = traceback.format_exc()
        ready_queue.put((worker.worker_id, None, error))
        result_queue.put(
            ("error", worker.worker_id, None, None, None, error)
        )


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


def run_parallel_task_workers(
    tasks: Sequence[Any],
    costs: Sequence[float],
    devices: Sequence[str],
    prepare: TaskWorkerPreparer,
    *,
    start_method: str = "spawn",
    startup_timeout_seconds: float = 1800.0,
    run_timeout_seconds: float = 7200.0,
) -> MultiGPUTaskExecution:
    """Execute a shared pending queue with one persistent process per device.

    Tasks are inserted in descending estimated-cost order. Workers pull the next
    pending item only after completing their current item, providing work
    stealing without migrating active optimizer state.
    """

    normalized_tasks = tuple(tasks)
    normalized_costs = tuple(float(cost) for cost in costs)
    normalized_devices = tuple(str(device) for device in devices)
    if not normalized_tasks:
        raise ValueError("tasks must not be empty")
    if len(normalized_tasks) != len(normalized_costs):
        raise ValueError("tasks and costs must have the same length")
    if not normalized_devices:
        raise ValueError("devices must not be empty")
    if len(normalized_devices) > len(normalized_tasks):
        raise ValueError("the number of devices cannot exceed the number of tasks")
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError("devices must be unique")
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in normalized_costs):
        raise ValueError("costs must be finite and positive")
    if startup_timeout_seconds <= 0.0 or run_timeout_seconds <= 0.0:
        raise ValueError("worker timeouts must be positive")

    context = mp.get_context(start_method)
    task_queue = context.Queue()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    finish_event = context.Event()
    workers = tuple(
        TaskWorker(worker_id=index, device=device)
        for index, device in enumerate(normalized_devices)
    )
    ordered_task_indices = sorted(
        range(len(normalized_tasks)),
        key=lambda index: (-normalized_costs[index], index),
    )
    initial_items = tuple(
        (task_index, normalized_tasks[task_index])
        for task_index in ordered_task_indices[: len(workers)]
    )
    for task_index in ordered_task_indices[len(workers) :]:
        task_queue.put((task_index, normalized_tasks[task_index]))
    for _ in workers:
        task_queue.put(None)
    processes = [
        context.Process(
            target=_task_worker_entry,
            args=(
                worker,
                prepare,
                initial_items[worker.worker_id],
                task_queue,
                ready_queue,
                result_queue,
                start_event,
                finish_event,
            ),
            name=f"batch-mlip-task-worker-{worker.worker_id}",
        )
        for worker in workers
    ]
    total_started = time.perf_counter()
    try:
        for process in processes:
            process.start()
        startup_deadline = time.perf_counter() + startup_timeout_seconds
        startup_by_worker: dict[int, float] = {}
        while len(startup_by_worker) < len(processes):
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
            startup_by_worker[int(worker_id)] = float(startup_seconds)
        startup_wall_seconds = time.perf_counter() - total_started

        run_started = time.perf_counter()
        start_event.set()
        run_deadline = time.perf_counter() + run_timeout_seconds
        task_results: dict[int, TaskResult] = {}
        worker_results: dict[int, TaskWorkerResult] = {}
        while (
            len(task_results) < len(normalized_tasks)
            or len(worker_results) < len(workers)
        ):
            kind, worker_id, field_a, field_b, field_c, error = _get_message(
                result_queue,
                processes,
                deadline=run_deadline,
                phase="execution",
            )
            if error is not None:
                raise ParallelWorkerError(
                    f"worker {worker_id} failed during execution:\n{error}"
                )
            worker_id = int(worker_id)
            if kind == "task":
                task_index = int(field_a)
                if task_index in task_results:
                    raise ParallelWorkerError(
                        f"task {task_index} was returned more than once"
                    )
                task_results[task_index] = TaskResult(
                    task_index=task_index,
                    worker_id=worker_id,
                    run_seconds=float(field_b),
                    payload=field_c,
                )
            elif kind == "done":
                if worker_id in worker_results:
                    raise ParallelWorkerError(
                        f"worker {worker_id} returned completion more than once"
                    )
                worker_results[worker_id] = TaskWorkerResult(
                    worker=workers[worker_id],
                    startup_seconds=float(field_a),
                    run_seconds=float(field_b),
                    task_indices=tuple(int(index) for index in field_c),
                )
            else:
                raise ParallelWorkerError(
                    f"worker {worker_id} returned unknown message {kind!r}"
                )
        run_wall_seconds = time.perf_counter() - run_started
        finish_event.set()
        for process in processes:
            process.join(timeout=30.0)
        failed = [process for process in processes if process.exitcode != 0]
        if failed:
            raise ParallelWorkerError("a parallel task worker exited unsuccessfully")
        return MultiGPUTaskExecution(
            task_results=tuple(
                task_results[index] for index in range(len(normalized_tasks))
            ),
            worker_results=tuple(
                worker_results[index] for index in range(len(workers))
            ),
            startup_wall_seconds=startup_wall_seconds,
            run_wall_seconds=run_wall_seconds,
            end_to_end_wall_seconds=time.perf_counter() - total_started,
        )
    except BaseException:
        _terminate(processes)
        raise
    finally:
        task_queue.close()
        ready_queue.close()
        result_queue.close()


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
