from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import pytest
import torch.multiprocessing as torch_mp

from batch_mlip import (
    ParallelWorkerError,
    PersistentTaskPool,
    balance_work,
    run_parallel_task_workers,
    run_parallel_workers,
)


@dataclass(frozen=True)
class StubRunner:
    indices: tuple[int, ...]

    def __call__(self):
        return [index * index for index in self.indices]


class StubPreparer:
    def __call__(self, shard):
        return StubRunner(shard.system_indices)


@dataclass(frozen=True)
class StubTaskRunner:
    worker_id: int

    def __call__(self, value):
        if value == "fail":
            raise RuntimeError("intentional task failure")
        return self.worker_id, value * value


class StubTaskPreparer:
    def __call__(self, worker):
        return StubTaskRunner(worker.worker_id)


@dataclass(frozen=True)
class EnvironmentTaskRunner:
    values: tuple[str | None, ...]

    def __call__(self, value):
        return value, self.values


@dataclass(frozen=True)
class EnvironmentTaskPreparer:
    names: tuple[str, ...]

    def __call__(self, worker):
        del worker
        return EnvironmentTaskRunner(
            tuple(os.environ.get(name) for name in self.names)
        )


@dataclass(frozen=True)
class SharingStrategyRunner:
    def __call__(self, value):
        return value, torch_mp.get_sharing_strategy()


@dataclass(frozen=True)
class SharingStrategyPreparer:
    def __call__(self, worker):
        del worker
        return SharingStrategyRunner()


class HangingExitTaskPreparer:
    """Keep a non-daemon thread alive after the worker acknowledges shutdown."""

    def __call__(self, worker):
        threading.Thread(
            target=time.sleep,
            args=(30.0,),
            daemon=False,
        ).start()
        return StubTaskRunner(worker.worker_id)


@dataclass
class StubTaskSource:
    values: tuple[int, ...]
    prepared_order: tuple[int, ...] = ()
    prefetch_capacity: int = 0
    resolved: list[int] | None = None
    finished: bool = False

    @property
    def task_count(self) -> int:
        return len(self.values)

    def prepare(
        self,
        ordered_task_indices,
        *,
        prefetch_capacity,
        initial_dispatch_count,
    ):
        del initial_dispatch_count
        self.prepared_order = tuple(ordered_task_indices)
        self.prefetch_capacity = prefetch_capacity
        self.resolved = []

    def resolve(self, task_index):
        if self.resolved is None:
            raise RuntimeError("source was not prepared")
        self.resolved.append(task_index)
        return self.values[task_index]

    def finish(self):
        self.finished = True


def test_balance_work_is_deterministic_and_cost_balanced():
    costs = [1.0, 9.0, 2.0, 8.0, 3.0, 7.0]

    shards = balance_work(costs, ["cuda:0", "cuda:1", "cuda:2"])

    assert shards == balance_work(costs, ["cuda:0", "cuda:1", "cuda:2"])
    assert sorted(
        index for shard in shards for index in shard.system_indices
    ) == list(range(len(costs)))
    assert [shard.estimated_cost for shard in shards] == [10.0, 10.0, 10.0]
    assert all(
        list(shard.system_indices) == sorted(shard.system_indices)
        for shard in shards
    )


def test_parallel_workers_preserve_shard_and_payload_order():
    shards = balance_work([4.0, 1.0, 3.0, 2.0], ["cpu:0", "cpu:1"])

    execution = run_parallel_workers(
        shards,
        StubPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    )

    assert [result.shard.worker_id for result in execution.worker_results] == [0, 1]
    records = [None] * 4
    for result in execution.worker_results:
        for index, value in zip(
            result.shard.system_indices, result.payload, strict=True
        ):
            records[index] = value
    assert records == [0, 1, 4, 9]
    assert execution.startup_wall_seconds >= 0.0
    assert execution.run_wall_seconds >= 0.0
    assert execution.end_to_end_wall_seconds >= execution.run_wall_seconds


def test_parallel_task_workers_execute_every_task_once():
    execution = run_parallel_task_workers(
        [0, 1, 2, 3, 4],
        [1.0, 5.0, 2.0, 4.0, 3.0],
        ["cpu:0", "cpu:1"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    )

    assert [result.task_index for result in execution.task_results] == list(range(5))
    assert [
        result.payload[1] for result in execution.task_results
    ] == [0, 1, 4, 9, 16]
    assigned = [
        index
        for worker in execution.worker_results
        for index in worker.task_indices
    ]
    assert sorted(assigned) == list(range(5))
    assert len(assigned) == len(set(assigned))
    assert [
        worker.task_indices[0] for worker in execution.worker_results
    ] == [1, 3]
    assert execution.end_to_end_wall_seconds >= execution.run_wall_seconds


def test_parallel_task_workers_honor_explicit_initial_dispatch_order():
    execution = run_parallel_task_workers(
        [0, 1, 2, 3],
        [1.0, 5.0, 2.0, 4.0],
        ["cpu:0", "cpu:1"],
        StubTaskPreparer(),
        dispatch_order=(0, 2, 1, 3),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    )

    assert [
        worker.task_indices[0] for worker in execution.worker_results
    ] == [0, 2]


def test_parallel_task_workers_propagate_worker_failure():
    with pytest.raises(ParallelWorkerError, match="intentional task failure"):
        run_parallel_task_workers(
            [1, "fail"],
            [1.0, 2.0],
            ["cpu:0"],
            StubTaskPreparer(),
            startup_timeout_seconds=30.0,
            run_timeout_seconds=30.0,
        )


def test_persistent_task_pool_reuses_workers_and_preserves_call_order():
    with PersistentTaskPool(
        ["cpu:0", "cpu:1"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as pool:
        worker_pids = pool.worker_pids
        first = pool.execute(
            [0, 1, 2, 3, 4],
            [1.0, 5.0, 2.0, 4.0, 3.0],
        )
        second = pool.execute([5, 6], [1.0, 2.0])

        assert pool.worker_pids == worker_pids
        assert first.call_id == 1
        assert second.call_id == 2
        assert [result.task_index for result in first.task_results] == list(range(5))
        assert [result.payload[1] for result in first.task_results] == [
            0,
            1,
            4,
            9,
            16,
        ]
        assert [result.payload[1] for result in second.task_results] == [25, 36]
        assert [
            worker.task_indices[0] for worker in first.worker_results
        ] == [1, 3]
        assert sorted(
            index
            for worker in first.worker_results
            for index in worker.task_indices
        ) == list(range(5))

    assert pool.closed
    assert pool.shutdown_acknowledged_workers == (0, 1)
    assert pool.shutdown_wall_seconds < 1.0


def test_persistent_task_pool_resolves_bounded_source_without_preassignment():
    source = StubTaskSource((0, 1, 2, 3, 4))
    with PersistentTaskPool(
        ["cpu:0", "cpu:1"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as pool:
        execution = pool.execute_source(
            source,
            [1.0, 5.0, 2.0, 4.0, 3.0],
            prefetch_depth=1,
        )

    assert source.prepared_order == (1, 3, 4, 2, 0)
    assert source.prefetch_capacity == 4
    assert sorted(source.resolved or []) == list(range(5))
    assert source.finished
    assert [result.payload[1] for result in execution.task_results] == [
        0,
        1,
        4,
        9,
        16,
    ]
    assert [
        worker.task_indices[0] for worker in execution.worker_results
    ] == [1, 3]


def test_persistent_task_pool_honors_explicit_dispatch_order():
    source = StubTaskSource((0, 1, 2, 3, 4))
    dispatch_order = (4, 0, 3, 2, 1)
    with PersistentTaskPool(
        ["cpu:0", "cpu:1"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as pool:
        execution = pool.execute_source(
            source,
            [1.0, 5.0, 2.0, 4.0, 3.0],
            dispatch_order=dispatch_order,
        )

    assert source.prepared_order == dispatch_order
    assert [
        worker.task_indices[0] for worker in execution.worker_results
    ] == [4, 0]


def test_persistent_task_pool_rejects_invalid_dispatch_order():
    with PersistentTaskPool(
        ["cpu:0"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as pool:
        with pytest.raises(ValueError, match="must be a permutation"):
            pool.execute(
                [0, 1],
                [1.0, 2.0],
                dispatch_order=(0, 0),
            )


def test_persistent_task_pool_uses_bounded_descriptor_sharing():
    parent_strategy = torch_mp.get_sharing_strategy()
    with PersistentTaskPool(
        ["cpu:0"],
        SharingStrategyPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as pool:
        execution = pool.execute([1], [1.0])

    assert execution.task_results[0].payload == (1, "file_descriptor")
    assert torch_mp.get_sharing_strategy() == parent_strategy


def test_persistent_task_pool_failure_breaks_generation():
    pool = PersistentTaskPool(
        ["cpu:0"],
        StubTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    )
    try:
        with pytest.raises(ParallelWorkerError, match="intentional task failure"):
            pool.execute([1, "fail"], [1.0, 2.0])
        assert pool.broken
        with pytest.raises(RuntimeError, match="broken"):
            pool.execute([2], [1.0])
    finally:
        pool.close()


def test_persistent_task_pool_bounds_acknowledged_shutdown_stragglers():
    pool = PersistentTaskPool(
        ["cpu:0", "cpu:1"],
        HangingExitTaskPreparer(),
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
        shutdown_timeout_seconds=0.25,
    )
    execution = pool.execute([1, 2], [1.0, 1.0])

    started = time.perf_counter()
    pool.close()
    elapsed = time.perf_counter() - started

    assert [result.payload[1] for result in execution.task_results] == [1, 4]
    assert pool.shutdown_acknowledged_workers == (0, 1)
    assert pool.shutdown_forced_worker_count == 2
    assert pool.shutdown_wall_seconds < 1.0
    assert elapsed < 1.0


def test_persistent_task_pool_rejects_invalid_shutdown_timeout():
    with pytest.raises(ValueError, match="worker timeouts"):
        PersistentTaskPool(
            ["cpu:0"],
            StubTaskPreparer(),
            shutdown_timeout_seconds=0.0,
        )


@pytest.mark.parametrize("worker_count", [1, 2])
def test_parallel_task_workers_install_child_only_environment(
    monkeypatch,
    worker_count,
):
    names = (
        "BATCH_MLIP_TEST_WORKER_SET",
        "BATCH_MLIP_TEST_WORKER_UNSET",
    )
    monkeypatch.setenv(names[0], "parent")
    monkeypatch.setenv(names[1], "remove-me")

    execution = run_parallel_task_workers(
        list(range(worker_count)),
        [1.0] * worker_count,
        [f"cpu:{index}" for index in range(worker_count)],
        EnvironmentTaskPreparer(names),
        worker_environment={names[0]: "child", names[1]: None},
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    )

    assert [
        result.payload[1] for result in execution.task_results
    ] == [("child", None)] * worker_count
    assert os.environ[names[0]] == "parent"
    assert os.environ[names[1]] == "remove-me"


def test_parallel_workers_reject_invalid_environment_before_spawn():
    shards = balance_work([1.0], ["cpu:0"])

    with pytest.raises(ValueError, match="environment name"):
        run_parallel_workers(
            shards,
            StubPreparer(),
            worker_environment={"BAD=NAME": "value"},
        )


@pytest.mark.parametrize(
    "costs,devices,error",
    [
        ([], ["cuda:0"], "costs"),
        ([1.0], [], "devices"),
        ([1.0], ["cuda:0", "cuda:1"], "cannot exceed"),
        ([0.0], ["cuda:0"], "positive"),
        ([1.0, 2.0], ["cuda:0", "cuda:0"], "unique"),
    ],
)
def test_balance_work_rejects_invalid_inputs(costs, devices, error):
    with pytest.raises(ValueError, match=error):
        balance_work(costs, devices)
