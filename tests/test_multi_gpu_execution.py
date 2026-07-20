from __future__ import annotations

from dataclasses import dataclass

import pytest

from batch_mlip import balance_work, run_parallel_workers


@dataclass(frozen=True)
class StubRunner:
    indices: tuple[int, ...]

    def __call__(self):
        return [index * index for index in self.indices]


class StubPreparer:
    def __call__(self, shard):
        return StubRunner(shard.system_indices)


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
