from __future__ import annotations

from benchmarks.analyze_outer_inner_opportunity import analyze_opportunity


def _chunk(
    task_index: int,
    worker_id: int,
    indices: list[int],
    dispatch: float,
    completion: float,
    compactions: list[tuple[float, int]],
) -> dict:
    return {
        "task_index": task_index,
        "bucket_index": 0,
        "system_indices": indices,
        "system_count": len(indices),
        "dispatch_offset_seconds": dispatch,
        "completion_offset_seconds": completion,
        "input": {"ready_on_dispatch": True},
        "runtime_profile": {
            "total_seconds": completion - dispatch,
            "events": [
                {
                    "name": "active_compaction_slots",
                    "elapsed_seconds": elapsed,
                    "atom_count": 10,
                    "generalized_dimension": 39,
                    "slots": slots,
                }
                for elapsed, slots in compactions
            ],
        },
        "_worker_id": worker_id,
    }


def test_analyzer_separates_refill_opportunity_from_unavoidable_tail():
    chunks = [
        _chunk(0, 0, [0, 1], 0.0, 10.0, [(2.0, 1)]),
        _chunk(1, 1, [2, 3], 0.0, 5.0, []),
        _chunk(2, 1, [4, 5], 5.0, 12.0, [(2.0, 1)]),
        _chunk(3, 0, [6, 7], 10.0, 15.0, [(1.0, 1)]),
    ]
    payload = {
        "pool_size": 8,
        "workload_id": "test",
        "scheduling": {
            "active_gpu_count": 2,
            "production_run_seconds": 15.0,
            "workers": [
                {
                    "worker_id": worker_id,
                    "chunks": [
                        {key: value for key, value in chunk.items() if key != "_worker_id"}
                        for chunk in chunks
                        if chunk["_worker_id"] == worker_id
                    ],
                }
                for worker_id in (0, 1)
            ],
        },
    }

    result = analyze_opportunity(payload, ((10, 39),) * 8)

    assert result["slot_seconds"]["refill_opportunity"] == 11.0
    assert result["slot_seconds"]["unavoidable_tail"] == 6.0
    assert result["gpu_seconds"]["unavoidable_tail"] == 3.0
    assert result["interpretation"]["not_a_speedup_prediction"]
