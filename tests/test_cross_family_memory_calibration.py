from benchmarks.run_cross_family_memory_calibration import (
    summarize_family_result,
)


def test_family_memory_summary_uses_resident_prediction_and_observed_peak():
    result = {
        "workload_id": "OPT-RB-TEST-R512-v1",
        "jobs": 512,
        "atoms": 1024,
        "call_records": [
            {
                "wall_time_s": 4.0,
                "systems_per_s": 128.0,
                "converged": 0,
                "peak_allocated_bytes": 600,
                "peak_reserved_bytes": 700,
                "model_evaluations": 2,
                "graph_evaluations": 1024,
                "schedule": {
                    "production_run_seconds": 3.0,
                    "resident_plan_chunk_count": 2,
                    "execution_chunk_count": 2,
                    "resident_plan_chunks": [
                        {"predicted_peak_bytes": 800},
                        {"predicted_peak_bytes": 1000},
                    ],
                    "planned_chunks": [
                        {"system_count": 256},
                        {"system_count": 256},
                    ],
                    "memory_budget_bytes_per_gpu": 1400,
                    "parallel_chunk_policy": "resident_chunks_work_stealing",
                },
            }
        ],
    }

    summary = summarize_family_result(result)

    assert summary["predicted_peak_bytes"] == 1000
    assert summary["observed_reserved_to_predicted"] == 0.7
    assert summary["observed_reserved_to_budget"] == 0.5
    assert summary["execution_batch_sizes"] == [256, 256]
