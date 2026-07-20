from benchmarks.summarize_controlled_matrix import (
    _add_derived_r256_speedups,
    _best_rows,
)


def _row(
    *,
    pool_size: int,
    workload_id: str,
    method: str,
    throughput: float,
    batch_size: int = 1,
    memory_fraction: float = 0.1,
) -> dict[str, object]:
    return {
        "model": "model",
        "task": "nve",
        "pool_size": pool_size,
        "workload_id": workload_id,
        "method": method,
        "batch_size": batch_size,
        "status": "passed" if method == "ase_b1" else "passed_without_reference",
        "throughput_per_s": throughput,
        "memory_gate_fraction": memory_fraction,
        "speedup_vs_ase_b1_measured": None,
        "speedup_vs_ase_b1_derived": None,
        "speedup_reference": None,
    }


def test_r256_speedup_uses_matching_r32_exact_repeat_reference() -> None:
    rows = [
        _row(
            pool_size=32,
            workload_id="MD-NVE-H46-R32-v1",
            method="ase_b1",
            throughput=10.0,
        ),
        _row(
            pool_size=256,
            workload_id="MD-NVE-H46-R256-v1",
            method="native_batch",
            throughput=125.0,
            batch_size=128,
        ),
    ]

    _add_derived_r256_speedups(rows)

    assert rows[1]["speedup_vs_ase_b1_derived"] == 12.5
    assert rows[1]["speedup_reference"] == ("measured_R32_ASE_throughput_exact_repeats")


def test_best_row_rejects_batch_above_memory_gate() -> None:
    rows = [
        _row(
            pool_size=256,
            workload_id="MD-NVE-H46-R256-v1",
            method="native_batch",
            throughput=100.0,
            batch_size=128,
            memory_fraction=0.5,
        ),
        _row(
            pool_size=256,
            workload_id="MD-NVE-H46-R256-v1",
            method="native_batch",
            throughput=110.0,
            batch_size=256,
            memory_fraction=0.9,
        ),
    ]

    best = _best_rows(rows)

    assert len(best) == 1
    assert best[0]["batch_size"] == 128


def test_best_row_uses_smaller_batch_when_gain_is_below_two_percent() -> None:
    rows = [
        _row(
            pool_size=256,
            workload_id="MD-NVE-H46-R256-v1",
            method="native_batch",
            throughput=100.0,
            batch_size=128,
        ),
        _row(
            pool_size=256,
            workload_id="MD-NVE-H46-R256-v1",
            method="native_batch",
            throughput=101.5,
            batch_size=256,
        ),
    ]

    best = _best_rows(rows)

    assert best[0]["batch_size"] == 128
