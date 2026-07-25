from __future__ import annotations

from batch_mlip import BatchPlan, PlannedBucket, SystemProfile
from benchmarks.benchmark_mixed_scheduling import build_execution_schedule
from benchmarks.summarize_mixed_scheduling import endpoint_diagnostics


def make_plan(*, budget: int = 1_000) -> BatchPlan:
    return BatchPlan(
        profiles=tuple(
            SystemProfile(
                index=index,
                atom_count=10 * (index + 1),
                edge_count=20 * (index + 1),
                dof_squared=100 * (index + 1),
            )
            for index in range(4)
        ),
        buckets=(
            PlannedBucket(
                system_indices=(2, 3),
                resident_capacity=1,
                predicted_peak_bytes=900,
                max_system_bytes=500,
            ),
            PlannedBucket(
                system_indices=(0, 1),
                resident_capacity=2,
                predicted_peak_bytes=600,
                max_system_bytes=300,
            ),
        ),
        memory_budget_bytes=budget,
        profiling_seconds=0.01,
    )


def test_auto_schedule_uses_whole_pool_when_predicted_to_fit():
    schedule, decision = build_execution_schedule(
        mode="auto",
        plan=make_plan(),
        workload_size=4,
        batch_size=4,
        total_predicted=950,
    )

    assert decision == "whole_batch_predicted_to_fit"
    assert schedule == [((0, 1, 2, 3), 4, False)]


def test_auto_schedule_falls_back_to_planned_capacity():
    schedule, decision = build_execution_schedule(
        mode="auto",
        plan=make_plan(),
        workload_size=4,
        batch_size=4,
        total_predicted=1_001,
    )

    assert decision == "fallback_to_cost_buckets"
    assert schedule == [
        ((2, 3), 1, True),
        ((0, 1), 2, False),
    ]


def test_refill_schedule_keeps_the_whole_pending_pool():
    schedule, decision = build_execution_schedule(
        mode="refill",
        plan=make_plan(),
        workload_size=4,
        batch_size=2,
        total_predicted=950,
    )

    assert decision == "fifo_active_refill"
    assert schedule == [((0, 1, 2, 3), 2, True)]


def test_bucketed_schedule_preserves_every_input_index_once():
    schedule, decision = build_execution_schedule(
        mode="bucketed",
        plan=make_plan(),
        workload_size=4,
        batch_size=2,
        total_predicted=950,
    )

    assert decision == "cost_buckets_fixed_capacity"
    assert sorted(index for indices, _, _ in schedule for index in indices) == [
        0,
        1,
        2,
        3,
    ]
    assert all(capacity <= 2 and not refill for _, capacity, refill in schedule)


def test_endpoint_diagnostics_normalizes_each_heterogeneous_system():
    reference = [
        {
            "source": "small",
            "energy_eV": 2.0,
            "max_force_eV_per_A": 0.1,
            "stress_eV_per_A3": [[0.0] * 3] * 3,
            "positions_A": [[0.0, 0.0, 0.0]],
            "cell_A": [[1.0, 0.0, 0.0]] * 3,
            "converged": True,
            "steps": 2,
        },
        {
            "source": "large",
            "energy_eV": 8.0,
            "max_force_eV_per_A": 0.1,
            "stress_eV_per_A3": [[0.0] * 3] * 3,
            "positions_A": [[0.0, 0.0, 0.0]] * 4,
            "cell_A": [[1.0, 0.0, 0.0]] * 3,
            "converged": True,
            "steps": 2,
        },
    ]
    candidate = [
        {**record, "energy_eV": record["energy_eV"] + difference}
        for record, difference in zip(reference, (1.0, 4.0), strict=True)
    ]

    result = endpoint_diagnostics(reference, candidate)

    assert result["absolute_energy_difference_eV_per_atom"]["maximum"] == 1.0
