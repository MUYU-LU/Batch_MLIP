from __future__ import annotations

import pytest
from ase import Atoms

from batch_mlip import (
    BatchPlanner,
    BatchTimingPoint,
    CalibrationObservation,
    MemoryCoefficients,
    OptimizationPilot,
    PilotRegime,
    SystemProfile,
    TaskAwarePolicy,
    fit_memory_coefficients,
    plan_relaxation_execution,
    plan_task_aware_relaxation,
)


def test_memory_calibration_recovers_synthetic_peak_model():
    expected = MemoryCoefficients(
        fixed_bytes=1_000_000.0,
        bytes_per_atom=1200.0,
        bytes_per_edge=350.0,
        bytes_per_dof_squared=16.0,
    )
    features = [
        (100, 800, 10_000),
        (240, 1_500, 30_000),
        (500, 4_000, 120_000),
        (900, 6_500, 300_000),
        (1_400, 12_000, 700_000),
        (2_000, 20_000, 1_200_000),
    ]
    observations = [
        CalibrationObservation(
            atom_count=atoms,
            edge_count=edges,
            dof_squared=dof_squared,
            peak_memory_bytes=expected.estimate(
                atom_count=atoms,
                edge_count=edges,
                dof_squared=dof_squared,
            ),
        )
        for atoms, edges, dof_squared in features
    ]

    fitted = fit_memory_coefficients(observations, optimizer_itemsize=8)
    for observation in observations:
        predicted = fitted.estimate(
            atom_count=observation.atom_count,
            edge_count=observation.edge_count,
            dof_squared=observation.dof_squared,
        )
        assert predicted == pytest.approx(
            observation.peak_memory_bytes, rel=2e-5
        )
    assert fitted.bytes_per_dof_squared >= 8.0


def test_planner_profiles_directed_edges_and_variable_cell_dofs():
    planner = BatchPlanner(
        MemoryCoefficients(0.0, 1.0, 1.0, 1.0),
        memory_budget_bytes=1_000_000,
    )
    systems = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
    ]

    profiles = planner.profile_systems(systems, cutoff=1.0)

    assert profiles[0].edge_count == 2
    assert profiles[0].dof_squared == (3 * 2 + 9) ** 2
    assert profiles[1].edge_count == 0
    assert profiles[1].dof_squared == (3 + 9) ** 2


def test_planner_buckets_heterogeneous_costs_and_enforces_budget():
    coefficients = MemoryCoefficients(
        fixed_bytes=100.0,
        bytes_per_atom=10.0,
        bytes_per_edge=2.0,
        bytes_per_dof_squared=1.0,
    )
    profiles = [
        SystemProfile(index=0, atom_count=10, edge_count=20, dof_squared=100),
        SystemProfile(index=1, atom_count=100, edge_count=200, dof_squared=1000),
        SystemProfile(index=2, atom_count=10, edge_count=22, dof_squared=100),
        SystemProfile(index=3, atom_count=100, edge_count=210, dof_squared=1000),
        SystemProfile(index=4, atom_count=10, edge_count=18, dof_squared=100),
    ]
    large_increment = coefficients.estimate(
        atom_count=100, edge_count=210, dof_squared=1000
    ) - int(coefficients.fixed_bytes)
    planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=int(coefficients.fixed_bytes) + 2 * large_increment,
        max_batch_size=8,
        max_cost_ratio=2.0,
    )

    plan = planner.plan_profiles(profiles)

    assert len(plan.buckets) == 2
    assert plan.buckets[0].system_indices == (1, 3)
    assert plan.buckets[0].resident_capacity == 2
    assert plan.buckets[1].system_indices == (0, 2, 4)
    assert sorted(
        index for bucket in plan.buckets for index in bucket.system_indices
    ) == list(range(5))
    assert all(
        bucket.predicted_peak_bytes <= plan.memory_budget_bytes
        for bucket in plan.buckets
    )
    expected_total = coefficients.estimate(
        atom_count=sum(profile.atom_count for profile in profiles),
        edge_count=sum(profile.edge_count for profile in profiles),
        dof_squared=sum(profile.dof_squared for profile in profiles),
    )
    assert planner.estimate_profiles_bytes(plan.profiles) == expected_total


def test_planner_rejects_a_system_larger_than_budget():
    planner = BatchPlanner(
        MemoryCoefficients(100.0, 0.0, 0.0, 1.0),
        memory_budget_bytes=1_000,
    )
    with pytest.raises(MemoryError, match="one system exceeds"):
        planner.plan_profiles(
            [SystemProfile(index=0, atom_count=1, edge_count=0, dof_squared=2_000)]
        )


def test_execution_planner_uses_refill_only_when_optimizer_supports_it():
    systems = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[0.2, 0.0, 0.0]]),
    ]
    planner = BatchPlanner(
        MemoryCoefficients(100.0, 1.0, 1.0, 1.0),
        memory_budget_bytes=1_000_000,
        max_batch_size=1,
        max_cost_ratio=100.0,
    )

    schedule = plan_relaxation_execution(
        planner,
        systems,
        cutoff=1.0,
        supports_refill=True,
    )

    assert schedule.decision == "memory_safe_planned_queues"
    assert len(schedule.batches) == 1
    assert schedule.batches[0].system_indices == (0, 1)
    assert schedule.batches[0].resident_capacity == 1
    assert schedule.batches[0].active_refill


def _task_aware_pilot(
    sampled_steps: tuple[int, ...],
    *,
    mps_systems_per_second: float | None = None,
) -> OptimizationPilot:
    return OptimizationPilot(
        optimizer="fire",
        source="unit-test",
        mps_systems_per_second=mps_systems_per_second,
        regimes=(
            PilotRegime(
                label="H1",
                atom_count=1,
                edge_count=0,
                sampled_steps=sampled_steps,
                timing_points=(
                    BatchTimingPoint(1, 0.010, 100),
                    BatchTimingPoint(2, 0.012, 200),
                ),
            ),
        ),
    )


def test_task_aware_policy_selects_refill_for_a_wide_convergence_tail():
    systems = [
        Atoms("H", positions=[[float(index), 0.0, 0.0]])
        for index in range(4)
    ]
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
        max_batch_size=2,
        max_cost_ratio=100.0,
    )

    schedule = plan_task_aware_relaxation(
        planner,
        systems,
        cutoff=1.0,
        pilot=_task_aware_pilot((0, 9), mps_systems_per_second=100.0),
        policy=TaskAwarePolicy(allow_refill_regime_extrapolation=True),
        supports_refill=True,
    )

    assert schedule.decision == "task_aware_pilot_policy"
    assert len(schedule.batches) == 1
    assert schedule.batches[0].active_refill
    assert schedule.batches[0].resident_capacity == 2
    assert schedule.batches[0].refill_storage == "slots"
    assert schedule.metadata["predicted_tensor_seconds"] == pytest.approx(0.14)
    assert schedule.metadata["recommended_worker_mode"] == "mps"
    decisions = schedule.metadata["bucket_decisions"][0]
    assert decisions["selected_mode"] == "refill"
    assert decisions["selected_capacity"] == 2


def test_task_aware_policy_rejects_refill_without_predicted_benefit():
    systems = [
        Atoms("H", positions=[[float(index), 0.0, 0.0]])
        for index in range(4)
    ]
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
        max_batch_size=2,
        max_cost_ratio=100.0,
    )

    schedule = plan_task_aware_relaxation(
        planner,
        systems,
        cutoff=1.0,
        pilot=_task_aware_pilot((4, 4)),
        policy=TaskAwarePolicy(min_refill_speedup=1.01),
        supports_refill=True,
    )

    assert len(schedule.batches) == 2
    assert all(not batch.active_refill for batch in schedule.batches)
    assert schedule.metadata["recommended_worker_mode"] == "tensor"
    decisions = schedule.metadata["bucket_decisions"][0]
    refill = next(
        candidate
        for candidate in decisions["candidates"]
        if candidate["mode"] == "refill" and candidate["capacity"] == 2
    )
    assert not refill["eligible"]


def test_task_aware_policy_does_not_extrapolate_refill_by_default():
    systems = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
        for _ in range(4)
    ]
    profiles = tuple(
        SystemProfile(index, 2, 2, 225) for index in range(4)
    )
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
        max_batch_size=2,
    )

    schedule = plan_task_aware_relaxation(
        planner,
        systems,
        cutoff=1.0,
        pilot=_task_aware_pilot((0, 9)),
        supports_refill=True,
        system_profiles=profiles,
    )

    assert all(not batch.active_refill for batch in schedule.batches)
    decision = schedule.metadata["bucket_decisions"][0]
    assert not decision["refill_evidence_matched"]
    assert all(
        candidate["mode"] == "drain"
        for candidate in decision["candidates"]
    )


def test_optimization_pilot_serialization_round_trip():
    pilot = _task_aware_pilot((1, 3, 5), mps_systems_per_second=2.5)

    restored = OptimizationPilot.from_dict(pilot.to_dict())

    assert restored == pilot


def test_task_aware_policy_rejects_an_optimizer_mismatched_pilot():
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
    )

    with pytest.raises(ValueError, match="does not match"):
        plan_task_aware_relaxation(
            planner,
            [Atoms("H", positions=[[0.0, 0.0, 0.0]])],
            cutoff=1.0,
            pilot=_task_aware_pilot((1,)),
            optimizer_name="bfgs",
        )


def test_task_aware_policy_accepts_cached_system_profiles():
    systems = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
    ]
    profiles = (
        SystemProfile(0, 1, 0, 144),
        SystemProfile(1, 1, 0, 144),
    )
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
        max_batch_size=2,
    )

    schedule = plan_task_aware_relaxation(
        planner,
        systems,
        cutoff=1.0,
        pilot=_task_aware_pilot((1, 1)),
        system_profiles=profiles,
    )

    assert schedule.plan.profiles == profiles
    assert schedule.plan.profiling_seconds == 0.0


def test_task_aware_policy_excludes_a_measured_capacity_over_budget():
    systems = [
        Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
    ]
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=150,
        max_batch_size=2,
    )

    schedule = plan_task_aware_relaxation(
        planner,
        systems,
        cutoff=1.0,
        pilot=_task_aware_pilot((1, 1)),
    )

    assert len(schedule.batches) == 2
    assert all(batch.resident_capacity == 1 for batch in schedule.batches)


@pytest.mark.parametrize(
    "observations,error",
    [
        ([], "at least four"),
        (
            [CalibrationObservation(1, 1, 1, 1) for _ in range(4)],
            "mandatory optimizer state",
        ),
    ],
)
def test_memory_calibration_rejects_invalid_inputs(observations, error):
    with pytest.raises(ValueError, match=error):
        fit_memory_coefficients(observations)
