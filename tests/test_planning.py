from __future__ import annotations

import pytest
from ase import Atoms

from batch_mlip import (
    BatchPlanner,
    CalibrationObservation,
    MemoryCoefficients,
    SystemProfile,
    fit_memory_coefficients,
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


def test_planner_rejects_a_system_larger_than_budget():
    planner = BatchPlanner(
        MemoryCoefficients(100.0, 0.0, 0.0, 1.0),
        memory_budget_bytes=1_000,
    )
    with pytest.raises(MemoryError, match="one system exceeds"):
        planner.plan_profiles(
            [SystemProfile(index=0, atom_count=1, edge_count=0, dof_squared=2_000)]
        )


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
