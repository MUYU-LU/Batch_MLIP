from __future__ import annotations

import pytest
import torch
from ase import Atoms
from batch_mlip.toy_models import QuadraticWellModel

from batch_mlip import (
    AseGraphBatch,
    AtomBitBatchCalculator,
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)


def test_velocity_verlet_has_small_energy_drift_for_quadratic_well():
    state = AseGraphBatch.from_ase(
        [Atoms("H", positions=[[0.2, 0.0, 0.0]])],
        cutoff=2.0,
        skin=1.0,
        device="cpu",
        dtype=torch.float64,
    )
    state.velocities[:] = torch.tensor([[0.0, 0.01, 0.0]], dtype=torch.float64)
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0), device="cpu", dtype=torch.float64
    )
    initial = potential(state).energy + state.kinetic_energy()
    result = batched_velocity_verlet(
        state, potential, timestep_fs=0.05, n_steps=1000
    )
    final = result.evaluation.energy + result.kinetic_energy
    assert float(torch.abs(final - initial).max()) < 2e-6


def test_langevin_supports_per_system_temperature_and_friction():
    state = AseGraphBatch.from_ase(
        [
            Atoms("H2", positions=[[0.1, 0, 0], [-0.1, 0, 0]]),
            Atoms("He", positions=[[0.2, 0.1, 0]]),
        ],
        cutoff=2.0,
        device="cpu",
        dtype=torch.float64,
    )
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(), device="cpu", dtype=torch.float64
    )
    initialize_maxwell_boltzmann(
        state, torch.tensor([100.0, 300.0]), seed=5, force_exact_temperature=True
    )
    result = batched_langevin_baoab(
        state,
        potential,
        timestep_fs=torch.tensor([0.1, 0.05]),
        n_steps=10,
        temperature_K=torch.tensor([100.0, 300.0]),
        friction_per_fs=torch.tensor([0.01, 0.02]),
        seed=6,
    )
    assert result.temperature.shape == (2,)
    assert torch.isfinite(result.temperature).all()


def test_per_system_velocity_seeds_are_invariant_to_batch_partitioning():
    systems = [
        Atoms("H2", positions=[[0.1 + offset, 0, 0], [-0.1 + offset, 0, 0]])
        for offset in (0.0, 0.2, 0.4)
    ]
    seeds = [101, 202, 303]
    combined = AseGraphBatch.from_ase(systems, cutoff=2.0, device="cpu", dtype=torch.float64)
    initialize_maxwell_boltzmann(
        combined,
        300.0,
        seed=seeds,
        force_exact_temperature=True,
    )

    partitioned = []
    for atoms, seed in zip(systems, seeds, strict=True):
        state = AseGraphBatch.from_ase([atoms], cutoff=2.0, device="cpu", dtype=torch.float64)
        initialize_maxwell_boltzmann(
            state,
            300.0,
            seed=[seed],
            force_exact_temperature=True,
        )
        partitioned.append(state.velocities)

    torch.testing.assert_close(combined.velocities, torch.cat(partitioned), rtol=0, atol=0)
    with pytest.raises(ValueError, match="one value per system"):
        initialize_maxwell_boltzmann(combined, 300.0, seed=[1, 2])
