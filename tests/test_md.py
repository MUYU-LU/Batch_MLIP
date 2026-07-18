from __future__ import annotations

import torch
from ase import Atoms

from atombit_batch import (
    AseGraphBatch,
    BatchedPotential,
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from atombit_batch.toy_models import QuadraticWellModel


def test_velocity_verlet_has_small_energy_drift_for_quadratic_well():
    state = AseGraphBatch.from_ase(
        [Atoms("H", positions=[[0.2, 0.0, 0.0]])],
        cutoff=2.0,
        skin=1.0,
        device="cpu",
        dtype=torch.float64,
    )
    state.velocities[:] = torch.tensor([[0.0, 0.01, 0.0]], dtype=torch.float64)
    potential = BatchedPotential(
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
    potential = BatchedPotential(
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
