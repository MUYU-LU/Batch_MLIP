from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.md.nose_hoover_chain import IsotropicMTKNPT

from batch_mlip import (
    ASECalculatorAdapter,
    AseGraphBatch,
    AtomBitBatchCalculator,
    BatchCalculator,
    BatchEvaluation,
    IsotropicMTKState,
    LangevinBAOABState,
    batched_isotropic_mtk,
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from batch_mlip.toy_models import QuadraticWellModel


class ZeroForceCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.0, device="cpu", dtype=torch.float64)

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy
        return BatchEvaluation(
            energy=torch.zeros(
                state.n_systems, device=state.device, dtype=state.dtype
            ),
            forces=torch.zeros_like(state.positions),
            stress=(
                torch.zeros(
                    (state.n_systems, 3, 3),
                    device=state.device,
                    dtype=state.dtype,
                )
                if compute_stress
                else None
            ),
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
    assert result.initial_total_energy is not None
    torch.testing.assert_close(result.initial_total_energy, initial)
    assert float(torch.abs(final - initial).max()) < 2e-6


def test_ase_velocity_boundary_preserves_velocity_and_kinetic_energy():
    atoms = Atoms("H2", positions=[[0.1, 0, 0], [-0.1, 0, 0]])
    atoms.set_velocities([[0.2, -0.1, 0.0], [-0.05, 0.15, 0.1]])
    state = AseGraphBatch.from_ase(
        [atoms], cutoff=2.0, device="cpu", dtype=torch.float64, build_neighbors=False
    )

    torch.testing.assert_close(
        state.velocities,
        torch.as_tensor(atoms.get_velocities() * units.fs, dtype=torch.float64),
    )
    assert float(state.kinetic_energy()[0]) == pytest.approx(atoms.get_kinetic_energy())
    np.testing.assert_allclose(
        state.to_ase(evaluation=None)[0].get_velocities(), atoms.get_velocities()
    )


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


def _argon_cell(offset: float = 0.0) -> Atoms:
    atoms = Atoms(
        "Ar4",
        positions=np.asarray(
            [
                [1.0 + offset, 1.0, 1.0],
                [4.0 + offset, 1.0, 1.0],
                [1.0 + offset, 4.0, 1.0],
                [1.0 + offset, 1.0, 4.0],
            ]
        ),
        cell=np.eye(3) * (8.0 + offset),
        pbc=True,
    )
    atoms.set_velocities(
        [
            [0.010, 0.020, -0.010],
            [-0.020, 0.010, 0.005],
            [0.005, -0.010, 0.020],
            [0.005, -0.020, -0.015],
        ]
    )
    return atoms


def _lj_batch_calculator() -> ASECalculatorAdapter:
    return ASECalculatorAdapter(
        LennardJones(epsilon=0.0103, sigma=3.4, rc=6.0),
        dtype=torch.float64,
    )


def test_isotropic_mtk_b1_matches_ase_reference():
    source = _argon_cell()
    reference = source.copy()
    reference.calc = LennardJones(epsilon=0.0103, sigma=3.4, rc=6.0)
    dynamics = IsotropicMTKNPT(
        reference,
        timestep=0.2 * units.fs,
        temperature_K=200.0,
        pressure_au=0.0,
        tdamp=20.0 * units.fs,
        pdamp=200.0 * units.fs,
        tchain=3,
        pchain=3,
    )
    dynamics.run(5)

    calculator = _lj_batch_calculator()
    state = calculator.create_state([source])
    result = batched_isotropic_mtk(
        state,
        calculator,
        timestep_fs=0.2,
        n_steps=5,
        temperature_K=200.0,
        pressure_eV_per_A3=0.0,
        thermostat_damping_fs=20.0,
        barostat_damping_fs=200.0,
    )
    output = result.structures[0]
    np.testing.assert_allclose(output.positions, reference.positions, atol=1e-8)
    np.testing.assert_allclose(
        output.cell.array, reference.cell.array, atol=1e-8
    )
    np.testing.assert_allclose(
        output.get_velocities(), reference.get_velocities(), atol=1e-10
    )
    extended = result.integrator_state
    assert isinstance(extended, IsotropicMTKState)
    np.testing.assert_allclose(
        extended.log_volume_scale.cpu(),
        [dynamics._eps],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        extended.volume_momenta.cpu(),
        [dynamics._p_eps],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        extended.thermostat_positions.cpu()[0],
        dynamics._thermostat._eta,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        extended.thermostat_momenta.cpu()[0],
        dynamics._thermostat._p_eta,
        atol=1e-8,
    )
    assert float(result.metadata["conserved_energy_drift"].abs().max()) < 1e-6


def test_isotropic_mtk_batch_matches_independent_runs():
    systems = [_argon_cell(0.0), _argon_cell(0.15)]
    temperature = torch.tensor([180.0, 240.0], dtype=torch.float64)
    pressure = torch.tensor([0.0, 0.001], dtype=torch.float64)
    tdamp = torch.tensor([20.0, 25.0], dtype=torch.float64)
    pdamp = torch.tensor([200.0, 250.0], dtype=torch.float64)
    calculator = _lj_batch_calculator()
    combined_state = calculator.create_state(systems)
    combined = batched_isotropic_mtk(
        combined_state,
        calculator,
        timestep_fs=torch.tensor([0.2, 0.15]),
        n_steps=4,
        temperature_K=temperature,
        pressure_eV_per_A3=pressure,
        thermostat_damping_fs=tdamp,
        barostat_damping_fs=pdamp,
    )

    singles = []
    for system_id, atoms in enumerate(systems):
        single_calculator = _lj_batch_calculator()
        single_state = single_calculator.create_state([atoms])
        singles.append(
            batched_isotropic_mtk(
                single_state,
                single_calculator,
                timestep_fs=[0.2, 0.15][system_id],
                n_steps=4,
                temperature_K=temperature[system_id],
                pressure_eV_per_A3=pressure[system_id],
                thermostat_damping_fs=tdamp[system_id],
                barostat_damping_fs=pdamp[system_id],
            )
        )

    torch.testing.assert_close(
        combined.state.positions,
        torch.cat([result.state.positions for result in singles]),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        combined.state.cells,
        torch.cat([result.state.cells for result in singles]),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        combined.state.velocities,
        torch.cat([result.state.velocities for result in singles]),
        rtol=1e-10,
        atol=1e-10,
    )
    combined_extended = combined.integrator_state
    assert isinstance(combined_extended, IsotropicMTKState)
    torch.testing.assert_close(
        combined_extended.volume_momenta,
        torch.cat(
            [result.integrator_state.volume_momenta for result in singles]
        ),
        rtol=1e-7,
        atol=1e-8,
    )


def test_isotropic_mtk_restart_matches_uninterrupted_execution():
    source = _argon_cell()
    full_calculator = _lj_batch_calculator()
    full_state = full_calculator.create_state([source])
    full = batched_isotropic_mtk(
        full_state,
        full_calculator,
        timestep_fs=0.2,
        n_steps=8,
        temperature_K=200.0,
        thermostat_damping_fs=20.0,
        barostat_damping_fs=200.0,
    )

    split_calculator = _lj_batch_calculator()
    split_state = split_calculator.create_state([source])
    first = batched_isotropic_mtk(
        split_state,
        split_calculator,
        timestep_fs=0.2,
        n_steps=3,
        temperature_K=200.0,
        thermostat_damping_fs=20.0,
        barostat_damping_fs=200.0,
    )
    restored = IsotropicMTKState.from_state_dict(
        first.integrator_state.state_dict(),
        device=split_state.device,
        dtype=split_state.dtype,
    )
    second = batched_isotropic_mtk(
        split_state,
        split_calculator,
        timestep_fs=0.2,
        n_steps=5,
        temperature_K=200.0,
        thermostat_damping_fs=20.0,
        barostat_damping_fs=200.0,
        integrator_state=restored,
    )

    torch.testing.assert_close(
        second.state.positions, full.state.positions, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        second.state.cells, full.state.cells, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        second.state.velocities, full.state.velocities, rtol=0.0, atol=0.0
    )
    assert second.integrator_state.steps == 8


def test_langevin_restart_matches_uninterrupted_execution():
    systems = [
        Atoms("H2", positions=[[0.1, 0, 0], [-0.1, 0, 0]]),
        Atoms("He", positions=[[0.2, 0.1, 0]]),
    ]
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(), cutoff=2.0, device="cpu", dtype=torch.float64
    )
    full_state = potential.create_state(systems)
    initialize_maxwell_boltzmann(full_state, [100.0, 300.0], seed=11)
    full = batched_langevin_baoab(
        full_state,
        potential,
        timestep_fs=0.1,
        n_steps=10,
        temperature_K=[100.0, 300.0],
        friction_per_fs=0.02,
        seed=12,
    )

    split_state = potential.create_state(systems)
    initialize_maxwell_boltzmann(split_state, [100.0, 300.0], seed=11)
    first = batched_langevin_baoab(
        split_state,
        potential,
        timestep_fs=0.1,
        n_steps=4,
        temperature_K=[100.0, 300.0],
        friction_per_fs=0.02,
        seed=12,
    )
    second = batched_langevin_baoab(
        split_state,
        potential,
        timestep_fs=0.1,
        n_steps=6,
        temperature_K=[100.0, 300.0],
        friction_per_fs=0.02,
        integrator_state=LangevinBAOABState.from_state_dict(
            first.integrator_state.state_dict()
        ),
    )
    torch.testing.assert_close(
        second.state.positions, full.state.positions, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        second.state.velocities, full.state.velocities, rtol=0.0, atol=0.0
    )
    assert second.integrator_state.steps == 10


def test_langevin_samples_target_temperature():
    calculator = ZeroForceCalculator()
    systems = [Atoms("H", positions=[[0, 0, 0]]) for _ in range(64)]
    state = calculator.create_state(systems)
    initialize_maxwell_boltzmann(
        state,
        300.0,
        seed=21,
        remove_com=False,
    )
    samples = []

    def collect(step, current, evaluation, diagnostics):
        del current, evaluation
        if step >= 500:
            samples.append(float(diagnostics["temperature"].mean()))

    batched_langevin_baoab(
        state,
        calculator,
        timestep_fs=0.2,
        n_steps=3000,
        temperature_K=300.0,
        friction_per_fs=0.05,
        seed=22,
        callback=collect,
        callback_interval=10,
    )
    mean_temperature = float(np.mean(samples))
    assert mean_temperature == pytest.approx(300.0, rel=0.05)


def test_isotropic_mtk_extended_energy_converges_with_timestep():
    drifts = []
    for timestep in (0.4, 0.2):
        calculator = _lj_batch_calculator()
        state = calculator.create_state([_argon_cell()])
        result = batched_isotropic_mtk(
            state,
            calculator,
            timestep_fs=timestep,
            n_steps=round(4.0 / timestep),
            temperature_K=200.0,
            thermostat_damping_fs=20.0,
            barostat_damping_fs=200.0,
        )
        drifts.append(float(result.metadata["conserved_energy_drift"].abs()))
        assert bool(torch.isfinite(result.state.cells).all())
        assert float(torch.linalg.det(result.state.cells).min()) > 0.0
    assert drifts[1] < 0.35 * drifts[0]


def test_isotropic_mtk_rejects_unsupported_states_and_parameters():
    calculator = _lj_batch_calculator()
    nonperiodic = calculator.create_state(
        [Atoms("Ar", positions=[[0, 0, 0]], cell=np.eye(3) * 8)]
    )
    with pytest.raises(NotImplementedError, match="three-dimensional periodic"):
        batched_isotropic_mtk(
            nonperiodic,
            calculator,
            timestep_fs=0.2,
            n_steps=1,
            temperature_K=200.0,
        )

    periodic = calculator.create_state([_argon_cell()])
    with pytest.raises(ValueError, match="temperature must be positive"):
        batched_isotropic_mtk(
            periodic,
            calculator,
            timestep_fs=0.2,
            n_steps=1,
            temperature_K=0.0,
        )

    constrained_atoms = _argon_cell()
    constrained_atoms.set_constraint(FixAtoms(indices=[0]))
    constrained = calculator.create_state([constrained_atoms])
    with pytest.raises(NotImplementedError, match="FixAtoms"):
        batched_isotropic_mtk(
            constrained,
            calculator,
            timestep_fs=0.2,
            n_steps=1,
            temperature_K=200.0,
        )
