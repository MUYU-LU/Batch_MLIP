from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from atombit_batch import (
    ASECalculatorAdapter,
    BatchedFrechetCellFilter,
    BatchedPotential,
    batched_fire_relax,
)
from atombit_batch.filters import GPA_TO_EV_PER_A3
from atombit_batch.toy_models import PairHarmonicModel


def test_graph_model_stress_matches_finite_difference_strain():
    cell = torch.tensor(
        [[4.2, 0.2, 0.1], [0.0, 4.0, 0.3], [0.1, 0.2, 4.4]],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[0.4, 0.5, 0.6], [2.0, 1.1, 0.8], [1.2, 2.5, 2.1]],
        dtype=torch.float64,
    )
    atoms = Atoms(
        "H3", positions=positions.numpy(), cell=cell.numpy(), pbc=True
    )
    calculator = BatchedPotential(
        PairHarmonicModel(k=2.0, r0=1.4, cutoff=3.5),
        cutoff=3.5,
        device="cpu",
        dtype=torch.float64,
    )
    evaluation = calculator(
        calculator.create_state([atoms]), compute_stress=True
    )

    delta = 1e-6
    volume = torch.linalg.det(cell).abs()
    finite_difference = torch.empty((3, 3), dtype=torch.float64)
    for row in range(3):
        for column in range(3):
            strain_parameter = torch.zeros((3, 3), dtype=torch.float64)
            strain_parameter[row, column] = delta
            symmetric_strain = 0.5 * (
                strain_parameter + strain_parameter.T
            )
            energies = []
            for sign in (1.0, -1.0):
                deformation = (
                    torch.eye(3, dtype=torch.float64)
                    + sign * symmetric_strain
                )
                displaced = atoms.copy()
                displaced.set_cell(
                    (cell @ deformation).numpy(), scale_atoms=False
                )
                displaced.positions[:] = (positions @ deformation).numpy()
                energies.append(
                    calculator(
                        calculator.create_state([displaced])
                    ).energy[0]
                )
            finite_difference[row, column] = (
                (energies[0] - energies[1]) / (2.0 * delta * volume)
            )

    torch.testing.assert_close(
        evaluation.stress[0], finite_difference, atol=1e-9, rtol=1e-7
    )


def _lj_calculator() -> LennardJones:
    return LennardJones(sigma=3.4, epsilon=0.0103, rc=8.5)


def _ase_frechet_fire(
    atoms: Atoms, *, hydrostatic: bool, pressure_GPa: float = 0.0
) -> tuple[Atoms, int]:
    reference = atoms.copy()
    reference.calc = _lj_calculator()
    optimizer = FIRE(
        FrechetCellFilter(
            reference,
            hydrostatic_strain=hydrostatic,
            scalar_pressure=pressure_GPa * GPA_TO_EV_PER_A3,
        ),
        logfile=None,
        dt=0.05,
        dtmax=0.5,
        maxstep=0.2,
    )
    optimizer.run(fmax=2e-5, steps=500)
    return reference, optimizer.nsteps


def test_batched_hydrostatic_frechet_fire_matches_individual_ase():
    systems = [bulk("Ar", "fcc", a=a, cubic=True) for a in (5.0, 6.2)]
    references = [_ase_frechet_fire(atoms, hydrostatic=True) for atoms in systems]
    calculator = ASECalculatorAdapter(_lj_calculator())

    result = batched_fire_relax(
        calculator.create_state(systems),
        calculator,
        cell_filter=BatchedFrechetCellFilter(hydrostatic_strain=True),
        fmax=2e-5,
        smax=None,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        max_step=0.2,
    )

    assert bool(result.converged.all())
    assert result.max_stress is not None
    assert result.converged_step.tolist() == [steps for _, steps in references]
    for system_id, (reference, _) in enumerate(references):
        np.testing.assert_allclose(
            result.state.cells[system_id].cpu(), reference.cell.array, atol=2e-12
        )
        np.testing.assert_allclose(
            result.evaluation.energy[system_id].cpu(),
            reference.get_potential_energy(),
            atol=2e-12,
        )


def test_batched_anisotropic_frechet_fire_matches_ase():
    atoms = bulk("Ar", "fcc", a=5.5, cubic=True)
    deformation = np.array(
        [[1.05, 0.08, 0.0], [0.0, 0.95, 0.04], [0.0, 0.0, 1.02]]
    )
    atoms.set_cell(atoms.cell.array @ deformation, scale_atoms=True)
    reference, reference_steps = _ase_frechet_fire(
        atoms, hydrostatic=False
    )
    calculator = ASECalculatorAdapter(_lj_calculator())

    result = batched_fire_relax(
        calculator.create_state([atoms]),
        calculator,
        cell_filter=BatchedFrechetCellFilter(),
        fmax=2e-5,
        smax=None,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        max_step=0.2,
    )

    assert result.steps == reference_steps
    np.testing.assert_allclose(
        result.state.cells[0].cpu(), reference.cell.array, atol=2e-12
    )
    np.testing.assert_allclose(
        result.state.positions.cpu(), reference.positions, atol=2e-12
    )
    np.testing.assert_allclose(
        result.evaluation.stress[0].cpu(),
        reference.get_stress(voigt=False),
        atol=2e-12,
    )


def test_external_compressive_pressure_matches_ase_enthalpy_force():
    atoms = bulk("Ar", "fcc", a=5.5, cubic=True)
    reference, reference_steps = _ase_frechet_fire(
        atoms, hydrostatic=True, pressure_GPa=1.0
    )
    calculator = ASECalculatorAdapter(_lj_calculator())

    result = batched_fire_relax(
        calculator.create_state([atoms]),
        calculator,
        cell_filter=BatchedFrechetCellFilter(
            pressure_GPa=1.0, hydrostatic_strain=True
        ),
        fmax=2e-5,
        smax=None,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        max_step=0.2,
    )

    assert result.steps == reference_steps
    np.testing.assert_allclose(
        result.state.cells[0].cpu(), reference.cell.array, atol=2e-12
    )
    np.testing.assert_allclose(
        result.evaluation.stress[0].cpu(),
        reference.get_stress(voigt=False),
        atol=2e-12,
    )


def test_variable_cell_active_compaction_matches_masked_trajectory_and_reduces_work():
    systems = [
        bulk("Ar", "fcc", a=a, cubic=True)
        for a in (5.2686752, 5.0, 6.2)
    ]

    def run(*, compact: bool):
        calculator = ASECalculatorAdapter(_lj_calculator())
        snapshots = []

        def callback(step, state, evaluation, diagnostics):
            snapshots.append(
                (
                    step,
                    state.positions.clone(),
                    state.cells.clone(),
                    evaluation.energy.clone(),
                    evaluation.stress.clone(),
                    diagnostics["converged"].clone(),
                )
            )

        result = batched_fire_relax(
            calculator.create_state(systems),
            calculator,
            cell_filter=BatchedFrechetCellFilter(
                pressure_GPa=[0.0, 0.2, 0.5],
                cell_factor=[4.0, 6.0, 8.0],
                hydrostatic_strain=True,
            ),
            active_compaction=compact,
            fmax=2e-5,
            smax=5e-7,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
            max_step=0.2,
            callback=callback,
            callback_interval=10,
        )
        return result, snapshots

    masked, masked_snapshots = run(compact=False)
    active, active_snapshots = run(compact=True)

    assert masked.converged_step.tolist() == [0, 54, 81]
    assert active.converged_step.tolist() == masked.converged_step.tolist()
    assert len(active_snapshots) == len(masked_snapshots)
    for masked_snapshot, active_snapshot in zip(
        masked_snapshots, active_snapshots, strict=True
    ):
        assert active_snapshot[0] == masked_snapshot[0]
        for masked_value, active_value in zip(
            masked_snapshot[1:], active_snapshot[1:], strict=True
        ):
            torch.testing.assert_close(active_value, masked_value)

    torch.testing.assert_close(active.state.positions, masked.state.positions)
    torch.testing.assert_close(active.state.cells, masked.state.cells)
    torch.testing.assert_close(active.evaluation.energy, masked.evaluation.energy)
    torch.testing.assert_close(active.evaluation.forces, masked.evaluation.forces)
    torch.testing.assert_close(active.evaluation.stress, masked.evaluation.stress)
    assert active.model_evaluations == masked.model_evaluations
    assert active.graph_evaluations == 138
    assert masked.graph_evaluations == 246
    assert active.active_batch_sizes[0] == 3
    assert active.active_batch_sizes[1] == 2
    assert active.active_batch_sizes[-1] == 1
