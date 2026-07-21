from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.optimize import BFGSLineSearch, QuasiNewton

import atombit_batch
from batch_mlip import (
    ASECalculatorAdapter,
    AtomBitBatchCalculator,
    BatchedBFGSLineSearch,
    BatchedQuasiNewton,
    FrechetCellFilter,
    available_optimizers,
    batched_bfgs_line_search_relax,
    create_optimizer,
    relax,
)
from batch_mlip.toy_models import QuadraticWellModel


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.5 * float((atoms.positions**2).sum())
        self.results["forces"] = -atoms.positions.copy()


def _quadratic_potential() -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0),
        cutoff=2.5,
        device="cpu",
        dtype=torch.float64,
    )


def _run_independent(system: Atoms, *, steps: int):
    calculator = _quadratic_potential()
    return batched_bfgs_line_search_relax(
        calculator.create_state([system]),
        calculator,
        fmax=1e-30,
        max_steps=steps,
    )


def test_quasinewton_aliases_bfgs_line_search():
    assert QuasiNewton is BFGSLineSearch
    assert BatchedQuasiNewton is BatchedBFGSLineSearch
    assert {
        "bfgslinesearch",
        "bfgs_line_search",
        "quasinewton",
    } <= set(available_optimizers())
    assert isinstance(create_optimizer("BFGSLineSearch"), BatchedBFGSLineSearch)
    assert isinstance(create_optimizer("QuasiNewton"), BatchedBFGSLineSearch)
    assert atombit_batch.BatchedQuasiNewton is BatchedBFGSLineSearch

    result = relax(
        Atoms("H", positions=[[0.1, 0.0, 0.0]]),
        _quadratic_potential(),
        optimizer="quasinewton",
        fmax=1e-30,
        max_steps=0,
    )
    assert result.steps == 0

    with pytest.raises(ValueError, match="active-batch refill"):
        relax(
            [Atoms("H", positions=[[0.1, 0.0, 0.0]])] * 2,
            _quadratic_potential(),
            optimizer="bfgslinesearch",
            refill_batch_size=1,
        )


def test_fixed_cell_b1_matches_ase_bfgs_line_search():
    initial = Atoms(
        "H2",
        positions=[[1.2, -0.4, 0.3], [-0.7, 0.8, -0.2]],
    )
    reference = initial.copy()
    reference.calc = QuadraticCalculator()
    optimizer = BFGSLineSearch(
        reference,
        logfile=None,
        alpha=10.0,
        maxstep=0.2,
    )
    optimizer.run(fmax=1e-30, steps=6)

    result = _run_independent(initial, steps=6)

    assert result.steps == optimizer.nsteps
    torch.testing.assert_close(
        result.state.positions,
        torch.as_tensor(reference.positions, dtype=torch.float64),
        atol=2e-12,
        rtol=2e-12,
    )
    assert result.model_evaluations == optimizer.force_calls + 1


def test_heterogeneous_batch_matches_independent_line_searches():
    systems = [
        Atoms("H", positions=[[0.35, -0.1, 0.2]]),
        Atoms(
            "H2",
            positions=[[1.2, -0.4, 0.3], [-0.7, 0.8, -0.2]],
        ),
        Atoms(
            "H3",
            positions=[
                [0.8, -0.3, 0.1],
                [-0.4, 0.6, -0.2],
                [0.2, -0.1, 0.5],
            ],
        ),
    ]
    calculator = _quadratic_potential()
    batched = batched_bfgs_line_search_relax(
        calculator.create_state(systems),
        calculator,
        fmax=1e-30,
        max_steps=5,
    )
    independent = [_run_independent(system, steps=5) for system in systems]

    expected_positions = torch.cat(
        [result.state.positions for result in independent]
    )
    expected_energy = torch.cat(
        [result.evaluation.energy for result in independent]
    )
    torch.testing.assert_close(batched.state.positions, expected_positions)
    torch.testing.assert_close(batched.evaluation.energy, expected_energy)


def test_active_compaction_preserves_line_search_results():
    systems = [
        Atoms("H", positions=[[1e-8, 0.0, 0.0]]),
        Atoms("H2", positions=[[0.4, 0.0, 0.0], [-0.2, 0.1, 0.0]]),
        Atoms(
            "H3",
            positions=[[1.5, -0.4, 0.2], [-0.8, 1.0, 0.0], [0.3, -0.2, 0.7]],
        ),
    ]

    def run(*, compact: bool):
        calculator = _quadratic_potential()
        return batched_bfgs_line_search_relax(
            calculator.create_state(systems),
            calculator,
            fmax=1e-5,
            max_steps=20,
            active_compaction=compact,
        )

    masked = run(compact=False)
    active = run(compact=True)

    assert bool(masked.converged.all())
    assert bool(active.converged.all())
    torch.testing.assert_close(active.state.positions, masked.state.positions)
    torch.testing.assert_close(active.evaluation.energy, masked.evaluation.energy)
    torch.testing.assert_close(active.converged_step, masked.converged_step)
    assert active.model_evaluations == masked.model_evaluations
    assert active.graph_evaluations < masked.graph_evaluations


def test_variable_cell_b1_matches_ase_bfgs_line_search():
    initial = bulk("Ar", "fcc", a=5.0, cubic=True)
    reference = initial.copy()
    reference.calc = LennardJones(sigma=3.4, epsilon=0.0103, rc=8.5)
    optimizer = BFGSLineSearch(
        ASEFrechetCellFilter(reference, hydrostatic_strain=True),
        logfile=None,
        alpha=10.0,
        maxstep=0.2,
    )
    optimizer.run(fmax=1e-30, steps=4)

    calculator = ASECalculatorAdapter(
        LennardJones(sigma=3.4, epsilon=0.0103, rc=8.5)
    )
    result = batched_bfgs_line_search_relax(
        calculator.create_state([initial]),
        calculator,
        cell_filter=FrechetCellFilter(hydrostatic_strain=True),
        fmax=1e-30,
        smax=None,
        max_steps=4,
        alpha=10.0,
        max_step=0.2,
    )

    assert result.steps == optimizer.nsteps
    np.testing.assert_allclose(
        result.state.cells[0].cpu(), reference.cell.array, atol=7e-12
    )
    np.testing.assert_allclose(
        result.state.positions.cpu(), reference.positions, atol=4e-12
    )
