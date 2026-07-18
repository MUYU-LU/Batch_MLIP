from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.optimize import BFGS
from batch_mlip.toy_models import QuadraticWellModel

from batch_mlip import (
    ASECalculatorAdapter,
    AtomBitBatchCalculator,
    BatchedBFGS,
    FrechetCellFilter,
    available_optimizers,
    batched_bfgs_relax,
    create_optimizer,
    relax,
)


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


def test_bfgs_is_registered_and_constructible():
    assert "bfgs" in available_optimizers()
    assert isinstance(create_optimizer("BFGS"), BatchedBFGS)
    result = relax(
        Atoms("H", positions=[[0.1, 0.0, 0.0]]),
        _quadratic_potential(),
        optimizer="bfgs",
        fmax=1e-30,
        max_steps=0,
    )
    assert result.steps == 0


def test_float32_bfgs_promotes_frechet_optimizer_state():
    atoms = bulk("Ar", "fcc", a=5.5, cubic=True)
    calculator = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0),
        cutoff=2.5,
        device="cpu",
        dtype=torch.float32,
    )
    state = calculator.create_state([atoms])
    bound = FrechetCellFilter().bind(state, dtype=torch.float64)
    assert bound.reference_cells.dtype == torch.float64
    assert bound.generalized_positions.dtype == torch.float64
    assert bound.log_deformation.dtype == torch.float64
    assert state.positions.dtype == torch.float32
    result = batched_bfgs_relax(
        state,
        calculator,
        cell_filter=FrechetCellFilter(),
        fmax=1e-30,
        smax=None,
        max_steps=1,
    )
    assert result.steps == 1
    assert result.state.positions.dtype == torch.float32


def test_fixed_cell_bfgs_matches_ase_update_order():
    initial = Atoms(
        "H2",
        positions=[[1.2, -0.4, 0.3], [-0.7, 0.8, -0.2]],
    )
    reference = initial.copy()
    reference.calc = QuadraticCalculator()
    optimizer = BFGS(
        reference,
        logfile=None,
        alpha=70.0,
        maxstep=0.2,
    )
    optimizer.run(fmax=1e-30, steps=10)

    calculator = _quadratic_potential()
    result = batched_bfgs_relax(
        calculator.create_state([initial]),
        calculator,
        fmax=1e-30,
        max_steps=10,
        alpha=70.0,
        max_step=0.2,
    )

    assert result.steps == optimizer.nsteps
    torch.testing.assert_close(
        result.state.positions,
        torch.as_tensor(reference.positions, dtype=torch.float64),
        atol=2e-12,
        rtol=2e-12,
    )


def test_bfgs_matches_ase_with_fixatoms():
    initial = Atoms(
        "H2",
        positions=[[1.0, 0.0, 0.0], [0.7, -0.4, 0.2]],
        constraint=FixAtoms(indices=[0]),
    )
    reference = initial.copy()
    reference.calc = QuadraticCalculator()
    optimizer = BFGS(reference, logfile=None, alpha=70.0, maxstep=0.2)
    optimizer.run(fmax=1e-30, steps=8)

    calculator = _quadratic_potential()
    result = batched_bfgs_relax(
        calculator.create_state([initial]),
        calculator,
        fmax=1e-30,
        max_steps=8,
    )

    torch.testing.assert_close(
        result.state.positions,
        torch.as_tensor(reference.positions, dtype=torch.float64),
        atol=2e-12,
        rtol=2e-12,
    )
    torch.testing.assert_close(
        result.state.positions[0],
        torch.as_tensor(initial.positions[0], dtype=torch.float64),
    )


def test_active_bfgs_matches_masked_and_compacts_hessian_state():
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
        return batched_bfgs_relax(
            calculator.create_state(systems),
            calculator,
            fmax=1e-5,
            max_steps=100,
            active_compaction=compact,
        )

    masked = run(compact=False)
    active = run(compact=True)

    assert bool(masked.converged.all())
    assert bool(active.converged.all())
    torch.testing.assert_close(active.state.positions, masked.state.positions)
    torch.testing.assert_close(active.evaluation.energy, masked.evaluation.energy)
    torch.testing.assert_close(active.evaluation.forces, masked.evaluation.forces)
    torch.testing.assert_close(active.converged_step, masked.converged_step)
    assert active.model_evaluations == masked.model_evaluations
    assert active.graph_evaluations < masked.graph_evaluations
    assert active.active_batch_sizes[0] == 3
    assert active.active_batch_sizes[1] == 2
    assert active.active_batch_sizes[-1] == 1


def _ase_frechet_bfgs(atoms: Atoms) -> tuple[Atoms, int]:
    reference = atoms.copy()
    reference.calc = LennardJones(sigma=3.4, epsilon=0.0103, rc=8.5)
    optimizer = BFGS(
        ASEFrechetCellFilter(reference, hydrostatic_strain=True),
        logfile=None,
        alpha=70.0,
        maxstep=0.2,
    )
    optimizer.run(fmax=2e-5, steps=200)
    return reference, optimizer.nsteps


def test_variable_cell_active_bfgs_matches_masked_and_ase():
    systems = [
        bulk("Ar", "fcc", a=a, cubic=True)
        for a in (5.2686752, 5.0, 6.2)
    ]
    references = [_ase_frechet_bfgs(atoms) for atoms in systems]

    def run(*, compact: bool):
        calculator = ASECalculatorAdapter(
            LennardJones(sigma=3.4, epsilon=0.0103, rc=8.5)
        )
        return batched_bfgs_relax(
            calculator.create_state(systems),
            calculator,
            cell_filter=FrechetCellFilter(hydrostatic_strain=True),
            active_compaction=compact,
            fmax=2e-5,
            smax=None,
            max_steps=200,
            alpha=70.0,
            max_step=0.2,
        )

    masked = run(compact=False)
    active = run(compact=True)

    assert active.converged_step.tolist() == [0, 6, 8]
    assert active.converged_step.tolist() == [steps for _, steps in references]
    torch.testing.assert_close(active.state.positions, masked.state.positions)
    torch.testing.assert_close(active.state.cells, masked.state.cells)
    torch.testing.assert_close(active.evaluation.energy, masked.evaluation.energy)
    torch.testing.assert_close(active.evaluation.forces, masked.evaluation.forces)
    torch.testing.assert_close(active.evaluation.stress, masked.evaluation.stress)
    assert masked.graph_evaluations == 27
    assert active.graph_evaluations == 17
    for system_id, (reference, _) in enumerate(references):
        np.testing.assert_allclose(
            active.state.cells[system_id].cpu(),
            reference.cell.array,
            atol=3e-11,
        )
        atom_slice = active.state.atom_slice(system_id)
        np.testing.assert_allclose(
            active.state.positions[atom_slice].cpu(),
            reference.positions,
            atol=3e-11,
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"max_step": 0.0}, "max_step must be positive"),
        ({"max_steps": -1}, "max_steps must be non-negative"),
        ({"optimizer_dtype": "float16"}, "optimizer_dtype must be"),
    ],
)
def test_bfgs_rejects_invalid_options(kwargs, error):
    calculator = _quadratic_potential()
    with pytest.raises(ValueError, match=error):
        batched_bfgs_relax(
            calculator.create_state([Atoms("H", positions=[[0.1, 0.0, 0.0]])]),
            calculator,
            **kwargs,
        )
