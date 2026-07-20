from __future__ import annotations

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from batch_mlip.toy_models import QuadraticWellModel

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator, batched_fire_relax


class QuadraticCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.5 * float((atoms.positions**2).sum())
        self.results["forces"] = -atoms.positions.copy()


def test_fire_converges_multiple_heterogeneous_systems():
    systems = [
        Atoms("H", positions=[[1.0, -1.5, 0.5]]),
        Atoms("H2", positions=[[0.7, 0.2, 0.0], [-0.4, 1.1, 0.3]]),
    ]
    state = AseGraphBatch.from_ase(
        systems, cutoff=2.5, skin=0.3, device="cpu", dtype=torch.float64
    )
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0), device="cpu", dtype=torch.float64
    )
    result = batched_fire_relax(
        state,
        potential,
        fmax=1e-5,
        max_steps=1000,
        dt_start=0.05,
        dt_max=0.5,
        max_step=0.2,
    )
    assert bool(result.converged.all())
    assert float(torch.linalg.vector_norm(result.state.positions, dim=-1).max()) < 2e-5


def test_compacted_float32_fire_accepts_float64_energy_offsets():
    calculator = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0),
        cutoff=2.5,
        device="cpu",
        dtype=torch.float32,
        e0_dict={1: -13.605693122994},
    )
    state = calculator.create_state([Atoms("H", positions=[[0.1, 0.0, 0.0]])])

    result = batched_fire_relax(
        state, calculator, fmax=1e-30, max_steps=0, active_compaction=True
    )

    assert result.evaluation.energy.dtype == torch.float64


def test_fixatoms_position_is_unchanged():
    atoms = Atoms("H2", positions=[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    state = AseGraphBatch.from_ase(
        [atoms], cutoff=3.0, device="cpu", dtype=torch.float64
    )
    initial_fixed = state.positions[0].clone()
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(), device="cpu", dtype=torch.float64
    )
    result = batched_fire_relax(state, potential, fmax=1e-5, max_steps=1000)
    torch.testing.assert_close(result.state.positions[0], initial_fixed)
    assert torch.linalg.vector_norm(result.state.positions[1]) < 2e-5


def test_batched_fire_matches_ase_update_order():
    initial = Atoms(
        "H2",
        positions=[[1.2, -0.4, 0.3], [-0.7, 0.8, -0.2]],
    )
    ase_atoms = initial.copy()
    ase_atoms.calc = QuadraticCalculator()
    ase_fire = FIRE(ase_atoms, logfile=None, dt=0.05, dtmax=0.5, maxstep=0.2)
    ase_fire.run(fmax=1e-30, steps=15)

    state = AseGraphBatch.from_ase(
        [initial], cutoff=2.5, device="cpu", dtype=torch.float64
    )
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0), device="cpu", dtype=torch.float64
    )
    result = batched_fire_relax(
        state,
        potential,
        fmax=1e-30,
        max_steps=15,
        dt_start=0.05,
        dt_max=0.5,
        max_step=0.2,
    )

    torch.testing.assert_close(
        result.state.positions,
        torch.as_tensor(ase_atoms.positions, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_active_compaction_matches_masked_fire_and_reduces_graph_work():
    systems = [
        Atoms("H", positions=[[1e-8, 0.0, 0.0]]),
        Atoms("H2", positions=[[0.4, 0.0, 0.0], [-0.2, 0.1, 0.0]]),
        Atoms(
            "H3",
            positions=[[1.5, -0.4, 0.2], [-0.8, 1.0, 0.0], [0.3, -0.2, 0.7]],
        ),
    ]
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.0), device="cpu", dtype=torch.float64
    )

    def relax(*, compact: bool):
        state = AseGraphBatch.from_ase(
            systems, cutoff=2.5, skin=0.0, device="cpu", dtype=torch.float64
        )
        return batched_fire_relax(
            state,
            potential,
            fmax=1e-5,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
            max_step=0.2,
            active_compaction=compact,
        )

    masked = relax(compact=False)
    compacted = relax(compact=True)

    assert bool(masked.converged.all())
    assert bool(compacted.converged.all())
    torch.testing.assert_close(compacted.state.positions, masked.state.positions)
    torch.testing.assert_close(compacted.evaluation.energy, masked.evaluation.energy)
    torch.testing.assert_close(compacted.evaluation.forces, masked.evaluation.forces)
    torch.testing.assert_close(compacted.converged_step, masked.converged_step)
    assert compacted.model_evaluations == masked.model_evaluations
    assert compacted.graph_evaluations < masked.graph_evaluations
    assert compacted.active_batch_sizes[0] == len(systems)
    assert compacted.active_batch_sizes[-1] == 1
