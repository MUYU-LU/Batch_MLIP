from __future__ import annotations

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from batch_mlip import (
    ASECalculatorAdapter,
    BatchCalculator,
    BatchEvaluation,
    evaluate,
    molecular_dynamics,
    relax,
)


class QuadraticBatchCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.5, device="cpu", dtype=torch.float64)
        self.calls = 0

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy
        if compute_stress:
            raise NotImplementedError
        self.calls += 1
        atom_energy = 0.5 * (state.positions * state.positions).sum(dim=-1)
        energy = torch.zeros(
            state.n_systems, device=state.device, dtype=state.dtype
        )
        energy.index_add_(0, state.system_idx, atom_energy)
        return BatchEvaluation(energy=energy, forces=-state.positions.clone())


class QuadraticASECalculator(Calculator):
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


def test_one_generic_batch_calculator_drives_evaluation_fire_and_md():
    systems = [
        Atoms("H", positions=[[0.8, -0.2, 0.1]]),
        Atoms("H2", positions=[[0.3, 0.1, 0.0], [-0.4, 0.2, 0.1]]),
    ]
    calculator = QuadraticBatchCalculator()

    prediction = evaluate(systems, calculator)
    assert prediction.evaluation.energy.shape == (2,)
    assert len(prediction.structures) == 2
    np.testing.assert_allclose(
        prediction.structures[0].get_potential_energy(), 0.345, atol=1e-15
    )

    relaxed = relax(
        systems,
        calculator,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        active_compaction=True,
    )
    assert bool(relaxed.converged.all())
    assert len(relaxed.structures) == 2

    dynamics = molecular_dynamics(
        systems,
        calculator,
        ensemble="nve",
        timestep_fs=0.1,
        n_steps=2,
    )
    assert dynamics.steps == 2
    assert len(dynamics.structures) == 2
    assert calculator.calls > 3


def test_ordinary_ase_calculator_adapter_is_compatible_but_sequential():
    systems = [
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        Atoms("He2", positions=[[0.0, 0.5, 0.0], [0.0, 0.0, -0.2]]),
    ]
    calculator = ASECalculatorAdapter(QuadraticASECalculator())

    result = evaluate(systems, calculator)

    torch.testing.assert_close(
        result.evaluation.energy,
        torch.tensor([0.5, 0.145], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.evaluation.forces,
        -torch.as_tensor(
            np.concatenate([atoms.positions for atoms in systems]),
            dtype=torch.float64,
        ),
    )
