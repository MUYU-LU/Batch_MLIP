"""Explicit ordinary-ASE relaxation reference path."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, FIRE, BFGSLineSearch

from ..core.calculator import ASECalculatorAdapter
from ..core.types import BatchEvaluation, RelaxationResult

ASE_OPTIMIZERS = {
    "bfgs": BFGS,
    "bfgslinesearch": BFGSLineSearch,
    "bfgs_line_search": BFGSLineSearch,
    "fire": FIRE,
    "quasinewton": BFGSLineSearch,
}


def relax_ase(
    systems: Atoms | Sequence[Atoms],
    calculator: Calculator,
    *,
    optimizer: str | Callable[..., Any] = "bfgs",
    cell_filter: Callable[[Atoms], Any] | None = None,
    fmax: float = 0.05,
    max_steps: int = 500,
    **optimizer_kwargs: Any,
) -> RelaxationResult:
    """Relax serially with native ASE objects as an explicit reference mode."""

    normalized = [systems] if isinstance(systems, Atoms) else list(systems)
    if not normalized or not all(isinstance(atoms, Atoms) for atoms in normalized):
        raise TypeError("systems must contain at least one ASE Atoms object")
    if not isinstance(calculator, Calculator):
        raise TypeError("relax_ase requires an ordinary ASE Calculator")
    if isinstance(optimizer, str):
        key = optimizer.strip().lower().replace("-", "_")
        try:
            optimizer_factory = ASE_OPTIMIZERS[key]
        except KeyError as exc:
            choices = ", ".join(sorted(ASE_OPTIMIZERS))
            raise ValueError(
                f"unknown ASE optimizer {optimizer!r}; choose one of: {choices}"
            ) from exc
    elif callable(optimizer):
        optimizer_factory = optimizer
    else:
        raise TypeError("optimizer must be an ASE optimizer name or callable")

    options = {"logfile": None, "trajectory": None, **optimizer_kwargs}
    final_structures = []
    energies = []
    force_blocks = []
    stress_blocks = []
    converged_values = []
    converged_steps = []
    steps_taken = []
    max_forces = []
    max_stresses = []
    for source in normalized:
        atoms = source.copy()
        atoms.calc = calculator
        target = atoms if cell_filter is None else cell_filter(atoms)
        ase_optimizer = optimizer_factory(target, **options)
        converged = bool(ase_optimizer.run(fmax=fmax, steps=max_steps))
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        energy = float(atoms.get_potential_energy())
        final_structures.append(atoms)
        energies.append(energy)
        force_blocks.append(forces)
        converged_values.append(converged)
        steps = int(ase_optimizer.nsteps)
        steps_taken.append(steps)
        converged_steps.append(steps if converged else -1)
        max_forces.append(float(np.linalg.norm(forces, axis=1).max()))
        if cell_filter is not None:
            stress = np.asarray(
                atoms.get_stress(voigt=False),
                dtype=np.float64,
            )
            stress_blocks.append(stress)
            max_stresses.append(float(np.abs(stress).max()))

    adapter = ASECalculatorAdapter(calculator)
    state = adapter.create_state(final_structures, build_neighbors=False)
    evaluation = BatchEvaluation(
        energy=torch.as_tensor(energies, dtype=torch.float64),
        forces=torch.as_tensor(
            np.concatenate(force_blocks, axis=0),
            dtype=torch.float64,
        ),
        stress=(
            torch.as_tensor(np.stack(stress_blocks), dtype=torch.float64)
            if cell_filter is not None
            else None
        ),
    )
    return RelaxationResult(
        state=state,
        evaluation=evaluation,
        converged=torch.as_tensor(converged_values, dtype=torch.bool),
        converged_step=torch.as_tensor(converged_steps, dtype=torch.long),
        max_force=torch.as_tensor(max_forces, dtype=torch.float64),
        max_stress=(
            torch.as_tensor(max_stresses, dtype=torch.float64)
            if cell_filter is not None
            else None
        ),
        steps=max(steps_taken),
        metadata={
            "execution": "strict_ase_serial",
            "model_evaluations": "not instrumented",
        },
    )
