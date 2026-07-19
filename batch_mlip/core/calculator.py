"""Core model-agnostic calculator contract for batched simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from ..profiling.runtime import profile_event, profile_phase
from .state import AseGraphBatch
from .types import BatchEvaluation

NeighborPolicy = Literal["auto", "always", "never"]


class BatchCalculator(ABC):
    """Generic energy/force provider consumed by the batch integrators.

    A model-specific adapter translates :class:`AseGraphBatch` into the MLIP's
    native inputs. FIRE and MD depend only on this contract.
    """

    def __init__(
        self,
        *,
        cutoff: float | None,
        skin: float = 0.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if cutoff is not None and cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if skin < 0.0:
            raise ValueError("skin must be non-negative")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.skin = float(skin)
        self.device = torch.device(device)
        self.dtype = dtype

    def create_state(
        self,
        systems: Sequence[Atoms],
        *,
        build_neighbors: bool = True,
    ) -> AseGraphBatch:
        """Convert ASE structures to the common tensor batch representation.

        ``build_neighbors=False`` creates a lightweight state shell whose
        graph is constructed on its first AtomBit evaluation.
        """

        if self.cutoff is None:
            raise ValueError(
                "this calculator has no cutoff; configure cutoff when constructing it "
                "before using the structure-level API"
            )
        return AseGraphBatch.from_ase(
            systems,
            cutoff=self.cutoff,
            skin=self.skin,
            device=self.device,
            dtype=self.dtype,
            build_neighbors=build_neighbors,
        )

    @abstractmethod
    def calculate(
        self,
        state: AseGraphBatch,
        *,
        neighbor_policy: NeighborPolicy = "auto",
        compute_stress: bool = False,
    ) -> BatchEvaluation:
        """Return one energy per system and forces in concatenated atom order."""

    def __call__(
        self,
        state: AseGraphBatch,
        *,
        neighbor_policy: NeighborPolicy = "auto",
        compute_stress: bool = False,
    ) -> BatchEvaluation:
        return self.calculate(
            state,
            neighbor_policy=neighbor_policy,
            compute_stress=compute_stress,
        )


class ASECalculatorAdapter(BatchCalculator):
    """Use an ordinary ASE calculator through the batch calculator contract.

    This is a compatibility and reference adapter. It evaluates systems
    sequentially and therefore does not provide batched MLIP acceleration.
    """

    def __init__(
        self,
        calculator: Calculator,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        # The common graph state currently requires a positive cutoff. The ASE
        # adapter never consumes its edges, so this value has no physical role.
        super().__init__(cutoff=1.0, skin=0.0, device=device, dtype=dtype)
        self.calculator = calculator

    def calculate(
        self,
        state: AseGraphBatch,
        *,
        neighbor_policy: NeighborPolicy = "auto",
        compute_stress: bool = False,
    ) -> BatchEvaluation:
        if neighbor_policy not in ("auto", "always", "never"):
            raise ValueError(f"unsupported neighbor policy {neighbor_policy!r}")
        if state.device != self.device or state.dtype != self.dtype:
            raise ValueError("state device and dtype must match the calculator")

        with profile_phase(
            "model.ase_sequential",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
        ):
            energies: list[float] = []
            forces: list[np.ndarray] = []
            stresses: list[np.ndarray] = []
            for atoms in state.to_ase(evaluation=None, wrap=False):
                atoms.calc = self.calculator
                energies.append(float(atoms.get_potential_energy()))
                forces.append(np.asarray(atoms.get_forces(), dtype=np.float64))
                if compute_stress:
                    stresses.append(
                        np.asarray(atoms.get_stress(voigt=False), dtype=np.float64)
                    )

        profile_event(
            "model_evaluation",
            adapter="ase",
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=0,
            neighbor_rebuilds=0,
            compute_stress=compute_stress,
        )

        return BatchEvaluation(
            energy=torch.as_tensor(energies, device=state.device, dtype=state.dtype),
            forces=torch.as_tensor(
                np.concatenate(forces, axis=0), device=state.device, dtype=state.dtype
            ),
            stress=(
                None
                if not compute_stress
                else torch.as_tensor(
                    np.stack(stresses, axis=0), device=state.device, dtype=state.dtype
                )
            ),
        )
