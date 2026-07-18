"""Core data structures for batched atomistic simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GraphData:
    """Minimal PyG-like attribute container accepted by AtomBitModel.

    The uploaded model only reads attributes from its ``data`` argument, so a
    full :mod:`torch_geometric` dependency is unnecessary for inference.
    """

    z: torch.Tensor
    pos: torch.Tensor
    cell: torch.Tensor | None
    edge_index: torch.Tensor
    shifts_int: torch.Tensor
    batch: torch.Tensor
    num_graphs: int

    def to(self, *args: Any, **kwargs: Any) -> GraphData:
        """Return a copy with every tensor moved/cast like ``Tensor.to``."""

        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.to(*args, **kwargs)

        return GraphData(
            z=move(self.z),
            pos=move(self.pos),
            cell=move(self.cell),
            edge_index=move(self.edge_index),
            shifts_int=move(self.shifts_int),
            batch=move(self.batch),
            num_graphs=self.num_graphs,
        )


@dataclass
class BatchEvaluation:
    """Potential result for one heterogeneous graph batch."""

    energy: torch.Tensor  # [B], eV
    forces: torch.Tensor  # [N, 3], eV / Angstrom
    stress: torch.Tensor | None = None  # [B, 3, 3], eV / Angstrom^3


@dataclass
class EvaluationResult:
    """State and predictions returned by the structure-level API."""

    state: Any
    evaluation: BatchEvaluation

    @property
    def structures(self) -> list[Any]:
        """Return ASE structures carrying single-point energy/force results."""

        return self.state.to_ase(self.evaluation, wrap=False)


@dataclass
class RelaxationResult:
    """Final state and convergence information from a batched relaxation."""

    state: Any
    evaluation: BatchEvaluation
    converged: torch.Tensor  # [B], bool
    converged_step: torch.Tensor  # [B], -1 means not converged
    max_force: torch.Tensor  # [B], eV / Angstrom
    steps: int
    max_stress: torch.Tensor | None = None  # [B], eV / Angstrom^3
    model_evaluations: int = 0
    graph_evaluations: int = 0
    active_batch_sizes: tuple[int, ...] = ()

    @property
    def structures(self) -> list[Any]:
        """Return relaxed ASE structures carrying final predictions."""

        return self.state.to_ase(self.evaluation, wrap=False)


@dataclass
class MDResult:
    """Final state and thermodynamic summary from a batched MD run."""

    state: Any
    evaluation: BatchEvaluation
    steps: int
    kinetic_energy: torch.Tensor
    temperature: torch.Tensor

    @property
    def structures(self) -> list[Any]:
        """Return final ASE structures carrying final predictions."""

        return self.state.to_ase(self.evaluation, wrap=False)


StepCallback = Callable[
    [int, Any, BatchEvaluation, dict[str, torch.Tensor]],
    None,
]
