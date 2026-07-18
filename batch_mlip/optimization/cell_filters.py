"""Cell degrees of freedom for batched geometry optimization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..core.math_utils import as_system_parameter
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation

GPA_TO_EV_PER_A3 = 1.0 / 160.21766208


def _cell_mask(
    mask: Sequence[bool] | torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((3, 3), device=device, dtype=torch.bool)
    tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if tensor.shape == (6,):
        xx, yy, zz, yz, xz, xy = tensor.unbind()
        tensor = torch.stack(
            (
                torch.stack((xx, xy, xz)),
                torch.stack((xy, yy, yz)),
                torch.stack((xz, yz, zz)),
            )
        )
    if tensor.shape != (3, 3):
        raise ValueError("cell mask must have shape [6] or [3, 3]")
    if not torch.equal(tensor, tensor.T):
        raise ValueError("cell mask must be symmetric")
    return tensor


@dataclass(frozen=True)
class FrechetCellFilter:
    """ASE-style log-deformation cell coordinates for a heterogeneous batch.

    Pressure is positive in compression. Cell forces minimize ``E + pV``.
    The initial implementation supports full-rank, fully periodic cells and no
    atomic constraints; fixed-cell optimization retains its existing support.
    """

    pressure_GPa: float | Sequence[float] | torch.Tensor = 0.0
    mask: Sequence[bool] | torch.Tensor | None = None
    cell_factor: float | Sequence[float] | torch.Tensor | None = None
    hydrostatic_strain: bool = False

    def bind(
        self,
        state: AseGraphBatch,
        *,
        dtype: torch.dtype | None = None,
    ) -> BoundFrechetCellFilter:
        return BoundFrechetCellFilter.from_config(state, self, dtype=dtype)


@dataclass
class BoundFrechetCellFilter:
    """Optimizer state associated with one bound batch."""

    reference_cells: torch.Tensor
    generalized_positions: torch.Tensor
    log_deformation: torch.Tensor
    cell_factor: torch.Tensor
    pressure: torch.Tensor
    mask: torch.Tensor
    hydrostatic_strain: bool

    @classmethod
    def from_config(
        cls,
        state: AseGraphBatch,
        config: FrechetCellFilter,
        *,
        dtype: torch.dtype | None = None,
    ) -> BoundFrechetCellFilter:
        if not bool(state.pbc.all()):
            raise ValueError(
                "variable-cell optimization currently requires fully periodic systems"
            )
        determinant = torch.linalg.det(state.cells)
        if bool((determinant <= 0.0).any()):
            raise ValueError(
                "variable-cell optimization requires right-handed full-rank cells"
            )
        if bool(state.fixed.any()):
            raise NotImplementedError(
                "FixAtoms with variable-cell optimization is not implemented"
            )

        optimizer_dtype = state.dtype if dtype is None else dtype
        pressure_gpa = as_system_parameter(
            config.pressure_GPa,
            n_systems=state.n_systems,
            device=state.device,
            dtype=optimizer_dtype,
            name="pressure_GPa",
        )
        if config.cell_factor is None:
            factor = state.counts.to(dtype=optimizer_dtype)
        else:
            factor = as_system_parameter(
                config.cell_factor,
                n_systems=state.n_systems,
                device=state.device,
                dtype=optimizer_dtype,
                name="cell_factor",
            )
        if bool((factor <= 0.0).any()):
            raise ValueError("cell_factor must be positive")

        return cls(
            reference_cells=state.cells.detach().to(optimizer_dtype).clone(),
            generalized_positions=(
                state.positions.detach().to(optimizer_dtype).clone()
            ),
            log_deformation=torch.zeros(
                (state.n_systems, 3, 3),
                device=state.device,
                dtype=optimizer_dtype,
            ),
            cell_factor=factor,
            pressure=pressure_gpa * GPA_TO_EV_PER_A3,
            mask=_cell_mask(config.mask, device=state.device),
            hydrostatic_strain=config.hydrostatic_strain,
        )

    @property
    def deformation(self) -> torch.Tensor:
        return torch.matrix_exp(self.log_deformation)

    def select_systems(
        self,
        state: AseGraphBatch,
        system_ids: Sequence[int],
    ) -> BoundFrechetCellFilter:
        """Select optimizer state in the same order as a compacted graph batch."""

        ids = [int(i) for i in system_ids]
        if not ids:
            raise ValueError("system_ids must not be empty")
        graph_ids = torch.as_tensor(ids, device=state.device, dtype=torch.long)
        atom_blocks = [
            torch.arange(
                state.ptr[i], state.ptr[i + 1], device=state.device, dtype=torch.long
            )
            for i in ids
        ]
        atom_ids = torch.cat(atom_blocks)
        return BoundFrechetCellFilter(
            reference_cells=self.reference_cells[graph_ids].clone(),
            generalized_positions=self.generalized_positions[atom_ids].clone(),
            log_deformation=self.log_deformation[graph_ids].clone(),
            cell_factor=self.cell_factor[graph_ids].clone(),
            pressure=self.pressure[graph_ids].clone(),
            mask=self.mask,
            hydrostatic_strain=self.hydrostatic_strain,
        )

    def current_cells(self) -> torch.Tensor:
        return torch.bmm(
            self.reference_cells, self.deformation.transpose(-1, -2)
        )

    def volumes(self, state: AseGraphBatch) -> torch.Tensor:
        del state
        return torch.linalg.det(self.current_cells()).abs()

    def stress_residual(self, evaluation: BatchEvaluation) -> torch.Tensor:
        if evaluation.stress is None:
            raise RuntimeError(
                "cell-filter optimization requires calculator stress"
            )
        stress = evaluation.stress.to(dtype=self.reference_cells.dtype)
        identity = torch.eye(3, device=stress.device, dtype=stress.dtype)
        residual = stress + self.pressure[:, None, None] * identity
        if self.hydrostatic_strain:
            mean = torch.diagonal(residual, dim1=-2, dim2=-1).mean(dim=-1)
            residual = mean[:, None, None] * identity
        return residual.masked_fill(~self.mask, 0.0)

    def max_stress(self, evaluation: BatchEvaluation) -> torch.Tensor:
        residual = self.stress_residual(evaluation)
        return residual.abs().flatten(start_dim=1).amax(dim=1)

    def generalized_forces(
        self,
        state: AseGraphBatch,
        evaluation: BatchEvaluation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deformation = self.deformation.detach()
        atom_deformation = deformation[state.system_idx]
        atomic_forces = torch.bmm(
            evaluation.forces.to(deformation.dtype).unsqueeze(1), atom_deformation
        ).squeeze(1)

        virial = -self.volumes(state)[:, None, None] * self.stress_residual(
            evaluation
        )
        unit_cell_force = torch.bmm(
            virial, torch.linalg.inv(deformation.transpose(-1, -2))
        )

        # Apply the adjoint Frechet derivative of exp(U) without depending on
        # the model graph. This is the exact force conjugate to log strain U.
        with torch.enable_grad():
            log_deformation = self.log_deformation.detach().requires_grad_(True)
            mapped = torch.matrix_exp(log_deformation)
            log_force = torch.autograd.grad(
                mapped,
                log_deformation,
                grad_outputs=unit_cell_force,
                create_graph=False,
            )[0]
        log_force = log_force.masked_fill(~self.mask, 0.0)
        if self.hydrostatic_strain:
            identity = torch.eye(
                3, device=state.device, dtype=log_force.dtype
            ).expand(state.n_systems, -1, -1)
            trace = torch.diagonal(log_force, dim1=-2, dim2=-1).sum(dim=-1)
            log_force = trace[:, None, None] * identity / 3.0
        return atomic_forces.detach(), (
            log_force / self.cell_factor[:, None, None]
        ).detach()

    def apply_displacement(
        self,
        state: AseGraphBatch,
        atomic_displacement: torch.Tensor,
        cell_displacement: torch.Tensor,
    ) -> None:
        self.generalized_positions = (
            self.generalized_positions + atomic_displacement
        ).detach()
        delta_log = cell_displacement / self.cell_factor[:, None, None]
        delta_log = delta_log.masked_fill(~self.mask, 0.0)
        if self.hydrostatic_strain:
            identity = torch.eye(
                3, device=state.device, dtype=delta_log.dtype
            ).expand(state.n_systems, -1, -1)
            trace = torch.diagonal(delta_log, dim1=-2, dim2=-1).sum(dim=-1)
            delta_log = trace[:, None, None] * identity / 3.0
        self.log_deformation = (self.log_deformation + delta_log).detach()

        deformation = self.deformation.detach()
        state.cells = self.current_cells().to(dtype=state.dtype).detach()
        atom_deformation = deformation[state.system_idx]
        state.positions = (
            torch.bmm(
                self.generalized_positions.unsqueeze(1),
                atom_deformation.transpose(-1, -2),
            )
            .squeeze(1)
            .to(dtype=state.dtype)
            .detach()
        )
        state._neighbor_reference_positions = None
        state._neighbor_reference_cells = None


# Public compatibility alias retained for the pre-0.2 API.
BatchedFrechetCellFilter = FrechetCellFilter
