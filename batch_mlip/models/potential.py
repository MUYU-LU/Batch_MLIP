"""Graph-MLIP adapter for batched energies, forces, and stress."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Literal

import torch

from ..core.calculator import BatchCalculator, NeighborPolicy
from ..core.math_utils import model_dtype, scatter_sum
from ..core.neighbors import NeighborBackend
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation
from ..profiling.runtime import profile_event, profile_phase

ForceMode = Literal["auto", "autograd", "direct"]


class AtomBitBatchCalculator(BatchCalculator):
    """Wrap an AtomBit-like graph model for batched atomistic simulation.

    Supported model outputs:

    * energy tensor with shape ``[B]`` or ``[B, 1]``;
    * dictionary containing ``energy`` and optionally ``force``/``forces``.

    ``force_mode='autograd'`` is recommended for conservative NVE dynamics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        force_mode: ForceMode = "autograd",
        e0_dict: Mapping[int, float] | None = None,
        model_call_kwargs: Mapping[str, object] | None = None,
        cutoff: float | None = None,
        skin: float = 0.0,
        neighbor_backend: NeighborBackend = "auto",
    ) -> None:
        if force_mode not in ("auto", "autograd", "direct"):
            raise ValueError(f"unsupported force_mode={force_mode!r}")

        resolved_dtype = model_dtype(model) if dtype is None else dtype
        super().__init__(
            cutoff=cutoff,
            skin=skin,
            device=device,
            dtype=resolved_dtype,
            neighbor_backend=neighbor_backend,
        )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.force_mode = force_mode
        self.e0_dict: dict[int, float] = (
            {} if e0_dict is None else {int(k): float(v) for k, v in e0_dict.items()}
        )
        self.model_call_kwargs = dict(model_call_kwargs or {})

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _e0_per_system(self, state: AseGraphBatch) -> torch.Tensor:
        if not self.e0_dict:
            return torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)

        offset_dtype = torch.float64 if state.dtype == torch.float32 else state.dtype
        atom_offsets = torch.zeros(state.n_atoms, device=state.device, dtype=offset_dtype)
        for atomic_number, energy in self.e0_dict.items():
            atom_offsets = torch.where(
                state.z == atomic_number,
                torch.as_tensor(energy, device=state.device, dtype=offset_dtype),
                atom_offsets,
            )
        return scatter_sum(atom_offsets, state.system_idx, state.n_systems)

    @staticmethod
    def _update_neighbors(state: AseGraphBatch, policy: NeighborPolicy) -> None:
        if policy == "always":
            state.rebuild_neighbor_list()
        elif policy == "auto":
            state.ensure_neighbor_list()
        elif policy != "never":
            raise ValueError(f"unsupported neighbor policy {policy!r}")

    def calculate(
        self,
        state: AseGraphBatch,
        *,
        neighbor_policy: NeighborPolicy = "auto",
        compute_stress: bool = False,
    ) -> BatchEvaluation:
        if state.device != self.device:
            raise ValueError(f"state is on {state.device}, model is on {self.device}")
        if state.dtype != self.dtype:
            raise ValueError(f"state dtype is {state.dtype}, model dtype is {self.dtype}")

        rebuilds_before = state.neighbor_rebuild_count
        with profile_phase(
            "calculator.neighbor_update",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
        ):
            self._update_neighbors(state, neighbor_policy)
            state.assert_graph_integrity()
        rebuilt = state.neighbor_rebuild_count - rebuilds_before

        # ``auto`` needs a position graph because direct-force availability is
        # known only after the forward call.
        need_position_grad = self.force_mode in ("auto", "autograd")
        base_positions = state.positions.detach().requires_grad_(need_position_grad)
        base_cells = state.cells.detach()

        strain = None
        model_positions = base_positions
        model_cells = base_cells
        if compute_stress:
            strain = torch.zeros(
                (state.n_systems, 3, 3),
                device=state.device,
                dtype=state.dtype,
                requires_grad=True,
            )
            sym_strain = 0.5 * (strain + strain.transpose(-1, -2))
            atom_strain = sym_strain[state.system_idx]
            displacement = torch.bmm(
                base_positions.unsqueeze(1), atom_strain.transpose(1, 2)
            ).squeeze(1)
            model_positions = base_positions + displacement
            model_cells = base_cells + torch.bmm(base_cells, sym_strain)

        with profile_phase(
            "calculator.graph_view",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
            candidate_edges=state.edge_index.shape[1],
        ):
            data = state.as_model_data(positions=model_positions, cells=model_cells)
        no_grad_ok = self.force_mode == "direct" and not compute_stress
        context = torch.no_grad() if no_grad_ok else nullcontext()
        with profile_phase(
            "model.forward",
            device=state.device,
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=data.edge_index.shape[1],
        ):
            with context:
                raw = self.model(data, **self.model_call_kwargs)

        direct_forces = None
        if isinstance(raw, dict):
            if "energy" not in raw:
                raise KeyError("model output dictionary must contain an 'energy' tensor")
            model_energy = raw["energy"]
            direct_forces = raw.get("forces", raw.get("force"))
        else:
            model_energy = raw

        if not isinstance(model_energy, torch.Tensor):
            raise TypeError("model energy output must be a torch.Tensor")
        if model_energy.numel() != state.n_systems:
            raise ValueError(
                "model must return one total energy per graph; "
                f"expected {state.n_systems} values, got shape {tuple(model_energy.shape)}"
            )
        model_energy = model_energy.reshape(state.n_systems)

        if self.force_mode == "direct":
            if direct_forces is None:
                raise RuntimeError("force_mode='direct' requested, but model returned no forces")
            use_direct_forces = True
        elif self.force_mode == "auto":
            use_direct_forces = direct_forces is not None
        else:
            use_direct_forces = False

        grad_targets: list[torch.Tensor] = []
        if not use_direct_forces:
            grad_targets.append(base_positions)
        if strain is not None:
            grad_targets.append(strain)

        gradients: tuple[torch.Tensor, ...] = ()
        if grad_targets:
            with profile_phase(
                "model.autograd",
                device=state.device,
                systems=state.n_systems,
                atoms=state.n_atoms,
                edges=data.edge_index.shape[1],
            ):
                gradients = torch.autograd.grad(
                    model_energy.sum(),
                    grad_targets,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )

        gradient_idx = 0
        if use_direct_forces:
            if not isinstance(direct_forces, torch.Tensor):
                raise TypeError("direct forces must be a torch.Tensor")
            if direct_forces.numel() != state.n_atoms * 3:
                raise ValueError(
                    f"direct forces must contain {state.n_atoms * 3} values, "
                    f"got shape {tuple(direct_forces.shape)}"
                )
            forces = direct_forces.reshape(state.n_atoms, 3)
        else:
            forces = -gradients[gradient_idx]
            gradient_idx += 1

        stress = None
        if strain is not None:
            d_energy_d_strain = gradients[gradient_idx]
            volume = torch.abs(torch.linalg.det(base_cells)).clamp_min(1e-12)
            stress = d_energy_d_strain / volume.view(-1, 1, 1)
            nonperiodic = ~state.pbc.any(dim=1)
            if bool(nonperiodic.any()):
                stress = stress.masked_fill(nonperiodic.view(-1, 1, 1), torch.nan)

        e0 = self._e0_per_system(state)
        total_energy = model_energy.to(e0.dtype) + e0
        profile_event(
            "model_evaluation",
            adapter="atombit",
            systems=state.n_systems,
            atoms=state.n_atoms,
            edges=data.edge_index.shape[1],
            candidate_edges=state.edge_index.shape[1],
            neighbor_rebuilds=rebuilt,
            compute_stress=compute_stress,
        )
        return BatchEvaluation(
            energy=total_energy.detach(),
            forces=forces.detach(),
            stress=None if stress is None else stress.detach(),
        )


# Public compatibility alias retained for existing scripts and checkpoints.
BatchedPotential = AtomBitBatchCalculator


def load_atombit_batch(
    model_factory: str,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = torch.float32,
    force_mode: ForceMode = "autograd",
    e0: str | Mapping[int, float] | None = None,
    model_call_kwargs: Mapping[str, object] | None = None,
    cutoff: float | None = None,
    skin: float = 0.0,
    neighbor_backend: NeighborBackend = "auto",
) -> AtomBitBatchCalculator:
    """Construct a generic AtomBit-style calculator from a model factory."""

    from .loaders import build_model, infer_cutoff, load_e0, parse_dtype

    model = build_model(model_factory, model_kwargs)
    return AtomBitBatchCalculator(
        model,
        device=device,
        dtype=parse_dtype(dtype),
        force_mode=force_mode,
        e0_dict=load_e0(e0),
        model_call_kwargs=model_call_kwargs,
        cutoff=infer_cutoff(model, cutoff),
        skin=skin,
        neighbor_backend=neighbor_backend,
    )
