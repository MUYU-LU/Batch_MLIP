"""ASE-compatible full BFGS optimization for heterogeneous batches."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from ..core.arena import HeterogeneousResidentArena, SystemSelection
from ..core.calculator import BatchCalculator
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation, RelaxationResult, StepCallback
from ..profiling.runtime import profile_event, profile_phase
from .cell_filters import BoundFrechetCellFilter, FrechetCellFilter
from .fire import max_force_per_system, max_generalized_force_per_system
from .refill import (
    REFILL_POLICIES as _REFILL_POLICIES,
)
from .refill import (
    REFILL_STORAGE_MODES as _REFILL_STORAGE_MODES,
)
from .refill import (
    global_atom_ids as _global_atom_ids,
)
from .refill import (
    refill_insert_count as _refill_insert_count,
)

_REFILL_STORAGE_MODES = _REFILL_STORAGE_MODES | frozenset(("arena",))


@dataclass
class _BFGSHistory:
    hessian: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    forces: torch.Tensor | None = None


def _profile_optimizer_evaluation(
    state: AseGraphBatch,
    *,
    scheduler_step: int,
    pending_systems: int,
) -> None:
    profile_event(
        "optimizer_evaluation",
        optimizer="bfgs",
        scheduler_step=scheduler_step,
        active_systems=state.n_systems,
        active_atoms=state.n_atoms,
        active_edges=state.edge_index.shape[1],
        pending_systems=pending_systems,
    )


def _system_coordinates(
    state: AseGraphBatch,
    system_id: int,
    cell_filter: BoundFrechetCellFilter | None,
    optimizer_positions: torch.Tensor | None,
) -> torch.Tensor:
    atom_slice = state.atom_slice(system_id)
    if cell_filter is None:
        if optimizer_positions is None:
            raise RuntimeError("fixed-cell BFGS optimizer positions are missing")
        return optimizer_positions[atom_slice]
    cell_coordinates = cell_filter.log_deformation[system_id] * cell_filter.cell_factor[system_id]
    return torch.cat(
        (cell_filter.generalized_positions[atom_slice], cell_coordinates),
        dim=0,
    )


def _system_forces(
    state: AseGraphBatch,
    system_id: int,
    atomic_forces: torch.Tensor,
    cell_forces: torch.Tensor | None,
) -> torch.Tensor:
    atom_slice = state.atom_slice(system_id)
    if cell_forces is None:
        return atomic_forces[atom_slice]
    return torch.cat(
        (atomic_forces[atom_slice], cell_forces[system_id]),
        dim=0,
    )


def _prepare_bfgs_step(
    coordinates: torch.Tensor,
    forces: torch.Tensor,
    history: _BFGSHistory,
    *,
    alpha: float,
    max_step: float,
) -> torch.Tensor:
    """Return one ASE-BFGS displacement and update its system history."""

    position_vector = coordinates.flatten()
    force_vector = forces.flatten()
    if history.hessian is None:
        history.hessian = (
            torch.eye(
                position_vector.numel(),
                device=coordinates.device,
                dtype=coordinates.dtype,
            )
            * alpha
        )
    else:
        if history.positions is None or history.forces is None:
            raise RuntimeError("BFGS history is incomplete")
        delta_position = position_vector - history.positions
        if bool(delta_position.abs().max() >= 1e-7):
            delta_force = force_vector - history.forces
            a = torch.dot(delta_position, delta_force)
            hessian_step = history.hessian @ delta_position
            b = torch.dot(delta_position, hessian_step)
            history.hessian = history.hessian - (
                torch.outer(delta_force, delta_force) / a
                + torch.outer(hessian_step, hessian_step) / b
            )

    eigenvalues, eigenvectors = torch.linalg.eigh(history.hessian)
    displacement = eigenvectors @ ((force_vector @ eigenvectors) / eigenvalues.abs())
    displacement = displacement.reshape_as(coordinates)
    max_row_norm = torch.linalg.vector_norm(displacement, dim=1).max()
    if bool(max_row_norm >= max_step):
        displacement = displacement * (max_step / max_row_norm)

    history.positions = position_vector.detach().clone()
    history.forces = force_vector.detach().clone()
    return displacement.detach()


_LINEAR_ALGEBRA_BACKENDS = frozenset(("auto", "cholesky", "grouped", "serial"))


def _use_grouped_linear_algebra(
    backend: str,
    *,
    device_type: str,
    group_size: int,
    dimension: int,
) -> bool:
    if backend == "cholesky" or (backend == "auto" and device_type == "cuda"):
        return True
    if group_size < 2:
        return False
    return backend == "grouped"


def _prepare_grouped_bfgs_steps(
    coordinates: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    histories: Sequence[_BFGSHistory],
    *,
    alpha: float,
    max_step: float,
    solver: str = "eigh",
) -> tuple[torch.Tensor, ...]:
    """Update equal-sized BFGS histories with one batched dense solve."""

    position_vectors = torch.stack([value.flatten() for value in coordinates])
    force_vectors = torch.stack([value.flatten() for value in forces])
    batch_size, dimension = position_vectors.shape
    device = position_vectors.device

    with profile_phase(
        "optimizer.bfgs_history_pack",
        device=device,
        systems=batch_size,
        dimension=dimension,
    ):
        identity = torch.eye(
            dimension,
            device=device,
            dtype=position_vectors.dtype,
        )
        hessians = identity.expand(batch_size, -1, -1).clone().mul_(alpha)
        initialized = [
            index for index, history in enumerate(histories) if history.hessian is not None
        ]
        initialized_histories = [histories[index] for index in initialized]
        if any(
            history.positions is None or history.forces is None for history in initialized_histories
        ):
            raise RuntimeError("BFGS history is incomplete")
        initialized_ids = torch.as_tensor(initialized, device=device, dtype=torch.long)
        if initialized:
            hessians[initialized_ids] = torch.stack(
                [cast(torch.Tensor, history.hessian) for history in initialized_histories]
            )
            previous_positions = torch.stack(
                [cast(torch.Tensor, history.positions) for history in initialized_histories]
            )
            previous_forces = torch.stack(
                [cast(torch.Tensor, history.forces) for history in initialized_histories]
            )

    with profile_phase(
        "optimizer.bfgs_hessian_update",
        device=device,
        systems=len(initialized),
        dimension=dimension,
    ):
        if initialized:
            delta_position = position_vectors[initialized_ids] - previous_positions
            update_local = torch.nonzero(
                delta_position.abs().amax(dim=1) >= 1e-7,
                as_tuple=False,
            ).flatten()
            if update_local.numel():
                update_ids = initialized_ids[update_local]
                selected_delta_position = delta_position[update_local]
                delta_force = force_vectors[update_ids] - previous_forces[update_local]
                selected_hessians = hessians[update_ids]
                hessian_step = torch.bmm(
                    selected_hessians,
                    selected_delta_position.unsqueeze(-1),
                ).squeeze(-1)
                a = torch.sum(selected_delta_position * delta_force, dim=1)
                b = torch.sum(selected_delta_position * hessian_step, dim=1)
                hessians[update_ids] = selected_hessians - (
                    torch.bmm(delta_force.unsqueeze(2), delta_force.unsqueeze(1)) / a[:, None, None]
                    + torch.bmm(hessian_step.unsqueeze(2), hessian_step.unsqueeze(1))
                    / b[:, None, None]
                )

    if solver == "cholesky":
        with profile_phase(
            "optimizer.bfgs_cholesky",
            device=device,
            systems=batch_size,
            dimension=dimension,
        ):
            factors, info = torch.linalg.cholesky_ex(hessians)
            displacements = torch.cholesky_solve(force_vectors.unsqueeze(-1), factors).squeeze(-1)
        failed_ids = torch.nonzero(info, as_tuple=False).flatten()
        if failed_ids.numel():
            with profile_phase(
                "optimizer.bfgs_eigh_fallback",
                device=device,
                systems=failed_ids.numel(),
                dimension=dimension,
            ):
                failed_hessians = hessians[failed_ids]
                eigenvalues, eigenvectors = torch.linalg.eigh(failed_hessians)
                failed_forces = force_vectors[failed_ids]
                projected_forces = torch.bmm(failed_forces.unsqueeze(1), eigenvectors).squeeze(1)
                displacements[failed_ids] = torch.bmm(
                    eigenvectors,
                    (projected_forces / eigenvalues.abs()).unsqueeze(-1),
                ).squeeze(-1)
        profile_event(
            "bfgs_cholesky",
            systems=batch_size,
            fallback_systems=failed_ids.numel(),
            dimension=dimension,
        )
    else:
        with profile_phase(
            "optimizer.bfgs_eigh",
            device=device,
            systems=batch_size,
            dimension=dimension,
        ):
            eigenvalues, eigenvectors = torch.linalg.eigh(hessians)
        projected_forces = torch.bmm(force_vectors.unsqueeze(1), eigenvectors).squeeze(1)
        displacements = torch.bmm(
            eigenvectors,
            (projected_forces / eigenvalues.abs()).unsqueeze(-1),
        ).squeeze(-1)

    with profile_phase(
        "optimizer.bfgs_solve",
        device=device,
        systems=batch_size,
        dimension=dimension,
    ):
        displacements = displacements.reshape(batch_size, *coordinates[0].shape)
        max_row_norm = torch.linalg.vector_norm(displacements, dim=2).amax(dim=1)
        scale = torch.where(
            max_row_norm >= max_step,
            max_step / max_row_norm,
            torch.ones_like(max_row_norm),
        )
        displacements = displacements * scale[:, None, None]

    with profile_phase(
        "optimizer.bfgs_history_store",
        device=device,
        systems=batch_size,
        dimension=dimension,
    ):
        for index, history in enumerate(histories):
            history.hessian = hessians[index].detach()
            history.positions = position_vectors[index].detach().clone()
            history.forces = force_vectors[index].detach().clone()
    return tuple(value.detach() for value in displacements.unbind())


def _prepare_bfgs_steps(
    coordinates: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    histories: Sequence[_BFGSHistory],
    *,
    alpha: float,
    max_step: float,
    linear_algebra_backend: str,
) -> tuple[torch.Tensor, ...]:
    if not (len(coordinates) == len(forces) == len(histories)):
        raise ValueError("BFGS step inputs must have equal lengths")
    if not coordinates:
        return ()
    if linear_algebra_backend == "serial":
        return tuple(
            _prepare_bfgs_step(
                current_coordinates,
                current_forces,
                history,
                alpha=alpha,
                max_step=max_step,
            )
            for current_coordinates, current_forces, history in zip(
                coordinates, forces, histories, strict=True
            )
        )

    grouped_ids: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, value in enumerate(coordinates):
        grouped_ids[tuple(value.shape)].append(index)
    output: list[torch.Tensor | None] = [None] * len(coordinates)
    for indices in grouped_ids.values():
        dimension = coordinates[indices[0]].numel()
        if not _use_grouped_linear_algebra(
            linear_algebra_backend,
            device_type=coordinates[indices[0]].device.type,
            group_size=len(indices),
            dimension=dimension,
        ):
            for index in indices:
                output[index] = _prepare_bfgs_step(
                    coordinates[index],
                    forces[index],
                    histories[index],
                    alpha=alpha,
                    max_step=max_step,
                )
            continue
        grouped_output = _prepare_grouped_bfgs_steps(
            [coordinates[index] for index in indices],
            [forces[index] for index in indices],
            [histories[index] for index in indices],
            alpha=alpha,
            max_step=max_step,
            solver=(
                "cholesky"
                if linear_algebra_backend == "cholesky"
                or (
                    linear_algebra_backend == "auto"
                    and coordinates[indices[0]].device.type == "cuda"
                )
                else "eigh"
            ),
        )
        for index, displacement in zip(indices, grouped_output, strict=True):
            output[index] = displacement
    if any(value is None for value in output):
        raise RuntimeError("BFGS grouped solve did not produce every displacement")
    return tuple(value for value in output if value is not None)


def _validate_options(
    *,
    fmax: float,
    max_steps: int,
    max_step: float,
    alpha: float,
    callback_interval: int,
    smax: float | None,
) -> None:
    if fmax <= 0.0:
        raise ValueError("fmax must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if max_step <= 0.0:
        raise ValueError("max_step must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be positive")
    if smax is not None and smax <= 0.0:
        raise ValueError("smax must be positive")


def _resolve_optimizer_dtype(
    value: torch.dtype | str | None,
    state_dtype: torch.dtype,
) -> torch.dtype:
    if value is None:
        resolved = state_dtype
    elif isinstance(value, str):
        aliases = {
            "float32": torch.float32,
            "torch.float32": torch.float32,
            "float64": torch.float64,
            "torch.float64": torch.float64,
        }
        try:
            resolved = aliases[value.lower()]
        except KeyError as exc:
            raise ValueError("optimizer_dtype must be float32, float64, or None") from exc
    else:
        resolved = value
    if resolved not in (torch.float32, torch.float64):
        raise ValueError("optimizer_dtype must be float32, float64, or None")
    return resolved


def batched_bfgs_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 1000,
    max_step: float = 0.2,
    alpha: float = 70.0,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    zero_output_velocities: bool = True,
    active_compaction: bool = False,
    cell_filter: FrechetCellFilter | None = None,
    smax: float | None = 0.005,
    optimizer_dtype: torch.dtype | str | None = None,
    refill_batch_size: int | None = None,
    refill_policy: str = "immediate",
    refill_storage: str = "repack",
    refill_low_watermark: float = 0.8,
    refill_min_chunk: int | None = None,
    linear_algebra_backend: str = "auto",
) -> RelaxationResult:
    """Relax systems with independent full BFGS Hessians.

    The update, eigensolve, and maximum-row displacement clipping follow ASE
    BFGS. Optional Frechet cell rows use the same generalized coordinates and
    forces as ASE's ``FrechetCellFilter``. ``optimizer_dtype`` can promote the
    optimizer state independently of the calculator when desired; by default
    it follows the calculator state dtype. ``linear_algebra_backend="auto"``
    groups equal-sized CUDA systems and uses a positive-definite Cholesky solve
    with per-system eigen fallback. CPU calculations retain the serial
    ASE-compatible path. ``refill_storage="slots"`` overwrites completed
    equal-size resident slots and safely falls back to repacking otherwise.
    ``refill_storage="arena"`` uses reusable compact double buffers for
    heterogeneous batches. It is an explicit BFGS-only experimental mode and
    is not selected by the automatic planner.
    """

    _validate_options(
        fmax=fmax,
        max_steps=max_steps,
        max_step=max_step,
        alpha=alpha,
        callback_interval=callback_interval,
        smax=smax,
    )

    optimizer_dtype = _resolve_optimizer_dtype(optimizer_dtype, state.dtype)
    if linear_algebra_backend not in _LINEAR_ALGEBRA_BACKENDS:
        choices = ", ".join(sorted(_LINEAR_ALGEBRA_BACKENDS))
        raise ValueError(f"linear_algebra_backend must be one of: {choices}")
    if refill_policy not in _REFILL_POLICIES:
        choices = ", ".join(sorted(_REFILL_POLICIES))
        raise ValueError(f"refill_policy must be one of: {choices}")
    if refill_storage not in _REFILL_STORAGE_MODES:
        choices = ", ".join(sorted(_REFILL_STORAGE_MODES))
        raise ValueError(f"refill_storage must be one of: {choices}")
    if not 0.0 <= refill_low_watermark < 1.0:
        raise ValueError("refill_low_watermark must be in [0, 1)")
    if refill_min_chunk is not None and (
        isinstance(refill_min_chunk, bool)
        or not isinstance(refill_min_chunk, int)
        or refill_min_chunk <= 0
    ):
        raise ValueError("refill_min_chunk must be a positive integer or None")
    if refill_batch_size is None and (refill_policy != "immediate" or refill_min_chunk is not None):
        raise ValueError("refill policy options require refill_batch_size")
    if refill_batch_size is not None:
        if (
            isinstance(refill_batch_size, bool)
            or not isinstance(refill_batch_size, int)
            or refill_batch_size <= 0
        ):
            raise ValueError("refill_batch_size must be a positive integer")
        return _batched_bfgs_refill_relax(
            state,
            potential,
            refill_batch_size=refill_batch_size,
            refill_policy=refill_policy,
            refill_storage=refill_storage,
            refill_low_watermark=refill_low_watermark,
            refill_min_chunk=(
                max(8, refill_batch_size // 8) if refill_min_chunk is None else refill_min_chunk
            ),
            fmax=fmax,
            max_steps=max_steps,
            max_step=max_step,
            alpha=alpha,
            callback=callback,
            callback_interval=callback_interval,
            zero_output_velocities=zero_output_velocities,
            cell_filter=cell_filter,
            smax=smax,
            optimizer_dtype=optimizer_dtype,
            linear_algebra_backend=linear_algebra_backend,
        )
    n_systems = state.n_systems
    device, dtype = state.device, state.dtype
    active_state = state
    active_system_ids = torch.arange(n_systems, device=device, dtype=torch.long)
    active_atom_ids = torch.arange(state.n_atoms, device=device, dtype=torch.long)
    active_filter = (
        None if cell_filter is None else cell_filter.bind(active_state, dtype=optimizer_dtype)
    )
    optimizer_positions = (
        active_state.positions.detach().to(optimizer_dtype).clone()
        if active_filter is None
        else None
    )
    full_pressure = None if active_filter is None else active_filter.pressure.detach().clone()
    histories = [_BFGSHistory() for _ in range(n_systems)]

    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)
    full_energy = torch.empty((n_systems,), device=device, dtype=dtype)
    full_forces = torch.empty_like(state.positions)
    full_stress = (
        None
        if active_filter is None
        else torch.empty((n_systems, 3, 3), device=device, dtype=dtype)
    )
    full_fmax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_smax = (
        None
        if active_filter is None
        else torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    )
    full_generalized_fmax = (
        None
        if active_filter is None
        else torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    )

    evaluation = potential(
        active_state,
        neighbor_policy="auto",
        compute_stress=active_filter is not None,
    )
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    _profile_optimizer_evaluation(active_state, scheduler_step=0, pending_systems=0)
    neighbor_rebuilds = active_state.neighbor_rebuild_count
    active_batch_sizes = [n_systems]
    completed_steps = 0

    def sync_full_outputs(
        current_fmax: torch.Tensor,
        current_smax: torch.Tensor | None,
        current_generalized_fmax: torch.Tensor | None,
    ) -> None:
        if active_state is not state:
            state.positions[active_atom_ids] = active_state.positions
            state.cells[active_system_ids] = active_state.cells
        full_energy[active_system_ids] = evaluation.energy
        full_forces[active_atom_ids] = evaluation.forces
        full_fmax[active_system_ids] = current_fmax
        if full_stress is not None:
            if evaluation.stress is None:
                raise RuntimeError("variable-cell BFGS requires calculator stress")
            full_stress[active_system_ids] = evaluation.stress
        if full_smax is not None and current_smax is not None:
            full_smax[active_system_ids] = current_smax.to(full_smax.dtype)
        if full_generalized_fmax is not None and current_generalized_fmax is not None:
            full_generalized_fmax[active_system_ids] = current_generalized_fmax.to(
                full_generalized_fmax.dtype
            )

    for step in range(max_steps + 1):
        physical_forces = evaluation.forces.masked_fill(active_state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            atomic_forces = physical_forces
            cell_forces = None
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax < fmax
        else:
            if evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all()):
                raise FloatingPointError(
                    "calculator returned missing or non-finite stress for cell optimization"
                )
            atomic_forces, cell_forces = active_filter.generalized_forces(active_state, evaluation)
            current_smax = active_filter.max_stress(evaluation)
            current_generalized_fmax = max_generalized_force_per_system(
                active_state, atomic_forces, cell_forces
            )
            convergence_now = (
                current_generalized_fmax < fmax
                if smax is None
                else (current_fmax <= fmax) & (current_smax <= smax)
            )

        sync_full_outputs(current_fmax, current_smax, current_generalized_fmax)
        local_not_converged = converged_step[active_system_ids] < 0
        newly_converged = convergence_now & local_not_converged
        converged_step[active_system_ids[newly_converged]] = step
        converged = converged_step >= 0

        diagnostics = {
            "energy": full_energy.detach(),
            "max_force": full_fmax.detach(),
            "converged": converged.detach(),
            "neighbor_rebuild_count": torch.full(
                (n_systems,),
                neighbor_rebuilds,
                device=device,
                dtype=torch.int64,
            ),
        }
        if full_stress is not None:
            if full_pressure is None or full_smax is None:
                raise RuntimeError("variable-cell BFGS diagnostics are incomplete")
            volumes = torch.linalg.det(state.cells).abs()
            diagnostics.update(
                {
                    "enthalpy": (full_energy + full_pressure * volumes).detach(),
                    "max_stress": full_smax.detach(),
                    "max_generalized_force": full_generalized_fmax.detach(),
                    "stress": full_stress.detach(),
                    "volume": volumes.detach(),
                }
            )
        if callback is not None and step % callback_interval == 0:
            callback(
                step,
                state,
                BatchEvaluation(
                    energy=full_energy,
                    forces=full_forces,
                    stress=full_stress,
                ),
                diagnostics,
            )

        if bool(converged.all()) or step == max_steps:
            completed_steps = step
            break

        if active_compaction and bool(newly_converged.any()):
            systems_before = active_state.n_systems
            with profile_phase(
                "scheduler.active_compaction",
                device=device,
                systems=systems_before,
                atoms=active_state.n_atoms,
            ):
                remaining_local = torch.nonzero(~newly_converged, as_tuple=False).flatten().tolist()
                atom_blocks = [
                    torch.arange(
                        active_state.ptr[i],
                        active_state.ptr[i + 1],
                        device=device,
                        dtype=torch.long,
                    )
                    for i in remaining_local
                ]
                remaining_atoms = torch.cat(atom_blocks)
                selector = torch.as_tensor(remaining_local, device=device, dtype=torch.long)
                next_filter = (
                    None
                    if active_filter is None
                    else active_filter.select_systems(active_state, remaining_local)
                )
                next_state = active_state.select_systems(remaining_local, rebuild_neighbors=False)
                atomic_forces = atomic_forces[remaining_atoms].clone()
                if cell_forces is not None:
                    cell_forces = cell_forces[selector].clone()
                active_atom_ids = active_atom_ids[remaining_atoms].clone()
                active_system_ids = active_system_ids[selector].clone()
                histories = [histories[i] for i in remaining_local]
                if optimizer_positions is not None:
                    optimizer_positions = optimizer_positions[remaining_atoms].clone()
                active_state = next_state
                active_filter = next_filter
            profile_event(
                "active_compaction",
                scheduler_step=step,
                systems_before=systems_before,
                systems_after=active_state.n_systems,
                removed=systems_before - active_state.n_systems,
            )

        with profile_phase(
            "optimizer.bfgs_update",
            device=device,
            systems=active_state.n_systems,
            atoms=active_state.n_atoms,
        ):
            atomic_displacement = torch.zeros(
                active_state.positions.shape,
                device=device,
                dtype=optimizer_dtype,
            )
            cell_displacement = (
                None
                if active_filter is None
                else torch.zeros(
                    (active_state.n_systems, 3, 3),
                    device=device,
                    dtype=optimizer_dtype,
                )
            )
            local_active = converged_step[active_system_ids] < 0
            active_ids = torch.nonzero(local_active, as_tuple=False).flatten().tolist()
            coordinates = [
                _system_coordinates(
                    active_state,
                    system_id,
                    active_filter,
                    optimizer_positions,
                )
                for system_id in active_ids
            ]
            generalized_forces = [
                _system_forces(active_state, system_id, atomic_forces, cell_forces)
                for system_id in active_ids
            ]
            with profile_phase(
                "optimizer.bfgs_linear_algebra",
                device=device,
                systems=len(active_ids),
                dimension=0 if not coordinates else coordinates[0].numel(),
            ):
                displacements = _prepare_bfgs_steps(
                    coordinates,
                    generalized_forces,
                    [histories[system_id] for system_id in active_ids],
                    alpha=alpha,
                    max_step=max_step,
                    linear_algebra_backend=linear_algebra_backend,
                )
            for system_id, displacement in zip(active_ids, displacements, strict=True):
                atom_slice = active_state.atom_slice(system_id)
                atom_count = atom_slice.stop - atom_slice.start
                atomic_displacement[atom_slice] = displacement[:atom_count]
                if cell_displacement is not None:
                    cell_displacement[system_id] = displacement[atom_count:]

            atomic_displacement = atomic_displacement.masked_fill(
                ~active_state.mobile.unsqueeze(-1), 0.0
            )
            with profile_phase(
                "optimizer.bfgs_apply_displacement",
                device=device,
                systems=active_state.n_systems,
                atoms=active_state.n_atoms,
            ):
                if active_filter is None:
                    if optimizer_positions is None:
                        raise RuntimeError("fixed-cell BFGS optimizer positions are missing")
                    optimizer_positions = (optimizer_positions + atomic_displacement).detach()
                    active_state.positions = optimizer_positions.to(dtype=dtype).detach()
                else:
                    if cell_displacement is None:
                        raise RuntimeError("variable-cell displacement is missing")
                    active_filter.apply_displacement(
                        active_state, atomic_displacement, cell_displacement
                    )

        rebuilds_before = active_state.neighbor_rebuild_count
        evaluation = potential(
            active_state,
            neighbor_policy="auto",
            compute_stress=active_filter is not None,
        )
        _profile_optimizer_evaluation(active_state, scheduler_step=step + 1, pending_systems=0)
        neighbor_rebuilds += active_state.neighbor_rebuild_count - rebuilds_before
        active_batch_sizes.append(active_state.n_systems)
        completed_steps = step + 1

    state._neighbor_reference_positions = None
    state._neighbor_reference_cells = None
    state.neighbor_rebuild_count = neighbor_rebuilds
    if zero_output_velocities:
        state.velocities.zero_()

    if full_generalized_fmax is not None and smax is None:
        final_converged = full_generalized_fmax < fmax
    elif full_smax is not None:
        final_converged = (full_fmax <= fmax) & (full_smax <= smax)
    else:
        final_converged = full_fmax < fmax
    return RelaxationResult(
        state=state,
        evaluation=BatchEvaluation(
            energy=full_energy,
            forces=full_forces,
            stress=full_stress,
        ),
        converged=final_converged,
        converged_step=converged_step,
        max_force=full_fmax,
        max_stress=full_smax,
        steps=completed_steps,
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )


def _batched_bfgs_refill_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    refill_batch_size: int,
    refill_policy: str,
    refill_storage: str,
    refill_low_watermark: float,
    refill_min_chunk: int,
    fmax: float,
    max_steps: int,
    max_step: float,
    alpha: float,
    callback: StepCallback | None,
    callback_interval: int,
    zero_output_velocities: bool,
    cell_filter: FrechetCellFilter | None,
    smax: float | None,
    optimizer_dtype: torch.dtype,
    linear_algebra_backend: str,
) -> RelaxationResult:
    """Run BFGS with a bounded resident batch and a pending-system queue."""

    n_systems = state.n_systems
    capacity = min(refill_batch_size, n_systems)
    device, dtype = state.device, state.dtype
    active_system_ids = torch.arange(capacity, device=device, dtype=torch.long)
    next_pending = capacity
    active_atom_ids = _global_atom_ids(state, active_system_ids)
    active_state = (
        state
        if capacity == n_systems
        else state.select_systems(active_system_ids.tolist(), rebuild_neighbors=False)
    )
    resident_arena = (
        HeterogeneousResidentArena(state, resident_capacity=capacity)
        if refill_storage == "arena"
        else None
    )

    full_filter = None if cell_filter is None else cell_filter.bind(state, dtype=optimizer_dtype)
    active_filter = (
        None
        if full_filter is None
        else (
            full_filter
            if active_state is state
            else full_filter.select_systems(state, active_system_ids.tolist())
        )
    )
    full_optimizer_positions = (
        state.positions.detach().to(optimizer_dtype).clone() if full_filter is None else None
    )
    optimizer_positions = (
        None
        if full_optimizer_positions is None
        else full_optimizer_positions[active_atom_ids].clone()
    )
    full_pressure = None if full_filter is None else full_filter.pressure.detach().clone()
    histories: list[_BFGSHistory | None] = [_BFGSHistory() for _ in range(n_systems)]
    local_steps = torch.zeros((n_systems,), device=device, dtype=torch.int64)
    finished = torch.zeros((n_systems,), device=device, dtype=torch.bool)
    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)

    full_energy = torch.full((n_systems,), torch.nan, device=device, dtype=dtype)
    full_forces = torch.full_like(state.positions, torch.nan)
    full_stress = (
        None
        if full_filter is None
        else torch.full((n_systems, 3, 3), torch.nan, device=device, dtype=dtype)
    )
    full_fmax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_smax = (
        None
        if full_filter is None
        else torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    )
    full_generalized_fmax = (
        None
        if full_filter is None
        else torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    )

    neighbor_rebuilds = state.neighbor_rebuild_count

    def evaluate_active(current_scheduler_step: int) -> BatchEvaluation:
        nonlocal neighbor_rebuilds
        rebuilds_before = active_state.neighbor_rebuild_count
        current = potential(
            active_state,
            neighbor_policy="auto",
            compute_stress=active_filter is not None,
        )
        neighbor_rebuilds += active_state.neighbor_rebuild_count - rebuilds_before
        _profile_optimizer_evaluation(
            active_state,
            scheduler_step=current_scheduler_step,
            pending_systems=n_systems - next_pending,
        )
        return current

    def sync_active_state(
        evaluation: BatchEvaluation,
        current_fmax: torch.Tensor,
        current_smax: torch.Tensor | None,
        current_generalized_fmax: torch.Tensor | None,
    ) -> None:
        if active_state is not state:
            state.positions[active_atom_ids] = active_state.positions
            state.cells[active_system_ids] = active_state.cells
        full_energy[active_system_ids] = evaluation.energy
        full_forces[active_atom_ids] = evaluation.forces
        full_fmax[active_system_ids] = current_fmax
        if full_optimizer_positions is not None:
            if optimizer_positions is None:
                raise RuntimeError("fixed-cell BFGS positions are missing")
            full_optimizer_positions[active_atom_ids] = optimizer_positions
        if full_filter is not None:
            if active_filter is None or evaluation.stress is None:
                raise RuntimeError("variable-cell BFGS state is incomplete")
            if active_filter is not full_filter:
                full_filter.generalized_positions[active_atom_ids] = (
                    active_filter.generalized_positions
                )
                full_filter.log_deformation[active_system_ids] = active_filter.log_deformation
            full_stress[active_system_ids] = evaluation.stress
            if current_smax is not None:
                full_smax[active_system_ids] = current_smax.to(full_smax.dtype)
            if current_generalized_fmax is not None:
                full_generalized_fmax[active_system_ids] = current_generalized_fmax.to(
                    full_generalized_fmax.dtype
                )

    evaluation = evaluate_active(0)
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    active_batch_sizes = [active_state.n_systems]
    scheduler_step = 0

    while True:
        physical_forces = evaluation.forces.masked_fill(active_state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            atomic_forces = physical_forces
            cell_forces = None
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax < fmax
        else:
            if evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all()):
                raise FloatingPointError(
                    "calculator returned missing or non-finite stress for cell optimization"
                )
            atomic_forces, cell_forces = active_filter.generalized_forces(active_state, evaluation)
            current_smax = active_filter.max_stress(evaluation)
            current_generalized_fmax = max_generalized_force_per_system(
                active_state, atomic_forces, cell_forces
            )
            convergence_now = (
                current_generalized_fmax < fmax
                if smax is None
                else (current_fmax <= fmax) & (current_smax <= smax)
            )

        sync_active_state(evaluation, current_fmax, current_smax, current_generalized_fmax)
        exhausted_now = local_steps[active_system_ids] >= max_steps
        finish_now = convergence_now | exhausted_now
        newly_converged_ids = active_system_ids[convergence_now]
        converged_step[newly_converged_ids] = local_steps[newly_converged_ids]
        finished[active_system_ids[finish_now]] = True

        diagnostics = {
            "energy": full_energy.detach(),
            "max_force": full_fmax.detach(),
            "converged": (converged_step >= 0).detach(),
            "finished": finished.detach(),
            "local_steps": local_steps.detach(),
            "neighbor_rebuild_count": torch.full(
                (n_systems,),
                neighbor_rebuilds,
                device=device,
                dtype=torch.int64,
            ),
        }
        if full_stress is not None:
            if full_pressure is None or full_smax is None:
                raise RuntimeError("variable-cell BFGS diagnostics are incomplete")
            volumes = torch.linalg.det(state.cells).abs()
            diagnostics.update(
                {
                    "enthalpy": (full_energy + full_pressure * volumes).detach(),
                    "max_stress": full_smax.detach(),
                    "max_generalized_force": full_generalized_fmax.detach(),
                    "stress": full_stress.detach(),
                    "volume": volumes.detach(),
                }
            )
        if callback is not None and scheduler_step % callback_interval == 0:
            callback(
                scheduler_step,
                state,
                BatchEvaluation(
                    energy=full_energy,
                    forces=full_forces,
                    stress=full_stress,
                ),
                diagnostics,
            )

        if bool(finished.all()):
            break

        ready_count = active_state.n_systems
        ready_local_ids = torch.arange(ready_count, device=device, dtype=torch.long)
        if bool(finish_now.any()):
            systems_before = active_state.n_systems
            remaining_local = torch.nonzero(~finish_now, as_tuple=False).flatten()
            remaining_list = remaining_local.tolist()
            survivor_ids = active_system_ids[remaining_local]
            finished_local = torch.nonzero(finish_now, as_tuple=False).flatten()
            for system_id in active_system_ids[finish_now].tolist():
                histories[system_id] = None

            pending_before = n_systems - next_pending
            insert_count = _refill_insert_count(
                policy=refill_policy,
                capacity=capacity,
                survivors=len(remaining_list),
                pending=pending_before,
                low_watermark=refill_low_watermark,
                min_chunk=refill_min_chunk,
            )
            refill_stop = next_pending + insert_count
            refill_ids = torch.arange(
                next_pending,
                refill_stop,
                device=device,
                dtype=torch.long,
            )
            next_pending = refill_stop
            slot_counts_match = insert_count == finished_local.numel() and all(
                int(active_state.counts[destination]) == int(state.counts[source_id])
                for destination, source_id in zip(
                    finished_local.tolist(),
                    refill_ids.tolist(),
                    strict=True,
                )
            )
            use_slot_swap = refill_storage == "slots" and slot_counts_match
            use_arena = refill_storage == "arena"
            with profile_phase(
                (
                    "scheduler.refill_slot_swap"
                    if use_slot_swap
                    else (
                        "scheduler.refill_arena"
                        if use_arena
                        else "scheduler.refill_repack"
                    )
                ),
                device=device,
                systems=systems_before,
                atoms=active_state.n_atoms,
            ):
                if use_slot_swap:
                    destination_list = finished_local.tolist()
                    source_list = refill_ids.tolist()
                    active_state.replace_systems_from_(
                        destination_list,
                        state,
                        source_list,
                    )
                    active_system_ids = active_system_ids.clone()
                    active_system_ids[finished_local] = refill_ids
                    active_atom_ids = _global_atom_ids(state, active_system_ids)
                    for destination, source_id in zip(destination_list, source_list, strict=True):
                        destination_atoms = active_state.atom_slice(destination)
                        source_atoms = state.atom_slice(source_id)
                        atomic_forces[destination_atoms] = 0.0
                        if optimizer_positions is not None:
                            if full_optimizer_positions is None:
                                raise RuntimeError("full optimizer positions are missing")
                            optimizer_positions[destination_atoms] = full_optimizer_positions[
                                source_atoms
                            ]
                        if active_filter is not None:
                            if full_filter is None or cell_forces is None:
                                raise RuntimeError("full variable-cell state is missing")
                            active_filter.reference_cells[destination] = (
                                full_filter.reference_cells[source_id]
                            )
                            active_filter.generalized_positions[destination_atoms] = (
                                full_filter.generalized_positions[source_atoms]
                            )
                            active_filter.log_deformation[destination] = (
                                full_filter.log_deformation[source_id]
                            )
                            active_filter.cell_factor[destination] = full_filter.cell_factor[
                                source_id
                            ]
                            active_filter.pressure[destination] = full_filter.pressure[source_id]
                            cell_forces[destination] = 0.0
                    ready_local_ids = remaining_local
                    ready_count = remaining_local.numel()
                else:
                    remaining_atom_ids = (
                        torch.cat(
                            [
                                torch.arange(
                                    active_state.ptr[i],
                                    active_state.ptr[i + 1],
                                    device=device,
                                    dtype=torch.long,
                                )
                                for i in remaining_list
                            ]
                        )
                        if remaining_list
                        else torch.empty(0, device=device, dtype=torch.long)
                    )
                    survivor_forces = atomic_forces[remaining_atom_ids].clone()
                    survivor_cell_forces = (
                        None if cell_forces is None else cell_forces[remaining_local].clone()
                    )
                    active_system_ids = torch.cat((survivor_ids, refill_ids))
                    active_atom_ids = _global_atom_ids(state, active_system_ids)
                    if resident_arena is None:
                        state_parts = []
                        if remaining_list:
                            state_parts.append(
                                active_state.select_systems(
                                    remaining_list,
                                    rebuild_neighbors=False,
                                )
                            )
                        if refill_ids.numel():
                            state_parts.append(
                                state.select_systems(
                                    refill_ids.tolist(),
                                    rebuild_neighbors=False,
                                )
                            )
                        active_state = AseGraphBatch.concatenate(state_parts)
                    else:
                        selections = []
                        if remaining_list:
                            selections.append(
                                SystemSelection(
                                    active_state,
                                    tuple(remaining_list),
                                )
                            )
                        if refill_ids.numel():
                            selections.append(
                                SystemSelection(
                                    state,
                                    tuple(refill_ids.tolist()),
                                )
                            )
                        active_state = resident_arena.pack(selections)
                    active_filter = (
                        None
                        if full_filter is None
                        else full_filter.select_systems(state, active_system_ids.tolist())
                    )
                    optimizer_positions = (
                        None
                        if full_optimizer_positions is None
                        else (
                            full_optimizer_positions[active_atom_ids].clone()
                            if resident_arena is None
                            else resident_arena.work_tensor(
                                "bfgs.optimizer_positions",
                                (active_state.n_atoms, 3),
                                dtype=optimizer_dtype,
                            )
                        )
                    )
                    if (
                        optimizer_positions is not None
                        and resident_arena is not None
                    ):
                        optimizer_positions.copy_(
                            full_optimizer_positions[active_atom_ids]
                        )
                    atomic_forces = (
                        torch.zeros(
                            active_state.positions.shape,
                            device=device,
                            dtype=optimizer_dtype,
                        )
                        if resident_arena is None
                        else resident_arena.work_tensor(
                            "bfgs.atomic_forces",
                            tuple(active_state.positions.shape),
                            dtype=optimizer_dtype,
                            zero=True,
                        )
                    )
                    atomic_forces[: survivor_forces.shape[0]] = survivor_forces
                    if active_filter is None:
                        cell_forces = None
                    else:
                        cell_forces = (
                            torch.zeros(
                                (active_state.n_systems, 3, 3),
                                device=device,
                                dtype=optimizer_dtype,
                            )
                            if resident_arena is None
                            else resident_arena.work_tensor(
                                "bfgs.cell_forces",
                                (active_state.n_systems, 3, 3),
                                dtype=optimizer_dtype,
                                zero=True,
                            )
                        )
                        if survivor_cell_forces is not None:
                            cell_forces[: len(remaining_list)] = survivor_cell_forces
                    ready_count = len(remaining_list)
                    ready_local_ids = torch.arange(ready_count, device=device, dtype=torch.long)
            profile_event(
                "refill",
                policy=refill_policy,
                low_watermark=refill_low_watermark,
                min_chunk=refill_min_chunk,
                scheduler_step=scheduler_step,
                systems_before=systems_before,
                survivors=ready_count,
                inserted=refill_ids.numel(),
                triggered=bool(refill_ids.numel()),
                storage=(
                    "slots"
                    if use_slot_swap
                    else ("arena" if use_arena else "repack")
                ),
                systems_after=active_state.n_systems,
                pending_after=n_systems - next_pending,
            )

        with profile_phase(
            "optimizer.bfgs_update",
            device=device,
            systems=active_state.n_systems,
            atoms=active_state.n_atoms,
        ):
            atomic_displacement = torch.zeros(
                active_state.positions.shape,
                device=device,
                dtype=optimizer_dtype,
            )
            cell_displacement = (
                None
                if active_filter is None
                else torch.zeros(
                    (active_state.n_systems, 3, 3),
                    device=device,
                    dtype=optimizer_dtype,
                )
            )
            coordinates = []
            generalized_forces = []
            active_histories = []
            for local_id in ready_local_ids.tolist():
                global_id = int(active_system_ids[local_id])
                history = histories[global_id]
                if history is None:
                    raise RuntimeError("active BFGS history was released")
                coordinates.append(
                    _system_coordinates(
                        active_state,
                        local_id,
                        active_filter,
                        optimizer_positions,
                    )
                )
                generalized_forces.append(
                    _system_forces(
                        active_state,
                        local_id,
                        atomic_forces,
                        cell_forces,
                    )
                )
                active_histories.append(history)
            with profile_phase(
                "optimizer.bfgs_linear_algebra",
                device=device,
                systems=len(active_histories),
                dimension=0 if not coordinates else coordinates[0].numel(),
            ):
                displacements = _prepare_bfgs_steps(
                    coordinates,
                    generalized_forces,
                    active_histories,
                    alpha=alpha,
                    max_step=max_step,
                    linear_algebra_backend=linear_algebra_backend,
                )
            for local_id, displacement in zip(ready_local_ids.tolist(), displacements, strict=True):
                atom_slice = active_state.atom_slice(local_id)
                atom_count = atom_slice.stop - atom_slice.start
                atomic_displacement[atom_slice] = displacement[:atom_count]
                if cell_displacement is not None:
                    cell_displacement[local_id] = displacement[atom_count:]

            stepped_ids = active_system_ids[ready_local_ids]
            local_steps[stepped_ids] += 1
            atomic_displacement = atomic_displacement.masked_fill(
                ~active_state.mobile.unsqueeze(-1), 0.0
            )
            with profile_phase(
                "optimizer.bfgs_apply_displacement",
                device=device,
                systems=active_state.n_systems,
                atoms=active_state.n_atoms,
            ):
                if active_filter is None:
                    if optimizer_positions is None:
                        raise RuntimeError("fixed-cell BFGS optimizer positions are missing")
                    optimizer_positions = (optimizer_positions + atomic_displacement).detach()
                    active_state.positions = optimizer_positions.to(dtype=dtype).detach()
                else:
                    if cell_displacement is None:
                        raise RuntimeError("variable-cell displacement is missing")
                    active_filter.apply_displacement(
                        active_state, atomic_displacement, cell_displacement
                    )

        evaluation = evaluate_active(scheduler_step + 1)
        active_batch_sizes.append(active_state.n_systems)
        scheduler_step += 1

    state._neighbor_reference_positions = None
    state._neighbor_reference_cells = None
    state.neighbor_rebuild_count = neighbor_rebuilds
    if zero_output_velocities:
        state.velocities.zero_()

    if full_generalized_fmax is not None and smax is None:
        final_converged = full_generalized_fmax < fmax
    elif full_smax is not None:
        final_converged = (full_fmax <= fmax) & (full_smax <= smax)
    else:
        final_converged = full_fmax < fmax
    return RelaxationResult(
        state=state,
        evaluation=BatchEvaluation(
            energy=full_energy,
            forces=full_forces,
            stress=full_stress,
        ),
        converged=final_converged,
        converged_step=converged_step,
        max_force=full_fmax,
        max_stress=full_smax,
        steps=int(local_steps.max().item()),
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )
