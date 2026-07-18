"""ASE-compatible full BFGS optimization for heterogeneous batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.calculator import BatchCalculator
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation, RelaxationResult, StepCallback
from .cell_filters import BoundFrechetCellFilter, FrechetCellFilter
from .fire import max_force_per_system, max_generalized_force_per_system


@dataclass
class _BFGSHistory:
    hessian: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    forces: torch.Tensor | None = None


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
    cell_coordinates = (
        cell_filter.log_deformation[system_id]
        * cell_filter.cell_factor[system_id]
    )
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
        history.hessian = torch.eye(
            position_vector.numel(),
            device=coordinates.device,
            dtype=coordinates.dtype,
        ) * alpha
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
    displacement = eigenvectors @ (
        (force_vector @ eigenvectors) / eigenvalues.abs()
    )
    displacement = displacement.reshape_as(coordinates)
    max_row_norm = torch.linalg.vector_norm(displacement, dim=1).max()
    if bool(max_row_norm >= max_step):
        displacement = displacement * (max_step / max_row_norm)

    history.positions = position_vector.detach().clone()
    history.forces = force_vector.detach().clone()
    return displacement.detach()


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
            raise ValueError(
                "optimizer_dtype must be float32, float64, or None"
            ) from exc
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
) -> RelaxationResult:
    """Relax systems with independent full BFGS Hessians.

    The update, eigensolve, and maximum-row displacement clipping follow ASE
    BFGS. Optional Frechet cell rows use the same generalized coordinates and
    forces as ASE's ``FrechetCellFilter``. ``optimizer_dtype`` can promote the
    optimizer state independently of the calculator when desired; by default
    it follows the calculator state dtype.
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
        )
    n_systems = state.n_systems
    device, dtype = state.device, state.dtype
    active_state = state
    active_system_ids = torch.arange(n_systems, device=device, dtype=torch.long)
    active_atom_ids = torch.arange(state.n_atoms, device=device, dtype=torch.long)
    active_filter = (
        None
        if cell_filter is None
        else cell_filter.bind(active_state, dtype=optimizer_dtype)
    )
    optimizer_positions = (
        active_state.positions.detach().to(optimizer_dtype).clone()
        if active_filter is None
        else None
    )
    full_pressure = (
        None if active_filter is None else active_filter.pressure.detach().clone()
    )
    histories = [_BFGSHistory() for _ in range(n_systems)]

    converged_step = torch.full(
        (n_systems,), -1, device=device, dtype=torch.int64
    )
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
        if (
            full_generalized_fmax is not None
            and current_generalized_fmax is not None
        ):
            full_generalized_fmax[active_system_ids] = current_generalized_fmax.to(
                full_generalized_fmax.dtype
            )

    for step in range(max_steps + 1):
        physical_forces = evaluation.forces.masked_fill(
            active_state.fixed.unsqueeze(-1), 0.0
        )
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            atomic_forces = physical_forces
            cell_forces = None
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax < fmax
        else:
            if evaluation.stress is None or not bool(
                torch.isfinite(evaluation.stress).all()
            ):
                raise FloatingPointError(
                    "calculator returned missing or non-finite stress for cell optimization"
                )
            atomic_forces, cell_forces = active_filter.generalized_forces(
                active_state, evaluation
            )
            current_smax = active_filter.max_stress(evaluation)
            current_generalized_fmax = max_generalized_force_per_system(
                active_state, atomic_forces, cell_forces
            )
            convergence_now = (
                current_generalized_fmax < fmax
                if smax is None
                else (current_fmax <= fmax) & (current_smax <= smax)
            )

        sync_full_outputs(
            current_fmax, current_smax, current_generalized_fmax
        )
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
            remaining_local = torch.nonzero(
                ~newly_converged, as_tuple=False
            ).flatten().tolist()
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
            selector = torch.as_tensor(
                remaining_local, device=device, dtype=torch.long
            )
            next_filter = (
                None
                if active_filter is None
                else active_filter.select_systems(active_state, remaining_local)
            )
            next_state = active_state.select_systems(
                remaining_local, rebuild_neighbors=False
            )
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
        for system_id in torch.nonzero(
            local_active, as_tuple=False
        ).flatten().tolist():
            coordinates = _system_coordinates(
                active_state,
                system_id,
                active_filter,
                optimizer_positions,
            )
            generalized_forces = _system_forces(
                active_state, system_id, atomic_forces, cell_forces
            )
            displacement = _prepare_bfgs_step(
                coordinates,
                generalized_forces,
                histories[system_id],
                alpha=alpha,
                max_step=max_step,
            )
            atom_slice = active_state.atom_slice(system_id)
            atom_count = atom_slice.stop - atom_slice.start
            atomic_displacement[atom_slice] = displacement[:atom_count]
            if cell_displacement is not None:
                cell_displacement[system_id] = displacement[atom_count:]

        atomic_displacement = atomic_displacement.masked_fill(
            ~active_state.mobile.unsqueeze(-1), 0.0
        )
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
        neighbor_rebuilds += (
            active_state.neighbor_rebuild_count - rebuilds_before
        )
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


def _global_atom_ids(
    state: AseGraphBatch,
    system_ids: torch.Tensor,
) -> torch.Tensor:
    blocks = [
        torch.arange(
            state.ptr[system_id],
            state.ptr[system_id + 1],
            device=state.device,
            dtype=torch.long,
        )
        for system_id in system_ids.tolist()
    ]
    return torch.cat(blocks)


def _batched_bfgs_refill_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    refill_batch_size: int,
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
        else state.select_systems(
            active_system_ids.tolist(), rebuild_neighbors=False
        )
    )

    full_filter = (
        None
        if cell_filter is None
        else cell_filter.bind(state, dtype=optimizer_dtype)
    )
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
        state.positions.detach().to(optimizer_dtype).clone()
        if full_filter is None
        else None
    )
    optimizer_positions = (
        None
        if full_optimizer_positions is None
        else full_optimizer_positions[active_atom_ids].clone()
    )
    full_pressure = (
        None if full_filter is None else full_filter.pressure.detach().clone()
    )
    histories: list[_BFGSHistory | None] = [
        _BFGSHistory() for _ in range(n_systems)
    ]
    local_steps = torch.zeros((n_systems,), device=device, dtype=torch.int64)
    finished = torch.zeros((n_systems,), device=device, dtype=torch.bool)
    converged_step = torch.full(
        (n_systems,), -1, device=device, dtype=torch.int64
    )

    full_energy = torch.full(
        (n_systems,), torch.nan, device=device, dtype=dtype
    )
    full_forces = torch.full_like(state.positions, torch.nan)
    full_stress = (
        None
        if full_filter is None
        else torch.full(
            (n_systems, 3, 3), torch.nan, device=device, dtype=dtype
        )
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

    def evaluate_active() -> BatchEvaluation:
        nonlocal neighbor_rebuilds
        rebuilds_before = active_state.neighbor_rebuild_count
        current = potential(
            active_state,
            neighbor_policy="auto",
            compute_stress=active_filter is not None,
        )
        neighbor_rebuilds += (
            active_state.neighbor_rebuild_count - rebuilds_before
        )
        return current

    def sync_active_state(
        evaluation: BatchEvaluation,
        current_fmax: torch.Tensor,
        current_smax: torch.Tensor | None,
        current_generalized_fmax: torch.Tensor | None,
    ) -> None:
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
            full_filter.generalized_positions[active_atom_ids] = (
                active_filter.generalized_positions
            )
            full_filter.log_deformation[active_system_ids] = (
                active_filter.log_deformation
            )
            full_stress[active_system_ids] = evaluation.stress
            if current_smax is not None:
                full_smax[active_system_ids] = current_smax.to(full_smax.dtype)
            if current_generalized_fmax is not None:
                full_generalized_fmax[active_system_ids] = (
                    current_generalized_fmax.to(full_generalized_fmax.dtype)
                )

    evaluation = evaluate_active()
    active_batch_sizes = [active_state.n_systems]
    scheduler_step = 0

    while True:
        physical_forces = evaluation.forces.masked_fill(
            active_state.fixed.unsqueeze(-1), 0.0
        )
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            atomic_forces = physical_forces
            cell_forces = None
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax < fmax
        else:
            if evaluation.stress is None or not bool(
                torch.isfinite(evaluation.stress).all()
            ):
                raise FloatingPointError(
                    "calculator returned missing or non-finite stress for cell optimization"
                )
            atomic_forces, cell_forces = active_filter.generalized_forces(
                active_state, evaluation
            )
            current_smax = active_filter.max_stress(evaluation)
            current_generalized_fmax = max_generalized_force_per_system(
                active_state, atomic_forces, cell_forces
            )
            convergence_now = (
                current_generalized_fmax < fmax
                if smax is None
                else (current_fmax <= fmax) & (current_smax <= smax)
            )

        sync_active_state(
            evaluation, current_fmax, current_smax, current_generalized_fmax
        )
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
        if bool(finish_now.any()):
            remaining_local = torch.nonzero(
                ~finish_now, as_tuple=False
            ).flatten()
            remaining_list = remaining_local.tolist()
            remaining_atom_ids = torch.cat(
                [
                    torch.arange(
                        active_state.ptr[i],
                        active_state.ptr[i + 1],
                        device=device,
                        dtype=torch.long,
                    )
                    for i in remaining_list
                ]
            ) if remaining_list else torch.empty(0, device=device, dtype=torch.long)
            survivor_forces = atomic_forces[remaining_atom_ids].clone()
            survivor_cell_forces = (
                None
                if cell_forces is None
                else cell_forces[remaining_local].clone()
            )
            survivor_ids = active_system_ids[remaining_local]
            for system_id in active_system_ids[finish_now].tolist():
                histories[system_id] = None

            slots = capacity - len(remaining_list)
            refill_stop = min(next_pending + slots, n_systems)
            refill_ids = torch.arange(
                next_pending, refill_stop, device=device, dtype=torch.long
            )
            next_pending = refill_stop
            active_system_ids = torch.cat((survivor_ids, refill_ids))
            active_atom_ids = _global_atom_ids(state, active_system_ids)
            active_state = state.select_systems(
                active_system_ids.tolist(), rebuild_neighbors=False
            )
            active_filter = (
                None
                if full_filter is None
                else full_filter.select_systems(
                    state, active_system_ids.tolist()
                )
            )
            optimizer_positions = (
                None
                if full_optimizer_positions is None
                else full_optimizer_positions[active_atom_ids].clone()
            )
            atomic_forces = torch.zeros(
                active_state.positions.shape,
                device=device,
                dtype=optimizer_dtype,
            )
            atomic_forces[: survivor_forces.shape[0]] = survivor_forces
            if active_filter is None:
                cell_forces = None
            else:
                cell_forces = torch.zeros(
                    (active_state.n_systems, 3, 3),
                    device=device,
                    dtype=optimizer_dtype,
                )
                if survivor_cell_forces is not None:
                    cell_forces[: len(remaining_list)] = survivor_cell_forces
            ready_count = len(remaining_list)

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
        for local_id in range(ready_count):
            global_id = int(active_system_ids[local_id])
            history = histories[global_id]
            if history is None:
                raise RuntimeError("active BFGS history was released")
            coordinates = _system_coordinates(
                active_state,
                local_id,
                active_filter,
                optimizer_positions,
            )
            generalized_forces = _system_forces(
                active_state,
                local_id,
                atomic_forces,
                cell_forces,
            )
            displacement = _prepare_bfgs_step(
                coordinates,
                generalized_forces,
                history,
                alpha=alpha,
                max_step=max_step,
            )
            atom_slice = active_state.atom_slice(local_id)
            atom_count = atom_slice.stop - atom_slice.start
            atomic_displacement[atom_slice] = displacement[:atom_count]
            if cell_displacement is not None:
                cell_displacement[local_id] = displacement[atom_count:]

        stepped_ids = active_system_ids[:ready_count]
        local_steps[stepped_ids] += 1
        atomic_displacement = atomic_displacement.masked_fill(
            ~active_state.mobile.unsqueeze(-1), 0.0
        )
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

        evaluation = evaluate_active()
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
