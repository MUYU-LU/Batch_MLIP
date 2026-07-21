"""ASE-compatible BFGS line search for heterogeneous tensor batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase.utils.linesearch import LineSearch

from ..core.calculator import BatchCalculator
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation, RelaxationResult, StepCallback
from ..profiling.runtime import profile_event, profile_phase
from .bfgs import (
    _resolve_optimizer_dtype,
    _system_coordinates,
    _system_forces,
)
from .cell_filters import BoundFrechetCellFilter, FrechetCellFilter
from .fire import max_force_per_system, max_generalized_force_per_system


@dataclass
class _LineSearchHistory:
    inverse_hessian: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    gradients: torch.Tensor | None = None
    direction: torch.Tensor | None = None
    step_size: float | None = None
    no_update: bool = False


@dataclass
class _ActiveLineSearch:
    system_id: int
    base_coordinates: torch.Tensor
    base_gradient: torch.Tensor
    direction: torch.Tensor
    search: LineSearch
    step_size: float
    evaluations: int = 0


def _max_row_norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.reshape(-1, 3), axis=1).max())


def _new_line_search() -> LineSearch:
    """Construct ASE's line search across the supported ASE API versions."""

    try:
        return LineSearch(get_gradient_norm=_max_row_norm)
    except TypeError:
        # ASE <= 3.26 hard-codes the same Cartesian row norm.
        return LineSearch()


def _update_inverse_hessian(
    coordinates: torch.Tensor,
    gradient: torch.Tensor,
    history: _LineSearchHistory,
) -> None:
    dimension = coordinates.numel()
    if history.inverse_hessian is None:
        history.inverse_hessian = torch.eye(
            dimension, device=coordinates.device, dtype=coordinates.dtype
        )
        return
    if (
        history.positions is None
        or history.gradients is None
        or history.direction is None
    ):
        raise RuntimeError("BFGSLineSearch history is incomplete")
    if history.step_size is None or history.step_size <= 0.0:
        return
    if (
        abs(torch.dot(gradient, history.direction).item())
        - abs(torch.dot(history.gradients, history.direction).item())
        >= 0.0
        or history.no_update
    ):
        return

    delta_position = coordinates - history.positions
    delta_gradient = gradient - history.gradients
    denominator = torch.dot(delta_gradient, delta_position)
    rho = 1.0 / denominator
    if not bool(torch.isfinite(rho)):
        rho = torch.as_tensor(
            1000.0, device=coordinates.device, dtype=coordinates.dtype
        )
    identity = torch.eye(
        dimension, device=coordinates.device, dtype=coordinates.dtype
    )
    left = identity - torch.outer(delta_position, delta_gradient) * rho
    right = identity - torch.outer(delta_gradient, delta_position) * rho
    history.inverse_hessian = (
        left @ history.inverse_hessian @ right
        + torch.outer(delta_position, delta_position) * rho
    )


def _start_line_search(
    system_id: int,
    coordinates: torch.Tensor,
    forces: torch.Tensor,
    objective: torch.Tensor,
    history: _LineSearchHistory,
    *,
    alpha: float,
    max_step: float,
    c1: float,
    c2: float,
    stpmax: float,
) -> _ActiveLineSearch:
    coordinates_vector = coordinates.flatten().detach().clone()
    gradient = (-forces.flatten() / alpha).detach()
    _update_inverse_hessian(coordinates_vector, gradient, history)
    if history.inverse_hessian is None:
        raise RuntimeError("BFGSLineSearch inverse Hessian was not initialized")

    direction = -(history.inverse_hessian @ gradient)
    direction_norm = torch.linalg.vector_norm(direction)
    minimum_norm = (coordinates_vector.numel() / 3.0 * 1e-10) ** 0.5
    if bool(direction_norm <= minimum_norm):
        direction = direction * (minimum_norm / direction_norm)

    search = _new_line_search()
    search.stpmin = 1e-8
    search.pk = direction.detach().cpu().numpy()
    search.stpmax = stpmax
    search.xtrapl = 1.1
    search.xtrapu = 4.0
    search.maxstep = max_step
    search.dim = coordinates_vector.numel()
    search.gms = np.sqrt(search.dim) * max_step
    search.no_update = False
    search.steps = []
    phi0 = float(objective.item()) / alpha
    derivative0 = float(torch.dot(gradient, direction).item())
    step_size = float(
        search.step(
            1.0,
            phi0,
            derivative0,
            c1,
            c2,
            search.xtol,
            search.isave,
            search.dsave,
        )
    )
    if search.task[:5] == "ERROR":
        raise RuntimeError(
            f"BFGSLineSearch failed for system {system_id}: {search.task}"
        )
    return _ActiveLineSearch(
        system_id=system_id,
        base_coordinates=coordinates_vector,
        base_gradient=gradient,
        direction=direction.detach(),
        search=search,
        step_size=step_size,
    )


def _apply_trial_coordinates(
    state: AseGraphBatch,
    searches: list[_ActiveLineSearch],
    cell_filter: BoundFrechetCellFilter | None,
    optimizer_positions: torch.Tensor | None,
) -> None:
    atomic_displacement = torch.zeros(
        state.positions.shape,
        device=state.device,
        dtype=searches[0].base_coordinates.dtype,
    )
    cell_displacement = (
        None
        if cell_filter is None
        else torch.zeros(
            (state.n_systems, 3, 3),
            device=state.device,
            dtype=searches[0].base_coordinates.dtype,
        )
    )
    for pending in searches:
        system_id = pending.system_id
        current = _system_coordinates(
            state, system_id, cell_filter, optimizer_positions
        )
        target = (
            pending.base_coordinates + pending.step_size * pending.direction
        ).reshape_as(current)
        displacement = target - current
        atom_slice = state.atom_slice(system_id)
        atom_count = atom_slice.stop - atom_slice.start
        atomic_displacement[atom_slice] = displacement[:atom_count]
        if cell_displacement is not None:
            cell_displacement[system_id] = displacement[atom_count:]

    atomic_displacement = atomic_displacement.masked_fill(
        ~state.mobile.unsqueeze(-1), 0.0
    )
    if cell_filter is None:
        if optimizer_positions is None:
            raise RuntimeError("fixed-cell optimizer positions are missing")
        optimizer_positions.add_(atomic_displacement)
        state.positions = optimizer_positions.to(dtype=state.dtype).detach()
    else:
        if cell_displacement is None:
            raise RuntimeError("variable-cell line-search displacement is missing")
        cell_filter.apply_displacement(
            state, atomic_displacement, cell_displacement
        )


def _objectives_and_forces(
    state: AseGraphBatch,
    evaluation: BatchEvaluation,
    cell_filter: BoundFrechetCellFilter | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    physical_forces = evaluation.forces.masked_fill(
        state.fixed.unsqueeze(-1), 0.0
    )
    if cell_filter is None:
        return evaluation.energy, physical_forces, None
    if evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all()):
        raise FloatingPointError(
            "calculator returned missing or non-finite stress for cell optimization"
        )
    atomic_forces, cell_forces = cell_filter.generalized_forces(state, evaluation)
    objectives = evaluation.energy.to(cell_filter.pressure.dtype) + (
        cell_filter.pressure * cell_filter.volumes(state)
    )
    return objectives, atomic_forces, cell_forces


def _validate_options(
    *,
    fmax: float,
    max_steps: int,
    max_step: float,
    alpha: float,
    c1: float,
    c2: float,
    stpmax: float,
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
    if not 0.0 < c1 < c2 < 1.0:
        raise ValueError("line-search constants must satisfy 0 < c1 < c2 < 1")
    if stpmax <= 0.0:
        raise ValueError("stpmax must be positive")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be positive")
    if smax is not None and smax <= 0.0:
        raise ValueError("smax must be positive")


def batched_bfgs_line_search_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 1000,
    max_step: float = 0.2,
    c1: float = 0.23,
    c2: float = 0.46,
    alpha: float = 10.0,
    stpmax: float = 50.0,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    zero_output_velocities: bool = True,
    active_compaction: bool = False,
    cell_filter: FrechetCellFilter | None = None,
    smax: float | None = 0.005,
    optimizer_dtype: torch.dtype | str | None = None,
) -> RelaxationResult:
    """Relax independent systems with ASE's BFGSLineSearch equations.

    Each structure owns its inverse Hessian and strong-Wolfe state. Trial
    points requested by unfinished line searches are evaluated together;
    structures whose searches finish earlier remain at their accepted point.
    """

    _validate_options(
        fmax=fmax,
        max_steps=max_steps,
        max_step=max_step,
        alpha=alpha,
        c1=c1,
        c2=c2,
        stpmax=stpmax,
        callback_interval=callback_interval,
        smax=smax,
    )
    optimizer_dtype = _resolve_optimizer_dtype(optimizer_dtype, state.dtype)
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
    histories = [_LineSearchHistory() for _ in range(n_systems)]

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
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    neighbor_rebuilds = active_state.neighbor_rebuild_count
    active_batch_sizes = [n_systems]
    profile_event(
        "optimizer_evaluation",
        optimizer="bfgslinesearch",
        scheduler_step=0,
        active_systems=n_systems,
        active_atoms=active_state.n_atoms,
        active_edges=active_state.edge_index.shape[1],
        pending_systems=0,
    )
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
                raise RuntimeError("variable-cell line search requires stress")
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
        objectives, atomic_forces, cell_forces = _objectives_and_forces(
            active_state, evaluation, active_filter
        )
        physical_forces = evaluation.forces.masked_fill(
            active_state.fixed.unsqueeze(-1), 0.0
        )
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax < fmax
        else:
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
                (n_systems,), neighbor_rebuilds, device=device, dtype=torch.int64
            ),
        }
        if full_stress is not None:
            if full_pressure is None or full_smax is None:
                raise RuntimeError("variable-cell diagnostics are incomplete")
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
                    energy=full_energy, forces=full_forces, stress=full_stress
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
                objectives = objectives[selector].clone()
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

        local_active = converged_step[active_system_ids] < 0
        active_ids = torch.nonzero(local_active, as_tuple=False).flatten().tolist()
        searches: list[_ActiveLineSearch] = []
        with profile_phase(
            "optimizer.bfgs_line_search_update",
            device=device,
            systems=len(active_ids),
            atoms=active_state.n_atoms,
        ):
            for system_id in active_ids:
                coordinates = _system_coordinates(
                    active_state, system_id, active_filter, optimizer_positions
                )
                generalized_forces = _system_forces(
                    active_state, system_id, atomic_forces, cell_forces
                )
                searches.append(
                    _start_line_search(
                        system_id,
                        coordinates,
                        generalized_forces,
                        objectives[system_id],
                        histories[system_id],
                        alpha=alpha,
                        max_step=max_step,
                        c1=c1,
                        c2=c2,
                        stpmax=stpmax,
                    )
                )

        unfinished = searches
        while unfinished:
            _apply_trial_coordinates(
                active_state, unfinished, active_filter, optimizer_positions
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
            profile_event(
                "optimizer_evaluation",
                optimizer="bfgslinesearch",
                scheduler_step=step + 1,
                active_systems=active_state.n_systems,
                active_atoms=active_state.n_atoms,
                active_edges=active_state.edge_index.shape[1],
                pending_systems=0,
            )
            trial_objectives, trial_atomic_forces, trial_cell_forces = (
                _objectives_and_forces(active_state, evaluation, active_filter)
            )
            next_unfinished: list[_ActiveLineSearch] = []
            for pending in unfinished:
                system_id = pending.system_id
                pending.evaluations += 1
                if pending.evaluations > 100:
                    raise RuntimeError(
                        "BFGSLineSearch exceeded 100 trial evaluations for system "
                        f"{int(active_system_ids[system_id])}"
                    )
                trial_forces = _system_forces(
                    active_state,
                    system_id,
                    trial_atomic_forces,
                    trial_cell_forces,
                )
                trial_gradient = -trial_forces.flatten() / alpha
                phi = float(trial_objectives[system_id].item()) / alpha
                derivative = float(
                    torch.dot(trial_gradient, pending.direction).item()
                )
                # ASE records the evaluated point before requesting the next
                # incremental, max-step-limited trial.
                pending.search.old_stp = pending.step_size
                next_step = pending.search.step(
                    pending.step_size,
                    phi,
                    derivative,
                    c1,
                    c2,
                    pending.search.xtol,
                    pending.search.isave,
                    pending.search.dsave,
                )
                if pending.search.task[:2] == "FG":
                    pending.step_size = float(next_step)
                    next_unfinished.append(pending)
                    continue
                if pending.search.task[:4] not in ("CONV", "WARN"):
                    raise RuntimeError(
                        "BFGSLineSearch failed for system "
                        f"{int(active_system_ids[system_id])}: "
                        f"{pending.search.task}"
                    )
                history = histories[system_id]
                history.positions = pending.base_coordinates.detach().clone()
                history.gradients = pending.base_gradient.detach().clone()
                history.direction = pending.direction.detach().clone()
                history.step_size = pending.step_size
                history.no_update = bool(pending.search.no_update)
            unfinished = next_unfinished
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
            energy=full_energy, forces=full_forces, stress=full_stress
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
