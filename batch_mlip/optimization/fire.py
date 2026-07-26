"""Batched FIRE and steepest-descent optimization kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.calculator import BatchCalculator
from ..core.math_utils import scatter_max, scatter_sum, system_l2_norm
from ..core.state import AseGraphBatch
from ..core.types import BatchEvaluation, RelaxationResult, StepCallback
from ..profiling.runtime import profile_event, profile_phase
from .cell_filters import FrechetCellFilter
from .refill import (
    REFILL_POLICIES,
    REFILL_STORAGE_MODES,
    global_atom_ids,
    refill_insert_count,
)


@dataclass(frozen=True)
class FIREConfig:
    fmax: float = 0.05
    max_steps: int = 1000
    dt_start: float = 0.1
    dt_max: float = 1.0
    max_step: float = 0.2
    alpha_start: float = 0.1
    n_min: int = 5
    f_inc: float = 1.1
    f_dec: float = 0.5
    f_alpha: float = 0.99
    callback_interval: int = 1

    def validate(self) -> None:
        if self.fmax <= 0.0:
            raise ValueError("fmax must be positive")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.dt_start <= 0.0 or self.dt_max <= 0.0:
            raise ValueError("FIRE time steps must be positive")
        if self.dt_start > self.dt_max:
            raise ValueError("dt_start must not exceed dt_max")
        if self.max_step <= 0.0:
            raise ValueError("max_step must be positive")
        if self.callback_interval <= 0:
            raise ValueError("callback_interval must be positive")


def max_force_per_system(state: AseGraphBatch, forces: torch.Tensor) -> torch.Tensor:
    constrained_forces = forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
    atom_force_norm = torch.linalg.vector_norm(constrained_forces, dim=-1)
    return scatter_max(atom_force_norm, state.system_idx, state.n_systems)


def max_generalized_force_per_system(
    state: AseGraphBatch,
    atomic_forces: torch.Tensor,
    cell_forces: torch.Tensor,
) -> torch.Tensor:
    """ASE filter-compatible maximum row norm of atomic and cell forces."""

    atomic_max = scatter_max(
        torch.linalg.vector_norm(atomic_forces, dim=-1),
        state.system_idx,
        state.n_systems,
    )
    cell_max = torch.linalg.vector_norm(cell_forces, dim=-1).amax(dim=1)
    return torch.maximum(atomic_max, cell_max)


def _batched_fire_relax_masked(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 1000,
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    max_step: float = 0.2,
    alpha_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    zero_output_velocities: bool = True,
) -> RelaxationResult:
    """Relax independent systems using vectorized ASE-style FIRE.

    Each graph owns its own ``dt``, ``alpha``, positive-power counter, and
    convergence flag. Converged systems are frozen but remain in the model
    batch; this is deliberately simple and provides a clean baseline for an
    active-batch compaction experiment.
    """

    cfg = FIREConfig(
        fmax=fmax,
        max_steps=max_steps,
        dt_start=dt_start,
        dt_max=dt_max,
        max_step=max_step,
        alpha_start=alpha_start,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        f_alpha=f_alpha,
        callback_interval=callback_interval,
    )
    cfg.validate()

    n_systems = state.n_systems
    device, dtype = state.device, state.dtype

    velocity = torch.zeros_like(state.positions)
    dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)
    n_positive = torch.zeros((n_systems,), device=device, dtype=torch.int64)
    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)

    evaluation = potential(state, neighbor_policy="auto")
    active_batch_sizes = [n_systems]
    first_update = True
    completed_steps = 0

    for step in range(cfg.max_steps + 1):
        forces = evaluation.forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(state, forces)

        newly_converged = (current_fmax <= cfg.fmax) & (converged_step < 0)
        converged_step[newly_converged] = step
        converged = converged_step >= 0
        active_system = ~converged

        diagnostics = {
            "energy": evaluation.energy.detach(),
            "max_force": current_fmax.detach(),
            "converged": converged.detach(),
            "dt": dt.detach(),
            "alpha": alpha.detach(),
            "neighbor_rebuild_count": torch.full(
                (n_systems,),
                state.neighbor_rebuild_count,
                device=device,
                dtype=torch.int64,
            ),
        }
        if callback is not None and step % cfg.callback_interval == 0:
            callback(step, state, evaluation, diagnostics)

        if bool(converged.all()) or step == cfg.max_steps:
            completed_steps = step
            break

        active_atom = active_system[state.system_idx] & state.mobile
        forces = forces.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        velocity = velocity.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        if not first_update:
            power_atom = (forces * velocity).sum(dim=-1)
            power = scatter_sum(power_atom, state.system_idx, n_systems)
            positive = (power > 0.0) & active_system
            negative = (~positive) & active_system

            increase = positive & (n_positive > cfg.n_min)
            dt[increase] = torch.minimum(
                dt[increase] * cfg.f_inc,
                torch.full_like(dt[increase], cfg.dt_max),
            )

            dt[negative] *= cfg.f_dec
            alpha[negative] = cfg.alpha_start
            n_positive[negative] = 0

            velocity_norm = system_l2_norm(velocity, state.system_idx, n_systems)
            force_norm = system_l2_norm(forces, state.system_idx, n_systems)
            eps = 1e-8 if dtype == torch.float32 else 1e-16

            atom_alpha = alpha[state.system_idx].unsqueeze(-1)
            mixing_scale = (
                velocity_norm[state.system_idx] / (force_norm[state.system_idx] + eps)
            ).unsqueeze(-1)
            mixed_velocity = (1.0 - atom_alpha) * velocity + atom_alpha * forces * mixing_scale
            velocity = torch.where(
                positive[state.system_idx].unsqueeze(-1), mixed_velocity, velocity
            )
            velocity = torch.where(
                negative[state.system_idx].unsqueeze(-1),
                torch.zeros_like(velocity),
                velocity,
            )
            # ASE applies the current alpha to velocity mixing, then decays it
            # for the next step after a sufficiently long positive run.
            alpha[increase] *= cfg.f_alpha
            n_positive[positive] += 1

        # FIRE treats force as a fictitious acceleration and does not use mass.
        atom_dt = dt[state.system_idx].unsqueeze(-1)
        velocity = velocity + atom_dt * forces
        displacement = atom_dt * velocity

        # Match ASE FIRE: clip the Euclidean norm of all positional degrees of
        # freedom separately for each graph.
        displacement_norm = system_l2_norm(displacement, state.system_idx, n_systems)
        scale = torch.clamp(cfg.max_step / displacement_norm.clamp_min(1e-30), max=1.0)
        displacement = displacement * scale[state.system_idx].unsqueeze(-1)
        displacement = displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        state.positions = (state.positions + displacement).detach()
        velocity = velocity.detach()
        evaluation = potential(state, neighbor_policy="auto")
        active_batch_sizes.append(n_systems)
        first_update = False
        completed_steps = step + 1

    final_forces = evaluation.forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
    final_fmax = max_force_per_system(state, final_forces)
    final_converged = final_fmax <= cfg.fmax

    if zero_output_velocities:
        state.velocities.zero_()

    return RelaxationResult(
        state=state,
        evaluation=evaluation,
        converged=final_converged,
        converged_step=converged_step,
        max_force=final_fmax,
        steps=completed_steps,
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )


def _batched_fire_relax_compacted(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    cfg: FIREConfig,
    callback: StepCallback | None,
    zero_output_velocities: bool,
) -> RelaxationResult:
    """Run FIRE while physically removing converged graphs from model calls."""

    n_systems = state.n_systems
    device, dtype = state.device, state.dtype
    active_state = state
    active_system_ids = torch.arange(n_systems, device=device, dtype=torch.long)
    active_atom_ids = torch.arange(state.n_atoms, device=device, dtype=torch.long)

    velocity = torch.zeros_like(active_state.positions)
    dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)
    n_positive = torch.zeros((n_systems,), device=device, dtype=torch.int64)

    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)
    full_energy = torch.empty((n_systems,), device=device, dtype=dtype)
    full_forces = torch.empty_like(state.positions)
    full_fmax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    full_alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)

    evaluation = potential(active_state, neighbor_policy="auto")
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    neighbor_rebuilds = active_state.neighbor_rebuild_count
    active_batch_sizes = [n_systems]
    first_update = True
    completed_steps = 0

    def sync_full_outputs(current_fmax: torch.Tensor) -> None:
        if active_state is not state:
            state.positions[active_atom_ids] = active_state.positions
        full_forces[active_atom_ids] = evaluation.forces
        full_energy[active_system_ids] = evaluation.energy
        full_fmax[active_system_ids] = current_fmax
        full_dt[active_system_ids] = dt
        full_alpha[active_system_ids] = alpha

    for step in range(cfg.max_steps + 1):
        forces = evaluation.forces.masked_fill(active_state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(active_state, forces)
        sync_full_outputs(current_fmax)

        newly_converged = current_fmax <= cfg.fmax
        converged_step[active_system_ids[newly_converged]] = step
        converged = converged_step >= 0

        if callback is not None and step % cfg.callback_interval == 0:
            callback(
                step,
                state,
                BatchEvaluation(energy=full_energy, forces=full_forces),
                {
                    "energy": full_energy.detach(),
                    "max_force": full_fmax.detach(),
                    "converged": converged.detach(),
                    "dt": full_dt.detach(),
                    "alpha": full_alpha.detach(),
                    "neighbor_rebuild_count": torch.full(
                        (n_systems,),
                        neighbor_rebuilds,
                        device=device,
                        dtype=torch.int64,
                    ),
                },
            )

        if bool(newly_converged.all()) or step == cfg.max_steps:
            completed_steps = step
            break

        if bool(newly_converged.any()):
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
            active_state = active_state.select_systems(remaining_local, rebuild_neighbors=False)
            active_atom_ids = active_atom_ids[remaining_atoms].clone()
            velocity = velocity[remaining_atoms].clone()
            forces = forces[remaining_atoms].clone()
            selector = torch.as_tensor(remaining_local, device=device, dtype=torch.long)
            active_system_ids = active_system_ids[selector].clone()
            dt = dt[selector].clone()
            alpha = alpha[selector].clone()
            n_positive = n_positive[selector].clone()

        n_active = active_state.n_systems
        active_atom = active_state.mobile
        forces = forces.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        velocity = velocity.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        if not first_update:
            power_atom = (forces * velocity).sum(dim=-1)
            power = scatter_sum(power_atom, active_state.system_idx, n_active)
            positive = power > 0.0
            negative = ~positive

            increase = positive & (n_positive > cfg.n_min)
            dt[increase] = torch.minimum(
                dt[increase] * cfg.f_inc,
                torch.full_like(dt[increase], cfg.dt_max),
            )

            dt[negative] *= cfg.f_dec
            alpha[negative] = cfg.alpha_start
            n_positive[negative] = 0

            velocity_norm = system_l2_norm(velocity, active_state.system_idx, n_active)
            force_norm = system_l2_norm(forces, active_state.system_idx, n_active)
            eps = 1e-8 if dtype == torch.float32 else 1e-16

            atom_alpha = alpha[active_state.system_idx].unsqueeze(-1)
            mixing_scale = (
                velocity_norm[active_state.system_idx] / (force_norm[active_state.system_idx] + eps)
            ).unsqueeze(-1)
            mixed_velocity = (1.0 - atom_alpha) * velocity + atom_alpha * forces * mixing_scale
            velocity = torch.where(
                positive[active_state.system_idx].unsqueeze(-1),
                mixed_velocity,
                velocity,
            )
            velocity = torch.where(
                negative[active_state.system_idx].unsqueeze(-1),
                torch.zeros_like(velocity),
                velocity,
            )
            alpha[increase] *= cfg.f_alpha
            n_positive[positive] += 1

        atom_dt = dt[active_state.system_idx].unsqueeze(-1)
        velocity = velocity + atom_dt * forces
        displacement = atom_dt * velocity
        displacement_norm = system_l2_norm(displacement, active_state.system_idx, n_active)
        scale = torch.clamp(cfg.max_step / displacement_norm.clamp_min(1e-30), max=1.0)
        displacement = displacement * scale[active_state.system_idx].unsqueeze(-1)
        displacement = displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        active_state.positions = (active_state.positions + displacement).detach()
        velocity = velocity.detach()
        rebuilds_before = active_state.neighbor_rebuild_count
        evaluation = potential(active_state, neighbor_policy="auto")
        neighbor_rebuilds += active_state.neighbor_rebuild_count - rebuilds_before
        active_batch_sizes.append(n_active)
        first_update = False
        completed_steps = step + 1

    state._neighbor_reference_positions = None
    state._neighbor_reference_cells = None
    state.neighbor_rebuild_count = neighbor_rebuilds
    if zero_output_velocities:
        state.velocities.zero_()

    return RelaxationResult(
        state=state,
        evaluation=BatchEvaluation(energy=full_energy, forces=full_forces),
        converged=full_fmax <= cfg.fmax,
        converged_step=converged_step,
        max_force=full_fmax,
        steps=completed_steps,
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )


def _batched_fire_relax_variable_cell(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    cfg: FIREConfig,
    cell_filter: FrechetCellFilter,
    smax: float | None,
    callback: StepCallback | None,
    zero_output_velocities: bool,
) -> RelaxationResult:
    """Run masked FIRE over atomic and log-deformation cell coordinates."""

    if smax is not None and smax <= 0.0:
        raise ValueError("smax must be positive")

    bound_filter = cell_filter.bind(state)
    n_systems = state.n_systems
    device, dtype = state.device, state.dtype

    atomic_velocity = torch.zeros_like(state.positions)
    cell_velocity = torch.zeros((n_systems, 3, 3), device=device, dtype=dtype)
    dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)
    n_positive = torch.zeros((n_systems,), device=device, dtype=torch.int64)
    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)

    evaluation = potential(state, neighbor_policy="auto", compute_stress=True)
    active_batch_sizes = [n_systems]
    first_update = True
    completed_steps = 0

    for step in range(cfg.max_steps + 1):
        if evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all()):
            raise FloatingPointError(
                "calculator returned missing or non-finite stress for cell optimization"
            )

        physical_forces = evaluation.forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(state, physical_forces)
        current_smax = bound_filter.max_stress(evaluation)
        atomic_forces, cell_forces = bound_filter.generalized_forces(state, evaluation)
        generalized_fmax = max_generalized_force_per_system(state, atomic_forces, cell_forces)
        if smax is None:
            convergence_now = generalized_fmax < cfg.fmax
        else:
            convergence_now = (current_fmax <= cfg.fmax) & (current_smax <= smax)
        newly_converged = convergence_now & (converged_step < 0)
        converged_step[newly_converged] = step
        converged = converged_step >= 0
        active_system = ~converged

        volumes = bound_filter.volumes(state)
        diagnostics = {
            "energy": evaluation.energy.detach(),
            "enthalpy": (evaluation.energy + bound_filter.pressure * volumes).detach(),
            "max_force": current_fmax.detach(),
            "max_stress": current_smax.detach(),
            "max_generalized_force": generalized_fmax.detach(),
            "stress": evaluation.stress.detach(),
            "volume": volumes.detach(),
            "converged": converged.detach(),
            "dt": dt.detach(),
            "alpha": alpha.detach(),
            "neighbor_rebuild_count": torch.full(
                (n_systems,),
                state.neighbor_rebuild_count,
                device=device,
                dtype=torch.int64,
            ),
        }
        if callback is not None and step % cfg.callback_interval == 0:
            callback(step, state, evaluation, diagnostics)

        if bool(converged.all()) or step == cfg.max_steps:
            completed_steps = step
            break

        active_atom = active_system[state.system_idx] & state.mobile
        atomic_forces = atomic_forces.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        cell_forces = cell_forces.masked_fill(~active_system[:, None, None], 0.0)
        atomic_velocity = atomic_velocity.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        cell_velocity = cell_velocity.masked_fill(~active_system[:, None, None], 0.0)

        if not first_update:
            atomic_power = scatter_sum(
                (atomic_forces * atomic_velocity).sum(dim=-1),
                state.system_idx,
                n_systems,
            )
            cell_power = (cell_forces * cell_velocity).flatten(1).sum(dim=1)
            power = atomic_power + cell_power
            positive = (power > 0.0) & active_system
            negative = (~positive) & active_system

            increase = positive & (n_positive > cfg.n_min)
            dt[increase] = torch.minimum(
                dt[increase] * cfg.f_inc,
                torch.full_like(dt[increase], cfg.dt_max),
            )
            dt[negative] *= cfg.f_dec
            alpha[negative] = cfg.alpha_start
            n_positive[negative] = 0

            velocity_sq = scatter_sum(
                (atomic_velocity * atomic_velocity).sum(dim=-1),
                state.system_idx,
                n_systems,
            ) + (cell_velocity * cell_velocity).flatten(1).sum(dim=1)
            force_sq = scatter_sum(
                (atomic_forces * atomic_forces).sum(dim=-1),
                state.system_idx,
                n_systems,
            ) + (cell_forces * cell_forces).flatten(1).sum(dim=1)
            velocity_norm = torch.sqrt(velocity_sq.clamp_min(0.0))
            force_norm = torch.sqrt(force_sq.clamp_min(0.0))
            eps = 1e-8 if dtype == torch.float32 else 1e-16
            mixing_scale = velocity_norm / (force_norm + eps)

            atom_alpha = alpha[state.system_idx].unsqueeze(-1)
            atom_scale = mixing_scale[state.system_idx].unsqueeze(-1)
            mixed_atomic_velocity = (
                1.0 - atom_alpha
            ) * atomic_velocity + atom_alpha * atomic_forces * atom_scale
            cell_alpha = alpha[:, None, None]
            mixed_cell_velocity = (
                1.0 - cell_alpha
            ) * cell_velocity + cell_alpha * cell_forces * mixing_scale[:, None, None]
            atomic_velocity = torch.where(
                positive[state.system_idx].unsqueeze(-1),
                mixed_atomic_velocity,
                atomic_velocity,
            )
            cell_velocity = torch.where(positive[:, None, None], mixed_cell_velocity, cell_velocity)
            atomic_velocity = torch.where(
                negative[state.system_idx].unsqueeze(-1),
                torch.zeros_like(atomic_velocity),
                atomic_velocity,
            )
            cell_velocity = torch.where(
                negative[:, None, None],
                torch.zeros_like(cell_velocity),
                cell_velocity,
            )
            alpha[increase] *= cfg.f_alpha
            n_positive[positive] += 1

        atom_dt = dt[state.system_idx].unsqueeze(-1)
        cell_dt = dt[:, None, None]
        atomic_velocity = atomic_velocity + atom_dt * atomic_forces
        cell_velocity = cell_velocity + cell_dt * cell_forces
        atomic_displacement = atom_dt * atomic_velocity
        cell_displacement = cell_dt * cell_velocity

        displacement_sq = scatter_sum(
            (atomic_displacement * atomic_displacement).sum(dim=-1),
            state.system_idx,
            n_systems,
        ) + (cell_displacement * cell_displacement).flatten(1).sum(dim=1)
        displacement_norm = torch.sqrt(displacement_sq.clamp_min(0.0))
        scale = torch.clamp(cfg.max_step / displacement_norm.clamp_min(1e-30), max=1.0)
        atomic_displacement *= scale[state.system_idx].unsqueeze(-1)
        cell_displacement *= scale[:, None, None]
        atomic_displacement = atomic_displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        cell_displacement = cell_displacement.masked_fill(~active_system[:, None, None], 0.0)

        bound_filter.apply_displacement(state, atomic_displacement, cell_displacement)
        atomic_velocity = atomic_velocity.detach()
        cell_velocity = cell_velocity.detach()
        evaluation = potential(state, neighbor_policy="auto", compute_stress=True)
        active_batch_sizes.append(n_systems)
        first_update = False
        completed_steps = step + 1

    final_forces = evaluation.forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
    final_fmax = max_force_per_system(state, final_forces)
    final_smax = bound_filter.max_stress(evaluation)
    final_atomic_forces, final_cell_forces = bound_filter.generalized_forces(state, evaluation)
    final_generalized_fmax = max_generalized_force_per_system(
        state, final_atomic_forces, final_cell_forces
    )
    final_converged = (
        final_generalized_fmax < cfg.fmax
        if smax is None
        else (final_fmax <= cfg.fmax) & (final_smax <= smax)
    )
    if zero_output_velocities:
        state.velocities.zero_()

    return RelaxationResult(
        state=state,
        evaluation=evaluation,
        converged=final_converged,
        converged_step=converged_step,
        max_force=final_fmax,
        max_stress=final_smax,
        steps=completed_steps,
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )


def _batched_fire_relax_variable_cell_compacted(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    cfg: FIREConfig,
    cell_filter: FrechetCellFilter,
    smax: float | None,
    callback: StepCallback | None,
    zero_output_velocities: bool,
) -> RelaxationResult:
    """Run variable-cell FIRE while removing converged graph/cell states."""

    if smax is not None and smax <= 0.0:
        raise ValueError("smax must be positive")

    n_systems = state.n_systems
    device, dtype = state.device, state.dtype
    active_state = state
    active_filter = cell_filter.bind(active_state)
    active_system_ids = torch.arange(n_systems, device=device, dtype=torch.long)
    active_atom_ids = torch.arange(state.n_atoms, device=device, dtype=torch.long)
    full_pressure = active_filter.pressure.clone()

    atomic_velocity = torch.zeros_like(active_state.positions)
    cell_velocity = torch.zeros((n_systems, 3, 3), device=device, dtype=dtype)
    dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)
    n_positive = torch.zeros((n_systems,), device=device, dtype=torch.int64)

    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)
    full_energy = torch.empty((n_systems,), device=device, dtype=dtype)
    full_forces = torch.empty_like(state.positions)
    full_stress = torch.empty((n_systems, 3, 3), device=device, dtype=dtype)
    full_fmax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_smax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_generalized_fmax = torch.full((n_systems,), torch.inf, device=device, dtype=dtype)
    full_dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    full_alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)

    evaluation = potential(active_state, neighbor_policy="auto", compute_stress=True)
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    neighbor_rebuilds = active_state.neighbor_rebuild_count
    active_batch_sizes = [n_systems]
    first_update = True
    completed_steps = 0

    def sync_full_outputs(
        current_fmax: torch.Tensor,
        current_smax: torch.Tensor,
        current_generalized_fmax: torch.Tensor,
    ) -> None:
        if active_state is not state:
            state.positions[active_atom_ids] = active_state.positions
            state.cells[active_system_ids] = active_state.cells
        full_forces[active_atom_ids] = evaluation.forces
        full_energy[active_system_ids] = evaluation.energy
        full_stress[active_system_ids] = evaluation.stress
        full_fmax[active_system_ids] = current_fmax
        full_smax[active_system_ids] = current_smax
        full_generalized_fmax[active_system_ids] = current_generalized_fmax
        full_dt[active_system_ids] = dt
        full_alpha[active_system_ids] = alpha

    for step in range(cfg.max_steps + 1):
        if evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all()):
            raise FloatingPointError(
                "calculator returned missing or non-finite stress for cell optimization"
            )

        physical_forces = evaluation.forces.masked_fill(active_state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(active_state, physical_forces)
        current_smax = active_filter.max_stress(evaluation)
        atomic_forces, cell_forces = active_filter.generalized_forces(active_state, evaluation)
        current_generalized_fmax = max_generalized_force_per_system(
            active_state, atomic_forces, cell_forces
        )
        sync_full_outputs(current_fmax, current_smax, current_generalized_fmax)

        if smax is None:
            newly_converged = current_generalized_fmax < cfg.fmax
        else:
            newly_converged = (current_fmax <= cfg.fmax) & (current_smax <= smax)
        converged_step[active_system_ids[newly_converged]] = step
        converged = converged_step >= 0

        if callback is not None and step % cfg.callback_interval == 0:
            volumes = torch.linalg.det(state.cells).abs()
            callback(
                step,
                state,
                BatchEvaluation(
                    energy=full_energy,
                    forces=full_forces,
                    stress=full_stress,
                ),
                {
                    "energy": full_energy.detach(),
                    "enthalpy": (full_energy + full_pressure * volumes).detach(),
                    "max_force": full_fmax.detach(),
                    "max_stress": full_smax.detach(),
                    "max_generalized_force": full_generalized_fmax.detach(),
                    "stress": full_stress.detach(),
                    "volume": volumes.detach(),
                    "converged": converged.detach(),
                    "dt": full_dt.detach(),
                    "alpha": full_alpha.detach(),
                    "neighbor_rebuild_count": torch.full(
                        (n_systems,),
                        neighbor_rebuilds,
                        device=device,
                        dtype=torch.int64,
                    ),
                },
            )

        if bool(newly_converged.all()) or step == cfg.max_steps:
            completed_steps = step
            break

        if bool(newly_converged.any()):
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

            next_filter = active_filter.select_systems(active_state, remaining_local)
            next_state = active_state.select_systems(remaining_local, rebuild_neighbors=False)
            evaluation = BatchEvaluation(
                energy=evaluation.energy[selector].clone(),
                forces=evaluation.forces[remaining_atoms].clone(),
                stress=evaluation.stress[selector].clone(),
            )
            atomic_forces = atomic_forces[remaining_atoms].clone()
            cell_forces = cell_forces[selector].clone()
            active_atom_ids = active_atom_ids[remaining_atoms].clone()
            active_system_ids = active_system_ids[selector].clone()
            atomic_velocity = atomic_velocity[remaining_atoms].clone()
            cell_velocity = cell_velocity[selector].clone()
            dt = dt[selector].clone()
            alpha = alpha[selector].clone()
            n_positive = n_positive[selector].clone()
            active_state = next_state
            active_filter = next_filter

        n_active = active_state.n_systems
        active_atom = active_state.mobile
        atomic_forces = atomic_forces.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        atomic_velocity = atomic_velocity.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        if not first_update:
            atomic_power = scatter_sum(
                (atomic_forces * atomic_velocity).sum(dim=-1),
                active_state.system_idx,
                n_active,
            )
            power = atomic_power + (cell_forces * cell_velocity).flatten(1).sum(dim=1)
            positive = power > 0.0
            negative = ~positive

            increase = positive & (n_positive > cfg.n_min)
            dt[increase] = torch.minimum(
                dt[increase] * cfg.f_inc,
                torch.full_like(dt[increase], cfg.dt_max),
            )
            dt[negative] *= cfg.f_dec
            alpha[negative] = cfg.alpha_start
            n_positive[negative] = 0

            velocity_sq = scatter_sum(
                (atomic_velocity * atomic_velocity).sum(dim=-1),
                active_state.system_idx,
                n_active,
            ) + (cell_velocity * cell_velocity).flatten(1).sum(dim=1)
            force_sq = scatter_sum(
                (atomic_forces * atomic_forces).sum(dim=-1),
                active_state.system_idx,
                n_active,
            ) + (cell_forces * cell_forces).flatten(1).sum(dim=1)
            velocity_norm = torch.sqrt(velocity_sq.clamp_min(0.0))
            force_norm = torch.sqrt(force_sq.clamp_min(0.0))
            eps = 1e-8 if dtype == torch.float32 else 1e-16
            mixing_scale = velocity_norm / (force_norm + eps)

            atom_alpha = alpha[active_state.system_idx].unsqueeze(-1)
            atom_scale = mixing_scale[active_state.system_idx].unsqueeze(-1)
            mixed_atomic_velocity = (
                1.0 - atom_alpha
            ) * atomic_velocity + atom_alpha * atomic_forces * atom_scale
            cell_alpha = alpha[:, None, None]
            mixed_cell_velocity = (
                1.0 - cell_alpha
            ) * cell_velocity + cell_alpha * cell_forces * mixing_scale[:, None, None]
            atomic_velocity = torch.where(
                positive[active_state.system_idx].unsqueeze(-1),
                mixed_atomic_velocity,
                atomic_velocity,
            )
            cell_velocity = torch.where(positive[:, None, None], mixed_cell_velocity, cell_velocity)
            atomic_velocity = torch.where(
                negative[active_state.system_idx].unsqueeze(-1),
                torch.zeros_like(atomic_velocity),
                atomic_velocity,
            )
            cell_velocity = torch.where(
                negative[:, None, None],
                torch.zeros_like(cell_velocity),
                cell_velocity,
            )
            alpha[increase] *= cfg.f_alpha
            n_positive[positive] += 1

        atom_dt = dt[active_state.system_idx].unsqueeze(-1)
        cell_dt = dt[:, None, None]
        atomic_velocity = atomic_velocity + atom_dt * atomic_forces
        cell_velocity = cell_velocity + cell_dt * cell_forces
        atomic_displacement = atom_dt * atomic_velocity
        cell_displacement = cell_dt * cell_velocity

        displacement_sq = scatter_sum(
            (atomic_displacement * atomic_displacement).sum(dim=-1),
            active_state.system_idx,
            n_active,
        ) + (cell_displacement * cell_displacement).flatten(1).sum(dim=1)
        displacement_norm = torch.sqrt(displacement_sq.clamp_min(0.0))
        scale = torch.clamp(cfg.max_step / displacement_norm.clamp_min(1e-30), max=1.0)
        atomic_displacement *= scale[active_state.system_idx].unsqueeze(-1)
        cell_displacement *= scale[:, None, None]
        atomic_displacement = atomic_displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)

        active_filter.apply_displacement(active_state, atomic_displacement, cell_displacement)
        atomic_velocity = atomic_velocity.detach()
        cell_velocity = cell_velocity.detach()
        rebuilds_before = active_state.neighbor_rebuild_count
        evaluation = potential(active_state, neighbor_policy="auto", compute_stress=True)
        neighbor_rebuilds += active_state.neighbor_rebuild_count - rebuilds_before
        active_batch_sizes.append(n_active)
        first_update = False
        completed_steps = step + 1

    state._neighbor_reference_positions = None
    state._neighbor_reference_cells = None
    state.neighbor_rebuild_count = neighbor_rebuilds
    if zero_output_velocities:
        state.velocities.zero_()

    return RelaxationResult(
        state=state,
        evaluation=BatchEvaluation(
            energy=full_energy,
            forces=full_forces,
            stress=full_stress,
        ),
        converged=(
            full_generalized_fmax < cfg.fmax
            if smax is None
            else (full_fmax <= cfg.fmax) & (full_smax <= smax)
        ),
        converged_step=converged_step,
        max_force=full_fmax,
        max_stress=full_smax,
        steps=completed_steps,
        model_evaluations=len(active_batch_sizes),
        graph_evaluations=sum(active_batch_sizes),
        active_batch_sizes=tuple(active_batch_sizes),
    )


def _batched_fire_refill_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    cfg: FIREConfig,
    refill_batch_size: int,
    refill_policy: str,
    refill_storage: str,
    refill_low_watermark: float,
    refill_min_chunk: int,
    refill_interval: int,
    refill_tail_compaction_threshold: float | None,
    cell_filter: FrechetCellFilter | None,
    smax: float | None,
    callback: StepCallback | None,
    zero_output_velocities: bool,
) -> RelaxationResult:
    """Run FIRE with bounded resident state and optimizer-safe admission."""

    if smax is not None and smax <= 0.0:
        raise ValueError("smax must be positive")

    n_systems = state.n_systems
    capacity = min(refill_batch_size, n_systems)
    device, dtype = state.device, state.dtype
    active_system_ids = torch.arange(capacity, device=device, dtype=torch.long)
    next_pending = capacity
    active_atom_ids = global_atom_ids(state, active_system_ids)
    active_state = (
        state
        if capacity == n_systems
        else state.select_systems(active_system_ids.tolist(), rebuild_neighbors=False)
    )

    full_filter = None if cell_filter is None else cell_filter.bind(state)
    active_filter = (
        None
        if full_filter is None
        else (
            full_filter
            if active_state is state
            else full_filter.select_systems(state, active_system_ids.tolist())
        )
    )
    full_pressure = None if full_filter is None else full_filter.pressure.detach().clone()

    atomic_velocity = torch.zeros_like(active_state.positions)
    cell_velocity = (
        None
        if active_filter is None
        else torch.zeros((active_state.n_systems, 3, 3), device=device, dtype=dtype)
    )
    dt = torch.full((active_state.n_systems,), cfg.dt_start, device=device, dtype=dtype)
    alpha = torch.full((active_state.n_systems,), cfg.alpha_start, device=device, dtype=dtype)
    n_positive = torch.zeros((active_state.n_systems,), device=device, dtype=torch.int64)
    local_steps = torch.zeros((n_systems,), device=device, dtype=torch.int64)
    finished = torch.zeros((n_systems,), device=device, dtype=torch.bool)
    converged_step = torch.full((n_systems,), -1, device=device, dtype=torch.int64)

    full_energy = torch.full((n_systems,), torch.nan, device=device, dtype=dtype)
    full_forces = torch.full_like(state.positions, torch.nan)
    full_stress = (
        None
        if active_filter is None
        else torch.full((n_systems, 3, 3), torch.nan, device=device, dtype=dtype)
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
    full_dt = torch.full((n_systems,), cfg.dt_start, device=device, dtype=dtype)
    full_alpha = torch.full((n_systems,), cfg.alpha_start, device=device, dtype=dtype)

    neighbor_rebuilds = state.neighbor_rebuild_count

    def evaluate_active(scheduler_step: int) -> BatchEvaluation:
        nonlocal neighbor_rebuilds
        rebuilds_before = active_state.neighbor_rebuild_count
        current = potential(
            active_state,
            neighbor_policy="auto",
            compute_stress=active_filter is not None,
        )
        neighbor_rebuilds += active_state.neighbor_rebuild_count - rebuilds_before
        profile_event(
            "optimizer_evaluation",
            optimizer="fire",
            scheduler_step=scheduler_step,
            active_systems=active_state.n_systems,
            active_atoms=active_state.n_atoms,
            active_edges=active_state.edge_index.shape[1],
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
        full_dt[active_system_ids] = dt
        full_alpha[active_system_ids] = alpha
        if active_filter is not None:
            if (
                full_filter is None
                or full_stress is None
                or full_smax is None
                or full_generalized_fmax is None
                or evaluation.stress is None
            ):
                raise RuntimeError("variable-cell FIRE state is incomplete")
            if active_filter is not full_filter:
                full_filter.generalized_positions[active_atom_ids] = (
                    active_filter.generalized_positions
                )
                full_filter.log_deformation[active_system_ids] = active_filter.log_deformation
            full_stress[active_system_ids] = evaluation.stress
            if current_smax is not None:
                full_smax[active_system_ids] = current_smax
            if current_generalized_fmax is not None:
                full_generalized_fmax[active_system_ids] = current_generalized_fmax

    evaluation = evaluate_active(0)
    if full_energy.dtype != evaluation.energy.dtype:
        full_energy = full_energy.to(evaluation.energy.dtype)
    active_batch_sizes = [active_state.n_systems]
    scheduler_step = 0
    tail_reference_size = active_state.n_systems

    while True:
        if active_filter is not None and (
            evaluation.stress is None or not bool(torch.isfinite(evaluation.stress).all())
        ):
            raise FloatingPointError(
                "calculator returned missing or non-finite stress for cell optimization"
            )

        physical_forces = evaluation.forces.masked_fill(active_state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(active_state, physical_forces)
        if active_filter is None:
            atomic_forces = physical_forces
            cell_forces = None
            current_smax = None
            current_generalized_fmax = None
            convergence_now = current_fmax <= cfg.fmax
        else:
            atomic_forces, cell_forces = active_filter.generalized_forces(active_state, evaluation)
            current_smax = active_filter.max_stress(evaluation)
            current_generalized_fmax = max_generalized_force_per_system(
                active_state, atomic_forces, cell_forces
            )
            convergence_now = (
                current_generalized_fmax < cfg.fmax
                if smax is None
                else (current_fmax <= cfg.fmax) & (current_smax <= smax)
            )

        sync_active_state(evaluation, current_fmax, current_smax, current_generalized_fmax)
        resident_finished_before = finished[active_system_ids]
        exhausted_now = local_steps[active_system_ids] >= cfg.max_steps
        finish_now = (convergence_now | exhausted_now) & ~resident_finished_before
        newly_converged = convergence_now & ~resident_finished_before
        converged_ids = active_system_ids[newly_converged]
        converged_step[converged_ids] = local_steps[converged_ids]
        finished[active_system_ids[finish_now]] = True

        diagnostics = {
            "energy": full_energy.detach(),
            "max_force": full_fmax.detach(),
            "converged": (converged_step >= 0).detach(),
            "finished": finished.detach(),
            "local_steps": local_steps.detach(),
            "dt": full_dt.detach(),
            "alpha": full_alpha.detach(),
            "neighbor_rebuild_count": torch.full(
                (n_systems,),
                neighbor_rebuilds,
                device=device,
                dtype=torch.int64,
            ),
        }
        if full_stress is not None:
            if full_pressure is None or full_smax is None or full_generalized_fmax is None:
                raise RuntimeError("variable-cell FIRE diagnostics are incomplete")
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
        if callback is not None and scheduler_step % cfg.callback_interval == 0:
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

        resident_finished = finished[active_system_ids]
        ready_local_ids = torch.nonzero(~resident_finished, as_tuple=False).flatten()
        ready_count = ready_local_ids.numel()
        if refill_tail_compaction_threshold is None:
            refill_due = bool(resident_finished.any()) and (
                (scheduler_step + 1) % refill_interval == 0
                or not bool((~resident_finished).any())
            )
        elif next_pending < n_systems:
            refill_due = bool(resident_finished.any())
        else:
            tail_limit = max(
                1,
                int(tail_reference_size * refill_tail_compaction_threshold),
            )
            refill_due = bool(resident_finished.any()) and ready_count <= tail_limit
        if refill_due:
            systems_before = active_state.n_systems
            remaining_local = torch.nonzero(~resident_finished, as_tuple=False).flatten()
            remaining_list = remaining_local.tolist()
            survivor_ids = active_system_ids[remaining_local]
            finished_local = torch.nonzero(resident_finished, as_tuple=False).flatten()
            pending_before = n_systems - next_pending
            insert_count = refill_insert_count(
                policy=refill_policy,
                capacity=capacity,
                survivors=len(remaining_list),
                pending=pending_before,
                low_watermark=refill_low_watermark,
                min_chunk=refill_min_chunk,
            )
            refill_stop = next_pending + insert_count
            refill_ids = torch.arange(next_pending, refill_stop, device=device, dtype=torch.long)
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
            phase = "scheduler.refill_slot_swap" if use_slot_swap else "scheduler.refill_repack"
            with profile_phase(
                phase,
                device=device,
                systems=systems_before,
                atoms=active_state.n_atoms,
            ):
                if use_slot_swap:
                    destination_list = finished_local.tolist()
                    source_list = refill_ids.tolist()
                    active_state.replace_systems_from_(destination_list, state, source_list)
                    active_system_ids = active_system_ids.clone()
                    active_system_ids[finished_local] = refill_ids
                    active_atom_ids = global_atom_ids(state, active_system_ids)
                    for destination, source_id in zip(destination_list, source_list, strict=True):
                        destination_atoms = active_state.atom_slice(destination)
                        source_atoms = state.atom_slice(source_id)
                        atomic_velocity[destination_atoms] = 0.0
                        atomic_forces[destination_atoms] = 0.0
                        dt[destination] = cfg.dt_start
                        alpha[destination] = cfg.alpha_start
                        n_positive[destination] = 0
                        if active_filter is not None:
                            if full_filter is None or cell_velocity is None or cell_forces is None:
                                raise RuntimeError("full variable-cell FIRE state is missing")
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
                            cell_velocity[destination] = 0.0
                            cell_forces[destination] = 0.0
                    ready_local_ids = remaining_local
                else:
                    remaining_atom_ids = global_atom_ids(active_state, remaining_local)
                    survivor_atomic_velocity = atomic_velocity[remaining_atom_ids].clone()
                    survivor_atomic_forces = atomic_forces[remaining_atom_ids].clone()
                    survivor_cell_velocity = (
                        None if cell_velocity is None else cell_velocity[remaining_local].clone()
                    )
                    survivor_cell_forces = (
                        None if cell_forces is None else cell_forces[remaining_local].clone()
                    )
                    survivor_dt = dt[remaining_local].clone()
                    survivor_alpha = alpha[remaining_local].clone()
                    survivor_n_positive = n_positive[remaining_local].clone()
                    active_system_ids = torch.cat((survivor_ids, refill_ids))
                    active_atom_ids = global_atom_ids(state, active_system_ids)
                    state_parts = []
                    if remaining_list:
                        state_parts.append(
                            active_state.select_systems(remaining_list, rebuild_neighbors=False)
                        )
                    if refill_ids.numel():
                        state_parts.append(
                            state.select_systems(refill_ids.tolist(), rebuild_neighbors=False)
                        )
                    active_state = AseGraphBatch.concatenate(state_parts)
                    active_filter = (
                        None
                        if full_filter is None
                        else full_filter.select_systems(state, active_system_ids.tolist())
                    )
                    atomic_velocity = torch.zeros_like(active_state.positions)
                    atomic_velocity[: survivor_atomic_velocity.shape[0]] = survivor_atomic_velocity
                    atomic_forces = torch.zeros_like(active_state.positions)
                    atomic_forces[: survivor_atomic_forces.shape[0]] = survivor_atomic_forces
                    dt = torch.full(
                        (active_state.n_systems,),
                        cfg.dt_start,
                        device=device,
                        dtype=dtype,
                    )
                    alpha = torch.full(
                        (active_state.n_systems,),
                        cfg.alpha_start,
                        device=device,
                        dtype=dtype,
                    )
                    n_positive = torch.zeros(
                        (active_state.n_systems,),
                        device=device,
                        dtype=torch.int64,
                    )
                    survivor_count = len(remaining_list)
                    dt[:survivor_count] = survivor_dt
                    alpha[:survivor_count] = survivor_alpha
                    n_positive[:survivor_count] = survivor_n_positive
                    if active_filter is None:
                        cell_velocity = None
                        cell_forces = None
                    else:
                        cell_velocity = torch.zeros(
                            (active_state.n_systems, 3, 3),
                            device=device,
                            dtype=dtype,
                        )
                        cell_forces = torch.zeros_like(cell_velocity)
                        if survivor_cell_velocity is not None:
                            cell_velocity[:survivor_count] = survivor_cell_velocity
                        if survivor_cell_forces is not None:
                            cell_forces[:survivor_count] = survivor_cell_forces
                    ready_local_ids = torch.arange(survivor_count, device=device, dtype=torch.long)
            profile_event(
                "refill",
                optimizer="fire",
                policy=refill_policy,
                interval=refill_interval,
                tail_compaction_threshold=refill_tail_compaction_threshold,
                tail_compaction=(
                    refill_tail_compaction_threshold is not None and pending_before == 0
                ),
                low_watermark=refill_low_watermark,
                min_chunk=refill_min_chunk,
                scheduler_step=scheduler_step,
                systems_before=systems_before,
                survivors=ready_local_ids.numel(),
                inserted=refill_ids.numel(),
                triggered=bool(refill_ids.numel()),
                storage="slots" if use_slot_swap else "repack",
                systems_after=active_state.n_systems,
                pending_after=n_systems - next_pending,
            )
            if refill_tail_compaction_threshold is not None and next_pending == n_systems:
                tail_reference_size = active_state.n_systems

        with profile_phase(
            "optimizer.fire_update",
            device=device,
            systems=active_state.n_systems,
            atoms=active_state.n_atoms,
        ):
            update_system = torch.zeros((active_state.n_systems,), device=device, dtype=torch.bool)
            update_system[ready_local_ids] = True
            experienced = update_system & (local_steps[active_system_ids] > 0)
            active_atom = update_system[active_state.system_idx] & active_state.mobile
            atomic_forces = atomic_forces.masked_fill(~active_atom.unsqueeze(-1), 0.0)
            atomic_velocity = atomic_velocity.masked_fill(~active_atom.unsqueeze(-1), 0.0)
            if cell_forces is not None and cell_velocity is not None:
                cell_forces = cell_forces.masked_fill(~update_system[:, None, None], 0.0)
                cell_velocity = cell_velocity.masked_fill(~update_system[:, None, None], 0.0)

            if bool(experienced.any()):
                atomic_power = scatter_sum(
                    (atomic_forces * atomic_velocity).sum(dim=-1),
                    active_state.system_idx,
                    active_state.n_systems,
                )
                power = atomic_power
                if cell_forces is not None and cell_velocity is not None:
                    power = power + (cell_forces * cell_velocity).flatten(1).sum(dim=1)
                positive = (power > 0.0) & experienced
                negative = (~positive) & experienced
                increase = positive & (n_positive > cfg.n_min)
                dt[increase] = torch.minimum(
                    dt[increase] * cfg.f_inc,
                    torch.full_like(dt[increase], cfg.dt_max),
                )
                dt[negative] *= cfg.f_dec
                alpha[negative] = cfg.alpha_start
                n_positive[negative] = 0

                velocity_sq = scatter_sum(
                    (atomic_velocity * atomic_velocity).sum(dim=-1),
                    active_state.system_idx,
                    active_state.n_systems,
                )
                force_sq = scatter_sum(
                    (atomic_forces * atomic_forces).sum(dim=-1),
                    active_state.system_idx,
                    active_state.n_systems,
                )
                if cell_forces is not None and cell_velocity is not None:
                    velocity_sq = velocity_sq + (cell_velocity * cell_velocity).flatten(1).sum(
                        dim=1
                    )
                    force_sq = force_sq + (cell_forces * cell_forces).flatten(1).sum(dim=1)
                velocity_norm = torch.sqrt(velocity_sq.clamp_min(0.0))
                force_norm = torch.sqrt(force_sq.clamp_min(0.0))
                eps = 1e-8 if dtype == torch.float32 else 1e-16
                mixing_scale = velocity_norm / (force_norm + eps)
                atom_alpha = alpha[active_state.system_idx].unsqueeze(-1)
                atom_scale = mixing_scale[active_state.system_idx].unsqueeze(-1)
                mixed_atomic_velocity = (
                    1.0 - atom_alpha
                ) * atomic_velocity + atom_alpha * atomic_forces * atom_scale
                atomic_velocity = torch.where(
                    positive[active_state.system_idx].unsqueeze(-1),
                    mixed_atomic_velocity,
                    atomic_velocity,
                )
                atomic_velocity = torch.where(
                    negative[active_state.system_idx].unsqueeze(-1),
                    torch.zeros_like(atomic_velocity),
                    atomic_velocity,
                )
                if cell_forces is not None and cell_velocity is not None:
                    cell_alpha = alpha[:, None, None]
                    mixed_cell_velocity = (
                        1.0 - cell_alpha
                    ) * cell_velocity + cell_alpha * cell_forces * mixing_scale[:, None, None]
                    cell_velocity = torch.where(
                        positive[:, None, None],
                        mixed_cell_velocity,
                        cell_velocity,
                    )
                    cell_velocity = torch.where(
                        negative[:, None, None],
                        torch.zeros_like(cell_velocity),
                        cell_velocity,
                    )
                alpha[increase] *= cfg.f_alpha
                n_positive[positive] += 1

            atom_dt = dt[active_state.system_idx].unsqueeze(-1)
            atomic_velocity = atomic_velocity + atom_dt * atomic_forces
            atomic_displacement = atom_dt * atomic_velocity
            if cell_forces is None or cell_velocity is None:
                displacement_sq = scatter_sum(
                    (atomic_displacement * atomic_displacement).sum(dim=-1),
                    active_state.system_idx,
                    active_state.n_systems,
                )
                cell_displacement = None
            else:
                cell_dt = dt[:, None, None]
                cell_velocity = cell_velocity + cell_dt * cell_forces
                cell_displacement = cell_dt * cell_velocity
                displacement_sq = scatter_sum(
                    (atomic_displacement * atomic_displacement).sum(dim=-1),
                    active_state.system_idx,
                    active_state.n_systems,
                ) + (cell_displacement * cell_displacement).flatten(1).sum(dim=1)
            displacement_norm = torch.sqrt(displacement_sq.clamp_min(0.0))
            scale = torch.clamp(cfg.max_step / displacement_norm.clamp_min(1e-30), max=1.0)
            atomic_displacement *= scale[active_state.system_idx].unsqueeze(-1)
            atomic_displacement = atomic_displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)
            stepped_ids = active_system_ids[ready_local_ids]
            local_steps[stepped_ids] += 1
            if active_filter is None:
                active_state.positions = (active_state.positions + atomic_displacement).detach()
            else:
                if cell_displacement is None:
                    raise RuntimeError("variable-cell displacement is missing")
                cell_displacement *= scale[:, None, None]
                active_filter.apply_displacement(
                    active_state, atomic_displacement, cell_displacement
                )
            atomic_velocity = atomic_velocity.detach()
            if cell_velocity is not None:
                cell_velocity = cell_velocity.detach()

        evaluation = evaluate_active(scheduler_step + 1)
        active_batch_sizes.append(active_state.n_systems)
        scheduler_step += 1

    state._neighbor_reference_positions = None
    state._neighbor_reference_cells = None
    state.neighbor_rebuild_count = neighbor_rebuilds
    if zero_output_velocities:
        state.velocities.zero_()

    if full_generalized_fmax is not None and smax is None:
        final_converged = full_generalized_fmax < cfg.fmax
    elif full_smax is not None:
        final_converged = (full_fmax <= cfg.fmax) & (full_smax <= smax)
    else:
        final_converged = full_fmax <= cfg.fmax
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


def batched_fire_relax(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 1000,
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    max_step: float = 0.2,
    alpha_start: float = 0.1,
    n_min: int = 5,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    zero_output_velocities: bool = True,
    active_compaction: bool = False,
    cell_filter: FrechetCellFilter | None = None,
    smax: float | None = 0.005,
    refill_batch_size: int | None = None,
    refill_policy: str = "immediate",
    refill_storage: str = "repack",
    refill_low_watermark: float = 0.8,
    refill_min_chunk: int | None = None,
    refill_interval: int = 1,
    refill_tail_compaction_threshold: float | None = None,
) -> RelaxationResult:
    """Relax independent systems with optional removal of converged graphs.

    The default masked mode preserves the original behavior. With
    ``active_compaction=True``, each model call contains only unconverged
    systems while returned tensors retain the original graph and atom order.
    """

    cfg = FIREConfig(
        fmax=fmax,
        max_steps=max_steps,
        dt_start=dt_start,
        dt_max=dt_max,
        max_step=max_step,
        alpha_start=alpha_start,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        f_alpha=f_alpha,
        callback_interval=callback_interval,
    )
    cfg.validate()
    if refill_policy not in REFILL_POLICIES:
        choices = ", ".join(sorted(REFILL_POLICIES))
        raise ValueError(f"refill_policy must be one of: {choices}")
    if refill_storage not in REFILL_STORAGE_MODES:
        choices = ", ".join(sorted(REFILL_STORAGE_MODES))
        raise ValueError(f"refill_storage must be one of: {choices}")
    if not 0.0 <= refill_low_watermark < 1.0:
        raise ValueError("refill_low_watermark must be in [0, 1)")
    if refill_min_chunk is not None and (
        isinstance(refill_min_chunk, bool)
        or not isinstance(refill_min_chunk, int)
        or refill_min_chunk <= 0
    ):
        raise ValueError("refill_min_chunk must be a positive integer or None")
    if (
        isinstance(refill_interval, bool)
        or not isinstance(refill_interval, int)
        or refill_interval <= 0
    ):
        raise ValueError("refill_interval must be a positive integer")
    if refill_tail_compaction_threshold is not None and not (
        0.0 < refill_tail_compaction_threshold < 1.0
    ):
        raise ValueError("refill_tail_compaction_threshold must be in (0, 1) or None")
    if refill_tail_compaction_threshold is not None and (
        refill_policy != "immediate" or refill_interval != 1
    ):
        raise ValueError(
            "tail compaction requires refill_policy='immediate' and refill_interval=1"
        )
    if refill_batch_size is None and (
        refill_policy != "immediate"
        or refill_min_chunk is not None
        or refill_interval != 1
        or refill_tail_compaction_threshold is not None
    ):
        raise ValueError("refill policy options require refill_batch_size")
    if refill_batch_size is not None:
        if (
            isinstance(refill_batch_size, bool)
            or not isinstance(refill_batch_size, int)
            or refill_batch_size <= 0
        ):
            raise ValueError("refill_batch_size must be a positive integer")
        return _batched_fire_refill_relax(
            state,
            potential,
            cfg=cfg,
            refill_batch_size=refill_batch_size,
            refill_policy=refill_policy,
            refill_storage=refill_storage,
            refill_low_watermark=refill_low_watermark,
            refill_min_chunk=(
                max(8, refill_batch_size // 8) if refill_min_chunk is None else refill_min_chunk
            ),
            refill_interval=refill_interval,
            refill_tail_compaction_threshold=refill_tail_compaction_threshold,
            cell_filter=cell_filter,
            smax=smax,
            callback=callback,
            zero_output_velocities=zero_output_velocities,
        )
    if cell_filter is not None:
        if active_compaction:
            return _batched_fire_relax_variable_cell_compacted(
                state,
                potential,
                cfg=cfg,
                cell_filter=cell_filter,
                smax=smax,
                callback=callback,
                zero_output_velocities=zero_output_velocities,
            )
        return _batched_fire_relax_variable_cell(
            state,
            potential,
            cfg=cfg,
            cell_filter=cell_filter,
            smax=smax,
            callback=callback,
            zero_output_velocities=zero_output_velocities,
        )
    if active_compaction:
        return _batched_fire_relax_compacted(
            state,
            potential,
            cfg=cfg,
            callback=callback,
            zero_output_velocities=zero_output_velocities,
        )
    return _batched_fire_relax_masked(
        state,
        potential,
        fmax=fmax,
        max_steps=max_steps,
        dt_start=dt_start,
        dt_max=dt_max,
        max_step=max_step,
        alpha_start=alpha_start,
        n_min=n_min,
        f_inc=f_inc,
        f_dec=f_dec,
        f_alpha=f_alpha,
        callback=callback,
        callback_interval=callback_interval,
        zero_output_velocities=zero_output_velocities,
    )


def batched_gradient_descent(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    step_size: float = 0.01,
    max_step: float = 0.1,
    fmax: float = 0.05,
    max_steps: int = 1000,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
) -> RelaxationResult:
    """Simple vectorized steepest-descent baseline.

    This is intentionally uncompetitive but useful as a correctness reference
    when experimenting with more sophisticated optimizers.
    """

    if step_size <= 0 or max_step <= 0 or fmax <= 0:
        raise ValueError("step_size, max_step, and fmax must be positive")
    if max_steps < 0 or callback_interval <= 0:
        raise ValueError("invalid max_steps or callback_interval")

    converged_step = torch.full((state.n_systems,), -1, device=state.device, dtype=torch.int64)
    evaluation = potential(state, neighbor_policy="auto")
    completed_steps = 0

    for step in range(max_steps + 1):
        forces = evaluation.forces.masked_fill(state.fixed.unsqueeze(-1), 0.0)
        current_fmax = max_force_per_system(state, forces)
        newly = (current_fmax <= fmax) & (converged_step < 0)
        converged_step[newly] = step
        converged = converged_step >= 0

        if callback is not None and step % callback_interval == 0:
            callback(
                step,
                state,
                evaluation,
                {
                    "energy": evaluation.energy.detach(),
                    "max_force": current_fmax.detach(),
                    "converged": converged.detach(),
                },
            )

        if bool(converged.all()) or step == max_steps:
            completed_steps = step
            break

        active_atom = (~converged)[state.system_idx] & state.mobile
        displacement = step_size * forces
        norm = system_l2_norm(displacement, state.system_idx, state.n_systems)
        scale = torch.clamp(max_step / norm.clamp_min(1e-30), max=1.0)
        displacement = displacement * scale[state.system_idx].unsqueeze(-1)
        displacement = displacement.masked_fill(~active_atom.unsqueeze(-1), 0.0)
        state.positions = (state.positions + displacement).detach()
        evaluation = potential(state, neighbor_policy="auto")
        completed_steps = step + 1

    final_fmax = max_force_per_system(state, evaluation.forces)
    state.velocities.zero_()
    return RelaxationResult(
        state=state,
        evaluation=evaluation,
        converged=final_fmax <= fmax,
        converged_step=converged_step,
        max_force=final_fmax,
        steps=completed_steps,
    )
