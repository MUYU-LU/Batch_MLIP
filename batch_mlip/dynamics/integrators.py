"""Fixed-cell batched molecular-dynamics integrators."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..core.calculator import BatchCalculator
from ..core.math_utils import as_system_parameter
from ..core.state import (
    AMU_A2_PER_FS2_TO_EV,
    EV_PER_A_PER_AMU_TO_A_PER_FS2,
    KB_EV_PER_K,
    AseGraphBatch,
)
from ..core.types import MDResult, StepCallback


def initialize_maxwell_boltzmann(
    state: AseGraphBatch,
    temperature_K: float | Sequence[float] | torch.Tensor,
    *,
    seed: int | Sequence[int] | torch.Tensor | None = None,
    remove_com: bool = True,
    force_exact_temperature: bool = False,
) -> None:
    """Initialize velocities independently for each system in Angstrom/fs."""

    temperatures = as_system_parameter(
        temperature_K,
        n_systems=state.n_systems,
        device=state.device,
        dtype=state.dtype,
        name="temperature_K",
    )
    if bool((temperatures < 0.0).any()):
        raise ValueError("temperature must be non-negative")

    atom_temperature = temperatures[state.system_idx]
    sigma = torch.sqrt(
        KB_EV_PER_K
        * atom_temperature
        / (state.masses * AMU_A2_PER_FS2_TO_EV)
    )
    if seed is None or isinstance(seed, int):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=state.device)
            generator.manual_seed(seed)
        noise = torch.randn(
            state.positions.shape,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
    else:
        seed_values = torch.as_tensor(seed, device="cpu", dtype=torch.int64).reshape(-1)
        if seed_values.numel() != state.n_systems:
            raise ValueError("seed sequence must contain one value per system")
        if bool((seed_values < 0).any()):
            raise ValueError("per-system seeds must be non-negative")
        noise = torch.empty_like(state.positions)
        for system_id, seed_value in enumerate(seed_values.tolist()):
            generator = torch.Generator(device=state.device)
            generator.manual_seed(seed_value)
            atom_slice = state.atom_slice(system_id)
            noise[atom_slice] = torch.randn(
                noise[atom_slice].shape,
                device=state.device,
                dtype=state.dtype,
                generator=generator,
            )
    state.velocities = noise * sigma.unsqueeze(-1)
    state.zero_fixed_motion_()

    if remove_com:
        state.remove_center_of_mass_velocity_()

    if force_exact_temperature:
        current = state.temperature(com_removed=remove_com)
        scale = torch.where(
            temperatures > 0.0,
            torch.sqrt(temperatures / current.clamp_min(1e-30)),
            torch.zeros_like(temperatures),
        )
        state.velocities *= scale[state.system_idx].unsqueeze(-1)
        state.zero_fixed_motion_()


def _acceleration(state: AseGraphBatch, forces: torch.Tensor) -> torch.Tensor:
    acceleration = (
        forces / state.masses.unsqueeze(-1) * EV_PER_A_PER_AMU_TO_A_PER_FS2
    )
    return acceleration.masked_fill(state.fixed.unsqueeze(-1), 0.0)


def _md_diagnostics(
    state: AseGraphBatch,
    potential_energy: torch.Tensor,
    *,
    initial_total_energy: torch.Tensor | None,
    com_removed_for_temperature: bool,
) -> dict[str, torch.Tensor]:
    kinetic = state.kinetic_energy().detach()
    total = potential_energy.detach() + kinetic
    diagnostics = {
        "potential_energy": potential_energy.detach(),
        "kinetic_energy": kinetic,
        "total_energy": total,
        "temperature": state.temperature(
            com_removed=com_removed_for_temperature
        ).detach(),
        "neighbor_rebuild_count": torch.full(
            (state.n_systems,),
            state.neighbor_rebuild_count,
            device=state.device,
            dtype=torch.int64,
        ),
    }
    if initial_total_energy is not None:
        diagnostics["total_energy_drift"] = total - initial_total_energy
    return diagnostics


def batched_velocity_verlet(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    timestep_fs: float | Sequence[float] | torch.Tensor,
    n_steps: int,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    com_removed_for_temperature: bool = False,
    wrap_interval: int | None = None,
) -> MDResult:
    """Run fixed-cell NVE velocity-Verlet for all graphs in one batch."""

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be positive")
    if wrap_interval is not None and wrap_interval <= 0:
        raise ValueError("wrap_interval must be positive when provided")

    dt_system = as_system_parameter(
        timestep_fs,
        n_systems=state.n_systems,
        device=state.device,
        dtype=state.dtype,
        name="timestep_fs",
    )
    if bool((dt_system <= 0.0).any()):
        raise ValueError("all time steps must be positive")
    dt_atom = dt_system[state.system_idx].unsqueeze(-1)

    state.zero_fixed_motion_()
    evaluation = potential(state, neighbor_policy="auto")
    initial_total = evaluation.energy.detach() + state.kinetic_energy().detach()

    if callback is not None:
        callback(
            0,
            state,
            evaluation,
            _md_diagnostics(
                state,
                evaluation.energy,
                initial_total_energy=initial_total,
                com_removed_for_temperature=com_removed_for_temperature,
            ),
        )

    for step in range(1, n_steps + 1):
        acceleration = _acceleration(state, evaluation.forces)
        state.velocities = state.velocities + 0.5 * dt_atom * acceleration
        state.zero_fixed_motion_()

        displacement = dt_atom * state.velocities
        displacement = displacement.masked_fill(state.fixed.unsqueeze(-1), 0.0)
        state.positions = (state.positions + displacement).detach()

        if wrap_interval is not None and step % wrap_interval == 0:
            state.wrap_()

        evaluation = potential(state, neighbor_policy="auto")
        new_acceleration = _acceleration(state, evaluation.forces)
        state.velocities = (
            state.velocities + 0.5 * dt_atom * new_acceleration
        ).detach()
        state.zero_fixed_motion_()

        if callback is not None and step % callback_interval == 0:
            callback(
                step,
                state,
                evaluation,
                _md_diagnostics(
                    state,
                    evaluation.energy,
                    initial_total_energy=initial_total,
                    com_removed_for_temperature=com_removed_for_temperature,
                ),
            )

    return MDResult(
        state=state,
        evaluation=evaluation,
        steps=n_steps,
        kinetic_energy=state.kinetic_energy(),
        temperature=state.temperature(com_removed=com_removed_for_temperature),
        initial_total_energy=initial_total,
    )


def batched_langevin_baoab(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    timestep_fs: float | Sequence[float] | torch.Tensor,
    n_steps: int,
    temperature_K: float | Sequence[float] | torch.Tensor,
    friction_per_fs: float | Sequence[float] | torch.Tensor = 0.01,
    seed: int | None = None,
    remove_com_each_step: bool = False,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    wrap_interval: int | None = None,
) -> MDResult:
    """Run fixed-cell NVT Langevin dynamics with BAOAB splitting."""

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be positive")
    if wrap_interval is not None and wrap_interval <= 0:
        raise ValueError("wrap_interval must be positive when provided")

    dt_system = as_system_parameter(
        timestep_fs,
        n_systems=state.n_systems,
        device=state.device,
        dtype=state.dtype,
        name="timestep_fs",
    )
    temperatures = as_system_parameter(
        temperature_K,
        n_systems=state.n_systems,
        device=state.device,
        dtype=state.dtype,
        name="temperature_K",
    )
    friction = as_system_parameter(
        friction_per_fs,
        n_systems=state.n_systems,
        device=state.device,
        dtype=state.dtype,
        name="friction_per_fs",
    )
    if bool((dt_system <= 0.0).any()):
        raise ValueError("all time steps must be positive")
    if bool((temperatures < 0.0).any()):
        raise ValueError("temperature must be non-negative")
    if bool((friction < 0.0).any()):
        raise ValueError("friction must be non-negative")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=state.device)
        generator.manual_seed(seed)

    dt_atom = dt_system[state.system_idx].unsqueeze(-1)
    c1_system = torch.exp(-friction * dt_system)
    c2_system = torch.sqrt((1.0 - c1_system * c1_system).clamp_min(0.0))
    c1_atom = c1_system[state.system_idx].unsqueeze(-1)
    c2_atom = c2_system[state.system_idx].unsqueeze(-1)
    thermal_sigma = torch.sqrt(
        KB_EV_PER_K
        * temperatures[state.system_idx]
        / (state.masses * AMU_A2_PER_FS2_TO_EV)
    )

    state.zero_fixed_motion_()
    evaluation = potential(state, neighbor_policy="auto")

    if callback is not None:
        callback(
            0,
            state,
            evaluation,
            _md_diagnostics(
                state,
                evaluation.energy,
                initial_total_energy=None,
                com_removed_for_temperature=remove_com_each_step,
            ),
        )

    for step in range(1, n_steps + 1):
        # B: half kick.
        acceleration = _acceleration(state, evaluation.forces)
        state.velocities = state.velocities + 0.5 * dt_atom * acceleration
        state.zero_fixed_motion_()

        # A: half drift.
        half_displacement = 0.5 * dt_atom * state.velocities
        half_displacement = half_displacement.masked_fill(
            state.fixed.unsqueeze(-1), 0.0
        )
        state.positions = (state.positions + half_displacement).detach()

        # O: exact Ornstein-Uhlenbeck update.
        noise = torch.randn(
            state.velocities.shape,
            device=state.device,
            dtype=state.dtype,
            generator=generator,
        )
        state.velocities = (
            c1_atom * state.velocities
            + c2_atom * thermal_sigma.unsqueeze(-1) * noise
        )
        state.zero_fixed_motion_()
        if remove_com_each_step:
            state.remove_center_of_mass_velocity_()

        # A: second half drift.
        half_displacement = 0.5 * dt_atom * state.velocities
        half_displacement = half_displacement.masked_fill(
            state.fixed.unsqueeze(-1), 0.0
        )
        state.positions = (state.positions + half_displacement).detach()

        if wrap_interval is not None and step % wrap_interval == 0:
            state.wrap_()

        # New force, then B: second half kick.
        evaluation = potential(state, neighbor_policy="auto")
        new_acceleration = _acceleration(state, evaluation.forces)
        state.velocities = (
            state.velocities + 0.5 * dt_atom * new_acceleration
        ).detach()
        state.zero_fixed_motion_()

        if callback is not None and step % callback_interval == 0:
            callback(
                step,
                state,
                evaluation,
                _md_diagnostics(
                    state,
                    evaluation.energy,
                    initial_total_energy=None,
                    com_removed_for_temperature=remove_com_each_step,
                ),
            )

    return MDResult(
        state=state,
        evaluation=evaluation,
        steps=n_steps,
        kinetic_energy=state.kinetic_energy(),
        temperature=state.temperature(com_removed=remove_com_each_step),
    )
