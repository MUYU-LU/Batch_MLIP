"""Batched isotropic Martyna-Tobias-Klein NPT dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ase import units

from ..core.calculator import BatchCalculator
from ..core.math_utils import as_system_parameter, scatter_sum
from ..core.state import KB_EV_PER_K, AseGraphBatch
from ..core.types import BatchEvaluation, MDResult, StepCallback
from .integrators import _md_diagnostics

_FOURTH_ORDER_COEFFS = (
    1.3512071919596578,
    -1.7024143839193153,
    1.3512071919596578,
)
_ASE_TIME_PER_FS = float(units.fs)


def _exprel(value: torch.Tensor) -> torch.Tensor:
    """Return ``(exp(x) - 1) / x`` without cancellation near zero."""

    small = value.abs() < 1e-7
    series = 1.0 + value * (
        0.5 + value * (1.0 / 6.0 + value * (1.0 / 24.0))
    )
    denominator = torch.where(small, torch.ones_like(value), value)
    ratio = torch.expm1(value) / denominator
    return torch.where(small, series, ratio)


@dataclass
class IsotropicMTKState:
    """Restartable extended state for independent isotropic MTK systems."""

    reference_cells: torch.Tensor
    reference_volumes: torch.Tensor
    atomic_momenta: torch.Tensor
    log_volume_scale: torch.Tensor
    volume_momenta: torch.Tensor
    thermostat_positions: torch.Tensor
    thermostat_momenta: torch.Tensor
    barostat_positions: torch.Tensor
    barostat_momenta: torch.Tensor
    temperature_K: torch.Tensor
    pressure_eV_per_A3: torch.Tensor
    thermostat_damping_fs: torch.Tensor
    barostat_damping_fs: torch.Tensor
    thermostat_substeps: int
    barostat_substeps: int
    steps: int = 0

    @property
    def thermostat_chain_length(self) -> int:
        return int(self.thermostat_positions.shape[1])

    @property
    def barostat_chain_length(self) -> int:
        return int(self.barostat_positions.shape[1])

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU payload suitable for ``torch.save``."""

        return {
            "schema_version": 1,
            "kind": "isotropic_mtk",
            "reference_cells": self.reference_cells.detach().cpu(),
            "reference_volumes": self.reference_volumes.detach().cpu(),
            "atomic_momenta": self.atomic_momenta.detach().cpu(),
            "log_volume_scale": self.log_volume_scale.detach().cpu(),
            "volume_momenta": self.volume_momenta.detach().cpu(),
            "thermostat_positions": self.thermostat_positions.detach().cpu(),
            "thermostat_momenta": self.thermostat_momenta.detach().cpu(),
            "barostat_positions": self.barostat_positions.detach().cpu(),
            "barostat_momenta": self.barostat_momenta.detach().cpu(),
            "temperature_K": self.temperature_K.detach().cpu(),
            "pressure_eV_per_A3": self.pressure_eV_per_A3.detach().cpu(),
            "thermostat_damping_fs": self.thermostat_damping_fs.detach().cpu(),
            "barostat_damping_fs": self.barostat_damping_fs.detach().cpu(),
            "thermostat_substeps": self.thermostat_substeps,
            "barostat_substeps": self.barostat_substeps,
            "steps": self.steps,
        }

    @classmethod
    def from_state_dict(
        cls,
        values: Mapping[str, Any],
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> IsotropicMTKState:
        """Restore a validated state payload on the requested tensor device."""

        if values.get("schema_version") != 1:
            raise ValueError("unsupported isotropic MTK state schema")
        if values.get("kind") != "isotropic_mtk":
            raise ValueError("state payload is not isotropic MTK")
        resolved = torch.device(device)

        def tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(values[name], device=resolved, dtype=dtype)

        return cls(
            reference_cells=tensor("reference_cells"),
            reference_volumes=tensor("reference_volumes"),
            atomic_momenta=tensor("atomic_momenta"),
            log_volume_scale=tensor("log_volume_scale"),
            volume_momenta=tensor("volume_momenta"),
            thermostat_positions=tensor("thermostat_positions"),
            thermostat_momenta=tensor("thermostat_momenta"),
            barostat_positions=tensor("barostat_positions"),
            barostat_momenta=tensor("barostat_momenta"),
            temperature_K=tensor("temperature_K"),
            pressure_eV_per_A3=tensor("pressure_eV_per_A3"),
            thermostat_damping_fs=tensor("thermostat_damping_fs"),
            barostat_damping_fs=tensor("barostat_damping_fs"),
            thermostat_substeps=int(values["thermostat_substeps"]),
            barostat_substeps=int(values["barostat_substeps"]),
            steps=int(values["steps"]),
        )


def _chain_masses(
    state: AseGraphBatch,
    extended: IsotropicMTKState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = state.counts.to(state.dtype)
    kT = KB_EV_PER_K * extended.temperature_K
    tdamp = extended.thermostat_damping_fs * _ASE_TIME_PER_FS
    pdamp = extended.barostat_damping_fs * _ASE_TIME_PER_FS

    thermostat = kT[:, None] * tdamp[:, None].square()
    thermostat = thermostat.expand(
        state.n_systems, extended.thermostat_chain_length
    ).clone()
    thermostat[:, 0] *= 3.0 * counts

    barostat = kT[:, None] * pdamp[:, None].square()
    barostat = barostat.expand(
        state.n_systems, extended.barostat_chain_length
    ).clone()
    barostat[:, 0] *= 9.0

    volume = (3.0 * counts + 3.0) * kT * pdamp.square()
    return thermostat, barostat, volume


def _atom_kinetic_sum(
    momenta: torch.Tensor,
    state: AseGraphBatch,
) -> torch.Tensor:
    per_atom = momenta.square().sum(dim=-1) / state.masses
    return scatter_sum(per_atom, state.system_idx, state.n_systems)


def _integrate_thermostat(
    momenta: torch.Tensor,
    state: AseGraphBatch,
    extended: IsotropicMTKState,
    thermostat_masses: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    chain_length = extended.thermostat_chain_length
    kT = KB_EV_PER_K * extended.temperature_K
    counts = state.counts.to(state.dtype)
    for _ in range(extended.thermostat_substeps):
        for coefficient in _FOURTH_ORDER_COEFFS:
            sub_delta = coefficient * delta / extended.thermostat_substeps
            delta2 = 0.5 * sub_delta
            delta4 = 0.25 * sub_delta
            for chain_index in range(chain_length - 1, -1, -1):
                if chain_index < chain_length - 1:
                    extended.thermostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.thermostat_momenta[:, chain_index + 1]
                        / thermostat_masses[:, chain_index + 1]
                    )
                if chain_index == 0:
                    driving = (
                        _atom_kinetic_sum(momenta, state)
                        - 3.0 * counts * kT
                    )
                else:
                    driving = (
                        extended.thermostat_momenta[
                            :, chain_index - 1
                        ].square()
                        / thermostat_masses[:, chain_index - 1]
                        - kT
                    )
                extended.thermostat_momenta[:, chain_index] += (
                    delta2 * driving
                )
                if chain_index < chain_length - 1:
                    extended.thermostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.thermostat_momenta[:, chain_index + 1]
                        / thermostat_masses[:, chain_index + 1]
                    )
            extended.thermostat_positions += (
                sub_delta[:, None]
                * extended.thermostat_momenta
                / thermostat_masses
            )
            scale = torch.exp(
                -sub_delta
                * extended.thermostat_momenta[:, 0]
                / thermostat_masses[:, 0]
            )
            momenta = momenta * scale[state.system_idx, None]
            for chain_index in range(chain_length):
                if chain_index < chain_length - 1:
                    extended.thermostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.thermostat_momenta[:, chain_index + 1]
                        / thermostat_masses[:, chain_index + 1]
                    )
                if chain_index == 0:
                    driving = (
                        _atom_kinetic_sum(momenta, state)
                        - 3.0 * counts * kT
                    )
                else:
                    driving = (
                        extended.thermostat_momenta[
                            :, chain_index - 1
                        ].square()
                        / thermostat_masses[:, chain_index - 1]
                        - kT
                    )
                extended.thermostat_momenta[:, chain_index] += (
                    delta2 * driving
                )
                if chain_index < chain_length - 1:
                    extended.thermostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.thermostat_momenta[:, chain_index + 1]
                        / thermostat_masses[:, chain_index + 1]
                    )
    return momenta


def _integrate_barostat_thermostat(
    extended: IsotropicMTKState,
    barostat_masses: torch.Tensor,
    volume_mass: torch.Tensor,
    delta: torch.Tensor,
) -> None:
    chain_length = extended.barostat_chain_length
    kT = KB_EV_PER_K * extended.temperature_K
    for _ in range(extended.barostat_substeps):
        for coefficient in _FOURTH_ORDER_COEFFS:
            sub_delta = coefficient * delta / extended.barostat_substeps
            delta2 = 0.5 * sub_delta
            delta4 = 0.25 * sub_delta
            for chain_index in range(chain_length - 1, -1, -1):
                if chain_index < chain_length - 1:
                    extended.barostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.barostat_momenta[:, chain_index + 1]
                        / barostat_masses[:, chain_index + 1]
                    )
                if chain_index == 0:
                    driving = (
                        extended.volume_momenta.square() / volume_mass - kT
                    )
                else:
                    driving = (
                        extended.barostat_momenta[
                            :, chain_index - 1
                        ].square()
                        / barostat_masses[:, chain_index - 1]
                        - kT
                    )
                extended.barostat_momenta[:, chain_index] += delta2 * driving
                if chain_index < chain_length - 1:
                    extended.barostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.barostat_momenta[:, chain_index + 1]
                        / barostat_masses[:, chain_index + 1]
                    )
            extended.barostat_positions += (
                sub_delta[:, None]
                * extended.barostat_momenta
                / barostat_masses
            )
            extended.volume_momenta *= torch.exp(
                -sub_delta
                * extended.barostat_momenta[:, 0]
                / barostat_masses[:, 0]
            )
            for chain_index in range(chain_length):
                if chain_index < chain_length - 1:
                    extended.barostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.barostat_momenta[:, chain_index + 1]
                        / barostat_masses[:, chain_index + 1]
                    )
                if chain_index == 0:
                    driving = (
                        extended.volume_momenta.square() / volume_mass - kT
                    )
                else:
                    driving = (
                        extended.barostat_momenta[
                            :, chain_index - 1
                        ].square()
                        / barostat_masses[:, chain_index - 1]
                        - kT
                    )
                extended.barostat_momenta[:, chain_index] += delta2 * driving
                if chain_index < chain_length - 1:
                    extended.barostat_momenta[:, chain_index] *= torch.exp(
                        -delta4
                        * extended.barostat_momenta[:, chain_index + 1]
                        / barostat_masses[:, chain_index + 1]
                    )


def _pressure(
    evaluation: BatchEvaluation,
    momenta: torch.Tensor,
    state: AseGraphBatch,
    volumes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if evaluation.stress is None:
        raise RuntimeError("isotropic MTK requires calculator stress")
    kinetic_sum = _atom_kinetic_sum(momenta, state)
    pressure = (
        -torch.diagonal(evaluation.stress, dim1=-2, dim2=-1).sum(dim=-1)
        / 3.0
        + kinetic_sum / (3.0 * volumes)
    )
    return pressure, kinetic_sum


def _conserved_energy(
    evaluation: BatchEvaluation,
    momenta: torch.Tensor,
    state: AseGraphBatch,
    extended: IsotropicMTKState,
    thermostat_masses: torch.Tensor,
    barostat_masses: torch.Tensor,
    volume_mass: torch.Tensor,
) -> torch.Tensor:
    counts = state.counts.to(state.dtype)
    kT = KB_EV_PER_K * extended.temperature_K
    kinetic = 0.5 * _atom_kinetic_sum(momenta, state)
    thermostat = (
        3.0 * counts * kT * extended.thermostat_positions[:, 0]
        + kT * extended.thermostat_positions[:, 1:].sum(dim=-1)
        + 0.5
        * (
            extended.thermostat_momenta.square() / thermostat_masses
        ).sum(dim=-1)
    )
    barostat = (
        0.5
        * (extended.barostat_momenta.square() / barostat_masses).sum(dim=-1)
        + kT * extended.barostat_positions.sum(dim=-1)
    )
    volume = extended.reference_volumes * torch.exp(
        3.0 * extended.log_volume_scale
    )
    return (
        evaluation.energy
        + kinetic
        + thermostat
        + barostat
        + 0.5 * extended.volume_momenta.square() / volume_mass
        + extended.pressure_eV_per_A3 * volume
    )


def _validate_extended_state(
    extended: IsotropicMTKState,
    state: AseGraphBatch,
    *,
    temperature_K: torch.Tensor,
    pressure_eV_per_A3: torch.Tensor,
    thermostat_damping_fs: torch.Tensor,
    barostat_damping_fs: torch.Tensor,
    thermostat_chain_length: int,
    barostat_chain_length: int,
    thermostat_substeps: int,
    barostat_substeps: int,
) -> None:
    batch = state.n_systems
    shapes = {
        "reference_cells": (batch, 3, 3),
        "reference_volumes": (batch,),
        "atomic_momenta": (state.n_atoms, 3),
        "log_volume_scale": (batch,),
        "volume_momenta": (batch,),
        "thermostat_positions": (batch, thermostat_chain_length),
        "thermostat_momenta": (batch, thermostat_chain_length),
        "barostat_positions": (batch, barostat_chain_length),
        "barostat_momenta": (batch, barostat_chain_length),
    }
    for name, expected in shapes.items():
        value = getattr(extended, name)
        if value.shape != expected:
            raise ValueError(f"integrator_state.{name} must have shape {expected}")
        if value.device != state.device or value.dtype != state.dtype:
            raise ValueError(
                f"integrator_state.{name} must match state device and dtype"
            )
    parameters = (
        ("temperature_K", temperature_K),
        ("pressure_eV_per_A3", pressure_eV_per_A3),
        ("thermostat_damping_fs", thermostat_damping_fs),
        ("barostat_damping_fs", barostat_damping_fs),
    )
    for name, expected in parameters:
        if not torch.equal(getattr(extended, name), expected):
            raise ValueError(f"integrator_state {name} does not match this run")
    if extended.thermostat_substeps != thermostat_substeps:
        raise ValueError("integrator_state thermostat_substeps does not match")
    if extended.barostat_substeps != barostat_substeps:
        raise ValueError("integrator_state barostat_substeps does not match")


def batched_isotropic_mtk(
    state: AseGraphBatch,
    potential: BatchCalculator,
    *,
    timestep_fs: float | Sequence[float] | torch.Tensor,
    n_steps: int,
    temperature_K: float | Sequence[float] | torch.Tensor,
    pressure_eV_per_A3: float | Sequence[float] | torch.Tensor = 0.0,
    thermostat_damping_fs: float | Sequence[float] | torch.Tensor = 50.0,
    barostat_damping_fs: float | Sequence[float] | torch.Tensor = 500.0,
    thermostat_chain_length: int = 3,
    barostat_chain_length: int = 3,
    thermostat_substeps: int = 1,
    barostat_substeps: int = 1,
    integrator_state: IsotropicMTKState | None = None,
    callback: StepCallback | None = None,
    callback_interval: int = 1,
    wrap_interval: int | None = None,
) -> MDResult:
    """Run independent isotropic MTK NPT replicas in one tensor batch."""

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if callback_interval <= 0:
        raise ValueError("callback_interval must be positive")
    if wrap_interval is not None and wrap_interval <= 0:
        raise ValueError("wrap_interval must be positive when provided")
    for name, value in (
        ("thermostat_chain_length", thermostat_chain_length),
        ("barostat_chain_length", barostat_chain_length),
        ("thermostat_substeps", thermostat_substeps),
        ("barostat_substeps", barostat_substeps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if bool(state.fixed.any()):
        raise NotImplementedError("isotropic MTK does not support FixAtoms")
    if not bool(state.pbc.all()):
        raise NotImplementedError(
            "isotropic MTK requires three-dimensional periodic systems"
        )

    def parameter(value: Any, name: str) -> torch.Tensor:
        return as_system_parameter(
            value,
            n_systems=state.n_systems,
            device=state.device,
            dtype=state.dtype,
            name=name,
        )

    dt_fs = parameter(timestep_fs, "timestep_fs")
    temperature = parameter(temperature_K, "temperature_K")
    pressure_target = parameter(
        pressure_eV_per_A3, "pressure_eV_per_A3"
    )
    thermostat_damping = parameter(
        thermostat_damping_fs, "thermostat_damping_fs"
    )
    barostat_damping = parameter(
        barostat_damping_fs, "barostat_damping_fs"
    )
    if bool((dt_fs <= 0.0).any()):
        raise ValueError("all time steps must be positive")
    if bool((temperature <= 0.0).any()):
        raise ValueError("isotropic MTK temperature must be positive")
    if bool((thermostat_damping <= 0.0).any()):
        raise ValueError("thermostat damping must be positive")
    if bool((barostat_damping <= 0.0).any()):
        raise ValueError("barostat damping must be positive")
    volumes = torch.linalg.det(state.cells)
    if bool((~torch.isfinite(volumes)).any() or (volumes <= 0.0).any()):
        raise ValueError("isotropic MTK requires positive finite cell volumes")

    if integrator_state is None:
        initial_momenta = (
            state.velocities
            / _ASE_TIME_PER_FS
            * state.masses.unsqueeze(-1)
        )
        extended = IsotropicMTKState(
            reference_cells=state.cells.detach().clone(),
            reference_volumes=volumes.detach().clone(),
            atomic_momenta=initial_momenta.detach().clone(),
            log_volume_scale=torch.zeros_like(volumes),
            volume_momenta=torch.zeros_like(volumes),
            thermostat_positions=torch.zeros(
                (state.n_systems, thermostat_chain_length),
                device=state.device,
                dtype=state.dtype,
            ),
            thermostat_momenta=torch.zeros(
                (state.n_systems, thermostat_chain_length),
                device=state.device,
                dtype=state.dtype,
            ),
            barostat_positions=torch.zeros(
                (state.n_systems, barostat_chain_length),
                device=state.device,
                dtype=state.dtype,
            ),
            barostat_momenta=torch.zeros(
                (state.n_systems, barostat_chain_length),
                device=state.device,
                dtype=state.dtype,
            ),
            temperature_K=temperature.detach().clone(),
            pressure_eV_per_A3=pressure_target.detach().clone(),
            thermostat_damping_fs=thermostat_damping.detach().clone(),
            barostat_damping_fs=barostat_damping.detach().clone(),
            thermostat_substeps=thermostat_substeps,
            barostat_substeps=barostat_substeps,
        )
    else:
        extended = integrator_state
        _validate_extended_state(
            extended,
            state,
            temperature_K=temperature,
            pressure_eV_per_A3=pressure_target,
            thermostat_damping_fs=thermostat_damping,
            barostat_damping_fs=barostat_damping,
            thermostat_chain_length=thermostat_chain_length,
            barostat_chain_length=barostat_chain_length,
            thermostat_substeps=thermostat_substeps,
            barostat_substeps=barostat_substeps,
        )
        expected_cells = extended.reference_cells * torch.exp(
            extended.log_volume_scale
        )[:, None, None]
        if not torch.allclose(state.cells, expected_cells, rtol=1e-6, atol=1e-8):
            raise ValueError("physical cells do not match integrator_state")

    thermostat_masses, barostat_masses, volume_mass = _chain_masses(
        state, extended
    )
    momenta = extended.atomic_momenta
    dt = dt_fs * _ASE_TIME_PER_FS
    half_dt = 0.5 * dt
    evaluation = potential(
        state, neighbor_policy="auto", compute_stress=True
    )
    initial_conserved = _conserved_energy(
        evaluation,
        momenta,
        state,
        extended,
        thermostat_masses,
        barostat_masses,
        volume_mass,
    ).detach()

    def diagnostics() -> dict[str, torch.Tensor]:
        current_volumes = extended.reference_volumes * torch.exp(
            3.0 * extended.log_volume_scale
        )
        current_pressure, _ = _pressure(
            evaluation, momenta, state, current_volumes
        )
        values = _md_diagnostics(
            state,
            evaluation.energy,
            initial_total_energy=None,
            com_removed_for_temperature=False,
        )
        conserved = _conserved_energy(
            evaluation,
            momenta,
            state,
            extended,
            thermostat_masses,
            barostat_masses,
            volume_mass,
        ).detach()
        values.update(
            {
                "pressure": current_pressure.detach(),
                "volume": current_volumes.detach(),
                "conserved_energy": conserved,
                "conserved_energy_drift": conserved - initial_conserved,
            }
        )
        return values

    if callback is not None:
        callback(0, state, evaluation, diagnostics())

    for step in range(1, n_steps + 1):
        _integrate_barostat_thermostat(
            extended, barostat_masses, volume_mass, half_dt
        )
        momenta = _integrate_thermostat(
            momenta, state, extended, thermostat_masses, half_dt
        )

        current_volumes = extended.reference_volumes * torch.exp(
            3.0 * extended.log_volume_scale
        )
        current_pressure, kinetic_sum = _pressure(
            evaluation, momenta, state, current_volumes
        )
        cell_force = (
            3.0
            * current_volumes
            * (current_pressure - extended.pressure_eV_per_A3)
            + kinetic_sum / state.counts.to(state.dtype)
        )
        extended.volume_momenta += half_dt * cell_force

        momentum_x = (
            (1.0 + 1.0 / state.counts.to(state.dtype))
            * extended.volume_momenta
            * half_dt
            / volume_mass
        )
        atom_momentum_x = momentum_x[state.system_idx, None]
        momenta = (
            momenta * torch.exp(-atom_momentum_x)
            + half_dt[state.system_idx, None]
            * evaluation.forces
            * _exprel(-atom_momentum_x)
        )

        position_x = dt * extended.volume_momenta / volume_mass
        atom_position_x = position_x[state.system_idx, None]
        state.positions = (
            state.positions * torch.exp(atom_position_x)
            + dt[state.system_idx, None]
            * momenta
            / state.masses[:, None]
            * _exprel(atom_position_x)
        ).detach()
        extended.log_volume_scale += position_x
        state.cells = (
            extended.reference_cells
            * torch.exp(extended.log_volume_scale)[:, None, None]
        ).detach()
        if wrap_interval is not None and step % wrap_interval == 0:
            state.wrap_()

        evaluation = potential(
            state, neighbor_policy="auto", compute_stress=True
        )
        momentum_x = (
            (1.0 + 1.0 / state.counts.to(state.dtype))
            * extended.volume_momenta
            * half_dt
            / volume_mass
        )
        atom_momentum_x = momentum_x[state.system_idx, None]
        momenta = (
            momenta * torch.exp(-atom_momentum_x)
            + half_dt[state.system_idx, None]
            * evaluation.forces
            * _exprel(-atom_momentum_x)
        )

        current_volumes = extended.reference_volumes * torch.exp(
            3.0 * extended.log_volume_scale
        )
        current_pressure, kinetic_sum = _pressure(
            evaluation, momenta, state, current_volumes
        )
        cell_force = (
            3.0
            * current_volumes
            * (current_pressure - extended.pressure_eV_per_A3)
            + kinetic_sum / state.counts.to(state.dtype)
        )
        extended.volume_momenta += half_dt * cell_force

        momenta = _integrate_thermostat(
            momenta, state, extended, thermostat_masses, half_dt
        )
        _integrate_barostat_thermostat(
            extended, barostat_masses, volume_mass, half_dt
        )
        state.velocities = (
            momenta
            / state.masses.unsqueeze(-1)
            * _ASE_TIME_PER_FS
        ).detach()
        extended.atomic_momenta = momenta.detach().clone()
        extended.steps += 1

        if callback is not None and step % callback_interval == 0:
            callback(step, state, evaluation, diagnostics())

    final_conserved = _conserved_energy(
        evaluation,
        momenta,
        state,
        extended,
        thermostat_masses,
        barostat_masses,
        volume_mass,
    ).detach()
    final_volumes = extended.reference_volumes * torch.exp(
        3.0 * extended.log_volume_scale
    )
    final_pressure, _ = _pressure(
        evaluation, momenta, state, final_volumes
    )
    return MDResult(
        state=state,
        evaluation=evaluation,
        steps=n_steps,
        kinetic_energy=state.kinetic_energy(),
        temperature=state.temperature(),
        integrator_state=extended,
        model_evaluations=n_steps + 1,
        graph_evaluations=n_steps + 1,
        metadata={
            "ensemble": "npt_mtk_isotropic",
            "initial_conserved_energy": initial_conserved,
            "final_conserved_energy": final_conserved,
            "conserved_energy_drift": final_conserved - initial_conserved,
            "final_pressure_eV_per_A3": final_pressure.detach(),
            "final_volume_A3": final_volumes.detach(),
        },
    )
