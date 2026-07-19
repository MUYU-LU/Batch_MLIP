"""Core flattened ASE batches and neighbor-list lifecycle management."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from ..profiling.runtime import profile_event, profile_phase
from .math_utils import scatter_sum
from .neighbors import neighbor_list
from .types import BatchEvaluation, GraphData

# Positions: Angstrom; time: femtosecond; energy: eV; mass: atomic mass unit.
KB_EV_PER_K = 8.617333262145e-5
AMU_A2_PER_FS2_TO_EV = 103.64269652680505
EV_PER_A_PER_AMU_TO_A_PER_FS2 = 1.0 / AMU_A2_PER_FS2_TO_EV


def _extract_fixatoms_mask(atoms: Atoms) -> np.ndarray:
    """Return fixed-atom mask and warn about unsupported constraints."""

    fixed = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed[np.asarray(constraint.get_indices(), dtype=np.int64)] = True
        else:
            warnings.warn(
                f"Ignoring unsupported ASE constraint {type(constraint).__name__}; "
                "only FixAtoms is currently implemented.",
                RuntimeWarning,
                stacklevel=2,
            )
    return fixed


@dataclass
class AseGraphBatch:
    """Flattened representation of independent ASE systems.

    Atomic arrays are concatenated along the atom dimension. ``system_idx``
    maps each atom to its graph, and ``ptr`` stores graph boundaries. Edges are
    built independently per graph, then offset and concatenated.
    """

    templates: list[Atoms]
    cutoff: float
    skin: float
    device: torch.device
    dtype: torch.dtype

    z: torch.Tensor
    positions: torch.Tensor
    cells: torch.Tensor
    pbc: torch.Tensor
    system_idx: torch.Tensor
    ptr: torch.Tensor
    masses: torch.Tensor
    fixed: torch.Tensor
    velocities: torch.Tensor

    edge_index: torch.Tensor
    shifts_int: torch.Tensor
    neighbor_rebuild_count: int = 0
    _neighbor_reference_positions: torch.Tensor | None = field(default=None, repr=False)
    _neighbor_reference_cells: torch.Tensor | None = field(default=None, repr=False)

    @classmethod
    def from_ase(
        cls,
        systems: Sequence[Atoms],
        *,
        cutoff: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        skin: float = 0.0,
        build_neighbors: bool = True,
    ) -> AseGraphBatch:
        if not systems:
            raise ValueError("systems must contain at least one ASE Atoms object")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if skin < 0.0:
            raise ValueError("skin must be non-negative")

        device_obj = torch.device(device)
        templates = [atoms.copy() for atoms in systems]
        counts = np.asarray([len(atoms) for atoms in templates], dtype=np.int64)
        if np.any(counts <= 0):
            raise ValueError("empty ASE structures are not supported")

        ptr_np = np.concatenate(([0], np.cumsum(counts)))
        z_np = np.concatenate([np.asarray(atoms.numbers, dtype=np.int64) for atoms in templates])
        pos_np = np.concatenate(
            [np.asarray(atoms.positions, dtype=np.float64) for atoms in templates], axis=0
        )
        cell_np = np.stack(
            [np.asarray(atoms.cell.array, dtype=np.float64) for atoms in templates], axis=0
        )
        pbc_np = np.stack([np.asarray(atoms.pbc, dtype=bool) for atoms in templates], axis=0)
        masses_np = np.concatenate(
            [np.asarray(atoms.get_masses(), dtype=np.float64) for atoms in templates]
        )
        if np.any(masses_np <= 0.0):
            raise ValueError("all atomic masses must be positive")

        fixed_np = np.concatenate([_extract_fixatoms_mask(atoms) for atoms in templates])
        system_idx_np = np.repeat(np.arange(len(templates), dtype=np.int64), counts)

        velocity_arrays: list[np.ndarray] = []
        for atoms in templates:
            velocity = atoms.get_velocities()
            if velocity is None:
                velocity = np.zeros((len(atoms), 3), dtype=np.float64)
            velocity_arrays.append(np.asarray(velocity, dtype=np.float64))
        velocities_np = np.concatenate(velocity_arrays, axis=0)

        batch = cls(
            templates=templates,
            cutoff=float(cutoff),
            skin=float(skin),
            device=device_obj,
            dtype=dtype,
            z=torch.as_tensor(z_np, device=device_obj, dtype=torch.long),
            positions=torch.as_tensor(pos_np, device=device_obj, dtype=dtype),
            cells=torch.as_tensor(cell_np, device=device_obj, dtype=dtype),
            pbc=torch.as_tensor(pbc_np, device=device_obj, dtype=torch.bool),
            system_idx=torch.as_tensor(system_idx_np, device=device_obj, dtype=torch.long),
            ptr=torch.as_tensor(ptr_np, device=device_obj, dtype=torch.long),
            masses=torch.as_tensor(masses_np, device=device_obj, dtype=dtype),
            fixed=torch.as_tensor(fixed_np, device=device_obj, dtype=torch.bool),
            velocities=torch.as_tensor(velocities_np, device=device_obj, dtype=dtype),
            edge_index=torch.empty((2, 0), device=device_obj, dtype=torch.long),
            shifts_int=torch.empty((0, 3), device=device_obj, dtype=torch.long),
        )
        if build_neighbors:
            batch.rebuild_neighbor_list()
        return batch

    @property
    def n_systems(self) -> int:
        return len(self.templates)

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    @property
    def mobile(self) -> torch.Tensor:
        return ~self.fixed

    @property
    def counts(self) -> torch.Tensor:
        return self.ptr[1:] - self.ptr[:-1]

    def atom_slice(self, system: int) -> slice:
        if not 0 <= system < self.n_systems:
            raise IndexError(f"system index {system} outside [0, {self.n_systems})")
        start = int(self.ptr[system].item())
        stop = int(self.ptr[system + 1].item())
        return slice(start, stop)

    def neighbor_list_needs_rebuild(self) -> bool:
        if self._neighbor_reference_positions is None or self._neighbor_reference_cells is None:
            return True
        if self.skin <= 0.0:
            return True
        if not torch.equal(self.cells, self._neighbor_reference_cells):
            return True
        displacement = torch.linalg.vector_norm(
            self.positions - self._neighbor_reference_positions, dim=-1
        )
        return bool((displacement.max() > 0.5 * self.skin).item())

    def ensure_neighbor_list(self) -> bool:
        """Rebuild only when required; return whether a rebuild occurred."""

        if self.neighbor_list_needs_rebuild():
            self.rebuild_neighbor_list()
            return True
        return False

    def rebuild_neighbor_list(self) -> None:
        """Build directed neighbour lists independently, then concatenate."""

        with profile_phase(
            "graph.geometry_to_host",
            device=self.device,
            systems=self.n_systems,
            atoms=self.n_atoms,
        ):
            pos_cpu = self.positions.detach().cpu().numpy()
            cell_cpu = self.cells.detach().cpu().numpy()
            pbc_cpu = self.pbc.detach().cpu().numpy()

        edge_blocks: list[np.ndarray] = []
        shift_blocks: list[np.ndarray] = []

        with profile_phase(
            "graph.neighbor_search",
            device=self.device,
            systems=self.n_systems,
            atoms=self.n_atoms,
        ):
            for graph_idx, template in enumerate(self.templates):
                atom_slice = self.atom_slice(graph_idx)
                atoms = template.copy()
                atoms.positions[:] = pos_cpu[atom_slice]
                atoms.set_cell(cell_cpu[graph_idx], scale_atoms=False)
                atoms.pbc = pbc_cpu[graph_idx]

                i_idx, j_idx, shifts = neighbor_list(
                    "ijS", atoms, self.cutoff + self.skin
                )
                edge_block = np.stack((i_idx, j_idx), axis=0).astype(
                    np.int64, copy=False
                )
                edge_block += atom_slice.start
                edge_blocks.append(edge_block)
                shift_blocks.append(np.asarray(shifts, dtype=np.int64))

        edge_np = (
            np.concatenate(edge_blocks, axis=1)
            if edge_blocks
            else np.empty((2, 0), dtype=np.int64)
        )
        shifts_np = (
            np.concatenate(shift_blocks, axis=0)
            if shift_blocks
            else np.empty((0, 3), dtype=np.int64)
        )

        with profile_phase(
            "graph.to_device",
            device=self.device,
            systems=self.n_systems,
            atoms=self.n_atoms,
            edges=edge_np.shape[1],
        ):
            self.edge_index = torch.as_tensor(
                edge_np, device=self.device, dtype=torch.long
            )
            self.shifts_int = torch.as_tensor(
                shifts_np, device=self.device, dtype=torch.long
            )
            self._neighbor_reference_positions = self.positions.detach().clone()
            self._neighbor_reference_cells = self.cells.detach().clone()
            self.neighbor_rebuild_count += 1
            self.assert_graph_integrity()
        profile_event(
            "neighbor_rebuild",
            systems=self.n_systems,
            atoms=self.n_atoms,
            edges=edge_np.shape[1],
            rebuild_count=self.neighbor_rebuild_count,
        )

    def assert_graph_integrity(self) -> None:
        """Raise if an edge crosses systems or tensor shapes are inconsistent."""

        if self.positions.shape != (self.n_atoms, 3):
            raise RuntimeError("positions must have shape [N, 3]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise RuntimeError("edge_index must have shape [2, E]")
        if self.shifts_int.shape != (self.edge_index.shape[1], 3):
            raise RuntimeError("shifts_int must have shape [E, 3]")
        if self.edge_index.numel() > 0:
            center, neighbor = self.edge_index
            if bool((center < 0).any() or (neighbor < 0).any()):
                raise RuntimeError("edge_index contains negative indices")
            if bool((center >= self.n_atoms).any() or (neighbor >= self.n_atoms).any()):
                raise RuntimeError("edge_index contains out-of-range indices")
            if not torch.equal(self.system_idx[center], self.system_idx[neighbor]):
                raise RuntimeError("neighbor list contains cross-system edges")

    def as_model_data(
        self,
        *,
        positions: torch.Tensor | None = None,
        cells: torch.Tensor | None = None,
    ) -> GraphData:
        return GraphData(
            z=self.z,
            pos=self.positions if positions is None else positions,
            cell=self.cells if cells is None else cells,
            edge_index=self.edge_index,
            shifts_int=self.shifts_int,
            batch=self.system_idx,
            num_graphs=self.n_systems,
        )

    def zero_fixed_motion_(self) -> None:
        self.velocities[self.fixed] = 0.0

    def remove_center_of_mass_velocity_(self) -> None:
        mobile_mass = torch.where(self.mobile, self.masses, torch.zeros_like(self.masses))
        momentum = self.velocities * mobile_mass.unsqueeze(-1)
        total_momentum = scatter_sum(momentum, self.system_idx, self.n_systems)
        total_mass = scatter_sum(mobile_mass, self.system_idx, self.n_systems).clamp_min(1e-30)
        com_velocity = total_momentum / total_mass.unsqueeze(-1)
        self.velocities[self.mobile] -= com_velocity[self.system_idx[self.mobile]]
        self.zero_fixed_motion_()

    def kinetic_energy(self) -> torch.Tensor:
        atom_ke = (
            0.5
            * self.masses
            * (self.velocities * self.velocities).sum(dim=-1)
            * AMU_A2_PER_FS2_TO_EV
        )
        atom_ke = torch.where(self.mobile, atom_ke, torch.zeros_like(atom_ke))
        return scatter_sum(atom_ke, self.system_idx, self.n_systems)

    def degrees_of_freedom(self, *, com_removed: bool = False) -> torch.Tensor:
        mobile_count = scatter_sum(self.mobile.to(self.dtype), self.system_idx, self.n_systems)
        dof = 3.0 * mobile_count
        if com_removed:
            dof = torch.where(mobile_count > 1.0, dof - 3.0, dof)
        return dof.clamp_min(1.0)

    def temperature(self, *, com_removed: bool = False) -> torch.Tensor:
        return 2.0 * self.kinetic_energy() / (
            self.degrees_of_freedom(com_removed=com_removed) * KB_EV_PER_K
        )

    def wrap_(self) -> None:
        """Wrap periodic atoms through ASE and keep the tensor batch in sync."""

        frames = self.to_ase(evaluation=None, wrap=True)
        self.positions = torch.as_tensor(
            np.concatenate([atoms.positions for atoms in frames], axis=0),
            device=self.device,
            dtype=self.dtype,
        )
        self._neighbor_reference_positions = None

    def to_ase(
        self,
        evaluation: BatchEvaluation | None = None,
        *,
        wrap: bool = False,
    ) -> list[Atoms]:
        pos_cpu = self.positions.detach().cpu().numpy()
        cell_cpu = self.cells.detach().cpu().numpy()
        vel_cpu = self.velocities.detach().cpu().numpy()

        energy_cpu = None if evaluation is None else evaluation.energy.detach().cpu().numpy()
        force_cpu = None if evaluation is None else evaluation.forces.detach().cpu().numpy()
        stress_cpu = (
            None
            if evaluation is None or evaluation.stress is None
            else evaluation.stress.detach().cpu().numpy()
        )

        out: list[Atoms] = []
        for graph_idx, template in enumerate(self.templates):
            atom_slice = self.atom_slice(graph_idx)
            atoms = template.copy()
            atoms.positions[:] = pos_cpu[atom_slice]
            atoms.set_cell(cell_cpu[graph_idx], scale_atoms=False)
            atoms.set_velocities(vel_cpu[atom_slice])
            if wrap:
                atoms.wrap()

            if evaluation is not None:
                kwargs = {
                    "energy": float(energy_cpu[graph_idx]),
                    "forces": np.asarray(force_cpu[atom_slice]),
                }
                if stress_cpu is not None and np.isfinite(stress_cpu[graph_idx]).all():
                    kwargs["stress"] = np.asarray(stress_cpu[graph_idx])
                atoms.calc = SinglePointCalculator(atoms, **kwargs)
            out.append(atoms)
        return out

    def select_systems(
        self,
        system_ids: Sequence[int],
        *,
        rebuild_neighbors: bool = True,
    ) -> AseGraphBatch:
        """Create a compact batch containing selected systems in the given order.

        Setting ``rebuild_neighbors=False`` avoids a redundant build when the
        caller will move atoms and evaluate with ``neighbor_policy='auto'``.
        """

        ids = [int(i) for i in system_ids]
        if not ids:
            raise ValueError("system_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("system_ids must be unique")
        if any(i < 0 or i >= self.n_systems for i in ids):
            raise IndexError("system id outside the batch")

        atom_blocks = [
            torch.arange(
                self.ptr[i], self.ptr[i + 1], device=self.device, dtype=torch.long
            )
            for i in ids
        ]
        atom_ids = torch.cat(atom_blocks)
        graph_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        counts = self.counts[graph_ids]
        ptr = torch.cat(
            (
                torch.zeros(1, device=self.device, dtype=torch.long),
                counts.cumsum(dim=0),
            )
        )
        system_idx = torch.repeat_interleave(
            torch.arange(len(ids), device=self.device, dtype=torch.long), counts
        )
        selected = AseGraphBatch(
            templates=[self.templates[i].copy() for i in ids],
            cutoff=self.cutoff,
            skin=self.skin,
            device=self.device,
            dtype=self.dtype,
            z=self.z[atom_ids].clone(),
            positions=self.positions[atom_ids].clone(),
            cells=self.cells[graph_ids].clone(),
            pbc=self.pbc[graph_ids].clone(),
            system_idx=system_idx,
            ptr=ptr,
            masses=self.masses[atom_ids].clone(),
            fixed=self.fixed[atom_ids].clone(),
            velocities=self.velocities[atom_ids].clone(),
            edge_index=torch.empty((2, 0), device=self.device, dtype=torch.long),
            shifts_int=torch.empty((0, 3), device=self.device, dtype=torch.long),
        )
        if rebuild_neighbors:
            selected.rebuild_neighbor_list()
        return selected
