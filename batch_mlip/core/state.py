"""Core flattened ASE batches and neighbor-list lifecycle management."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from ase import Atoms, units
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
    _neighbor_reference_valid: torch.Tensor | None = field(default=None, repr=False)

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
            # ASE velocities are Angstrom per ASE time unit; tensor-state MD
            # uses Angstrom/fs explicitly.
            velocity_arrays.append(np.asarray(velocity, dtype=np.float64) * units.fs)
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

    def neighbor_list_invalid_systems(self) -> torch.Tensor:
        """Return structures whose cached candidate topology is no longer safe.

        For a fully periodic changing cell, reference fractional coordinates
        separate affine cell motion from non-affine atomic motion. The inverse
        deformation norm then bounds every possible periodic pair, including a
        pair that was outside the cached candidate list at the reference state.
        """

        invalid = torch.ones(self.n_systems, device=self.device, dtype=torch.bool)
        if (
            self._neighbor_reference_positions is None
            or self._neighbor_reference_cells is None
            or self.skin <= 0.0
        ):
            return invalid

        valid = self._neighbor_reference_valid
        if valid is None:
            valid = torch.ones(
                self.n_systems, device=self.device, dtype=torch.bool
            )
        else:
            invalid = ~valid.clone()

        for system_id in range(self.n_systems):
            if not bool(valid[system_id]):
                continue
            atom_slice = self.atom_slice(system_id)
            positions = self.positions[atom_slice]
            reference_positions = self._neighbor_reference_positions[atom_slice]
            cell = self.cells[system_id]
            reference_cell = self._neighbor_reference_cells[system_id]
            periodic = self.pbc[system_id]

            if not bool(periodic.any()):
                displacement = torch.linalg.vector_norm(
                    positions - reference_positions, dim=-1
                ).max()
                invalid[system_id] = displacement > 0.5 * self.skin
                continue

            if not bool(periodic.all()):
                if not torch.equal(cell, reference_cell):
                    continue
                displacement = torch.linalg.vector_norm(
                    positions - reference_positions, dim=-1
                ).max()
                invalid[system_id] = displacement > 0.5 * self.skin
                continue

            try:
                reference_fractional = torch.linalg.solve(
                    reference_cell.transpose(0, 1),
                    reference_positions.transpose(0, 1),
                ).transpose(0, 1)
                affine_positions = reference_fractional @ cell
                non_affine = torch.linalg.vector_norm(
                    positions - affine_positions, dim=-1
                ).max()
                inverse_deformation = torch.linalg.solve(cell, reference_cell)
                inverse_stretch = torch.linalg.svdvals(inverse_deformation).max()
            except RuntimeError:
                continue

            reference_distance_bound = (
                self.cutoff + 2.0 * non_affine
            ) * inverse_stretch
            invalid[system_id] = reference_distance_bound > (
                self.cutoff + self.skin
            )
        return invalid

    def neighbor_list_needs_rebuild(self) -> bool:
        return bool(self.neighbor_list_invalid_systems().any().item())

    def ensure_neighbor_list(self) -> bool:
        """Rebuild only when required; return whether a rebuild occurred."""

        invalid = self.neighbor_list_invalid_systems()
        system_ids = torch.nonzero(invalid, as_tuple=False).flatten().tolist()
        if system_ids:
            self.rebuild_neighbor_list(system_ids)
            return True
        return False

    def rebuild_neighbor_list(
        self, system_ids: Sequence[int] | None = None
    ) -> None:
        """Rebuild selected candidate lists and retain clean graph topology."""

        ids = (
            list(range(self.n_systems))
            if system_ids is None
            else [int(value) for value in system_ids]
        )
        if len(set(ids)) != len(ids):
            raise ValueError("neighbor rebuild system_ids must be unique")
        if any(value < 0 or value >= self.n_systems for value in ids):
            raise IndexError("neighbor rebuild system id outside the batch")
        if not ids:
            return
        if self._neighbor_reference_positions is None:
            self._neighbor_reference_positions = self.positions.detach().clone()
        if self._neighbor_reference_cells is None:
            self._neighbor_reference_cells = self.cells.detach().clone()
        if self._neighbor_reference_valid is None:
            self._neighbor_reference_valid = torch.zeros(
                self.n_systems, device=self.device, dtype=torch.bool
            )
        rebuild_set = set(ids)

        selected_atom_blocks = [
            torch.arange(
                self.ptr[value],
                self.ptr[value + 1],
                device=self.device,
                dtype=torch.long,
            )
            for value in ids
        ]
        selected_atom_ids = torch.cat(selected_atom_blocks)
        selected_graph_ids = torch.as_tensor(
            ids, device=self.device, dtype=torch.long
        )

        with profile_phase(
            "graph.geometry_to_host",
            device=self.device,
            systems=len(ids),
            atoms=selected_atom_ids.numel(),
        ):
            selected_positions_cpu = (
                self.positions[selected_atom_ids].detach().cpu().numpy()
            )
            selected_cells_cpu = (
                self.cells[selected_graph_ids].detach().cpu().numpy()
            )
            selected_pbc_cpu = self.pbc[selected_graph_ids].detach().cpu().numpy()
            edge_counts = torch.bincount(
                self.system_idx[self.edge_index[0]], minlength=self.n_systems
            )
            edge_ptr = torch.cat(
                (
                    torch.zeros(1, device=self.device, dtype=torch.long),
                    edge_counts.cumsum(dim=0),
                )
            ).cpu().tolist()

        rebuilt_edges: dict[int, np.ndarray] = {}
        rebuilt_shifts: dict[int, np.ndarray] = {}

        with profile_phase(
            "graph.neighbor_search",
            device=self.device,
            systems=len(ids),
            atoms=selected_atom_ids.numel(),
        ):
            position_offset = 0
            for selected_idx, graph_idx in enumerate(ids):
                atom_slice = self.atom_slice(graph_idx)
                count = atom_slice.stop - atom_slice.start
                atoms = self.templates[graph_idx].copy()
                atoms.positions[:] = selected_positions_cpu[
                    position_offset : position_offset + count
                ]
                atoms.set_cell(selected_cells_cpu[selected_idx], scale_atoms=False)
                atoms.pbc = selected_pbc_cpu[selected_idx]
                position_offset += count

                i_idx, j_idx, shifts = neighbor_list(
                    "ijS", atoms, self.cutoff + self.skin
                )
                shifts = np.asarray(shifts, dtype=np.int64)
                order = np.lexsort(
                    (shifts[:, 2], shifts[:, 1], shifts[:, 0], j_idx, i_idx)
                )
                i_idx = i_idx[order]
                j_idx = j_idx[order]
                shifts = shifts[order]
                edge_block = np.stack((i_idx, j_idx), axis=0).astype(
                    np.int64, copy=False
                )
                edge_block += atom_slice.start
                rebuilt_edges[graph_idx] = edge_block
                rebuilt_shifts[graph_idx] = shifts

        with profile_phase(
            "graph.to_device",
            device=self.device,
            systems=len(ids),
            atoms=selected_atom_ids.numel(),
            edges=sum(block.shape[1] for block in rebuilt_edges.values()),
        ):
            edge_blocks = []
            shift_blocks = []
            for graph_idx in range(self.n_systems):
                if graph_idx in rebuild_set:
                    edge_blocks.append(
                        torch.as_tensor(
                            rebuilt_edges[graph_idx],
                            device=self.device,
                            dtype=torch.long,
                        )
                    )
                    shift_blocks.append(
                        torch.as_tensor(
                            rebuilt_shifts[graph_idx],
                            device=self.device,
                            dtype=torch.long,
                        )
                    )
                else:
                    edge_blocks.append(
                        self.edge_index[:, edge_ptr[graph_idx] : edge_ptr[graph_idx + 1]]
                    )
                    shift_blocks.append(
                        self.shifts_int[edge_ptr[graph_idx] : edge_ptr[graph_idx + 1]]
                    )
            self.edge_index = torch.cat(edge_blocks, dim=1)
            self.shifts_int = torch.cat(shift_blocks, dim=0)
            for system_id in ids:
                atom_slice = self.atom_slice(system_id)
                self._neighbor_reference_positions[atom_slice] = self.positions[
                    atom_slice
                ]
                self._neighbor_reference_cells[system_id] = self.cells[system_id]
                self._neighbor_reference_valid[system_id] = True
            self.neighbor_rebuild_count += 1
            self.assert_graph_integrity()
        profile_event(
            "neighbor_rebuild",
            resident_systems=self.n_systems,
            rebuilt_systems=len(ids),
            atoms=self.n_atoms,
            edges=self.edge_index.shape[1],
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
            edge_owners = self.system_idx[center]
            if bool((edge_owners[1:] < edge_owners[:-1]).any()):
                raise RuntimeError("neighbor edge blocks must be ordered by system")

    def as_model_data(
        self,
        *,
        positions: torch.Tensor | None = None,
        cells: torch.Tensor | None = None,
    ) -> GraphData:
        edge_index = self.edge_index
        shifts_int = self.shifts_int
        if self.skin > 0.0 and edge_index.shape[1] > 0:
            center, neighbor = edge_index
            # ASE evaluates neighbor cutoffs in float64 even when the stored
            # model geometry is float32. Match that topology decision here.
            topology_positions = self.positions.to(torch.float64)
            edge_cells = self.cells[self.system_idx[center]].to(torch.float64)
            cartesian_shifts = torch.bmm(
                shifts_int.unsqueeze(1).to(torch.float64), edge_cells
            ).squeeze(1)
            vectors = (
                topology_positions[center]
                - topology_positions[neighbor]
                - cartesian_shifts
            )
            physical = torch.linalg.vector_norm(vectors, dim=-1) < self.cutoff
            edge_index = edge_index[:, physical]
            shifts_int = shifts_int[physical]
        return GraphData(
            z=self.z,
            pos=self.positions if positions is None else positions,
            cell=self.cells if cells is None else cells,
            edge_index=edge_index,
            shifts_int=shifts_int,
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
            atoms.set_velocities(vel_cpu[atom_slice] / units.fs)
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

        Setting ``rebuild_neighbors=False`` preserves any available candidate
        graph and its reference geometry. Cache-cold systems remain invalid and
        are rebuilt independently by ``neighbor_policy='auto'``.
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
        edge_index = torch.empty((2, 0), device=self.device, dtype=torch.long)
        shifts_int = torch.empty((0, 3), device=self.device, dtype=torch.long)
        reference_positions = None
        reference_cells = None
        reference_valid = None
        if not rebuild_neighbors and self._neighbor_reference_positions is not None:
            old_to_new = torch.full(
                (self.n_atoms,), -1, device=self.device, dtype=torch.long
            )
            old_to_new[atom_ids] = torch.arange(
                atom_ids.numel(), device=self.device, dtype=torch.long
            )
            edge_owners = (
                self.system_idx[self.edge_index[0]]
                if self.edge_index.shape[1] > 0
                else torch.empty(0, device=self.device, dtype=torch.long)
            )
            edge_blocks = []
            shift_blocks = []
            for system_id in ids:
                edge_mask = edge_owners == system_id
                edge_blocks.append(old_to_new[self.edge_index[:, edge_mask]])
                shift_blocks.append(self.shifts_int[edge_mask])
            if edge_blocks:
                edge_index = torch.cat(edge_blocks, dim=1).clone()
                shifts_int = torch.cat(shift_blocks, dim=0).clone()
            reference_positions = self._neighbor_reference_positions[atom_ids].clone()
            if self._neighbor_reference_cells is None:
                raise RuntimeError("neighbor reference cells are missing")
            reference_cells = self._neighbor_reference_cells[graph_ids].clone()
            reference_valid = (
                torch.ones(len(ids), device=self.device, dtype=torch.bool)
                if self._neighbor_reference_valid is None
                else self._neighbor_reference_valid[graph_ids].clone()
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
            edge_index=edge_index,
            shifts_int=shifts_int,
            _neighbor_reference_positions=reference_positions,
            _neighbor_reference_cells=reference_cells,
            _neighbor_reference_valid=reference_valid,
        )
        if rebuild_neighbors:
            selected.rebuild_neighbor_list()
        else:
            selected.assert_graph_integrity()
        return selected

    @classmethod
    def concatenate(cls, batches: Sequence[AseGraphBatch]) -> AseGraphBatch:
        """Pack batches while retaining valid per-system neighbor caches."""

        parts = list(batches)
        if not parts:
            raise ValueError("batches must not be empty")
        first = parts[0]
        for part in parts[1:]:
            if (
                part.cutoff != first.cutoff
                or part.skin != first.skin
                or part.device != first.device
                or part.dtype != first.dtype
            ):
                raise ValueError(
                    "concatenated batches must have matching cutoff, skin, device, and dtype"
                )

        counts = torch.cat([part.counts for part in parts])
        ptr = torch.cat(
            (
                torch.zeros(1, device=first.device, dtype=torch.long),
                counts.cumsum(dim=0),
            )
        )
        system_idx = torch.repeat_interleave(
            torch.arange(counts.numel(), device=first.device, dtype=torch.long),
            counts,
        )
        edge_blocks = []
        shift_blocks = []
        atom_offset = 0
        for part in parts:
            edge_blocks.append(part.edge_index + atom_offset)
            shift_blocks.append(part.shifts_int)
            atom_offset += part.n_atoms

        has_references = any(
            part._neighbor_reference_positions is not None for part in parts
        )
        reference_positions = None
        reference_cells = None
        reference_valid = None
        if has_references:
            position_blocks = []
            cell_blocks = []
            valid_blocks = []
            for part in parts:
                if part._neighbor_reference_positions is None:
                    position_blocks.append(part.positions.detach())
                    cell_blocks.append(part.cells.detach())
                    valid_blocks.append(
                        torch.zeros(
                            part.n_systems, device=first.device, dtype=torch.bool
                        )
                    )
                else:
                    if part._neighbor_reference_cells is None:
                        raise RuntimeError("neighbor reference cells are missing")
                    position_blocks.append(part._neighbor_reference_positions)
                    cell_blocks.append(part._neighbor_reference_cells)
                    valid_blocks.append(
                        torch.ones(
                            part.n_systems, device=first.device, dtype=torch.bool
                        )
                        if part._neighbor_reference_valid is None
                        else part._neighbor_reference_valid
                    )
            reference_positions = torch.cat(position_blocks).clone()
            reference_cells = torch.cat(cell_blocks).clone()
            reference_valid = torch.cat(valid_blocks).clone()

        packed = cls(
            templates=[template.copy() for part in parts for template in part.templates],
            cutoff=first.cutoff,
            skin=first.skin,
            device=first.device,
            dtype=first.dtype,
            z=torch.cat([part.z for part in parts]).clone(),
            positions=torch.cat([part.positions for part in parts]).clone(),
            cells=torch.cat([part.cells for part in parts]).clone(),
            pbc=torch.cat([part.pbc for part in parts]).clone(),
            system_idx=system_idx,
            ptr=ptr,
            masses=torch.cat([part.masses for part in parts]).clone(),
            fixed=torch.cat([part.fixed for part in parts]).clone(),
            velocities=torch.cat([part.velocities for part in parts]).clone(),
            edge_index=torch.cat(edge_blocks, dim=1).clone(),
            shifts_int=torch.cat(shift_blocks, dim=0).clone(),
            _neighbor_reference_positions=reference_positions,
            _neighbor_reference_cells=reference_cells,
            _neighbor_reference_valid=reference_valid,
        )
        packed.assert_graph_integrity()
        return packed
