"""Reusable compact storage for heterogeneous resident batches."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from .state import AseGraphBatch


@dataclass(frozen=True)
class SystemSelection:
    """Ordered system IDs selected from one graph batch."""

    state: AseGraphBatch
    system_ids: tuple[int, ...]


@dataclass
class _ArenaBank:
    device: torch.device
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        minimum_capacity: int = 0,
    ) -> torch.Tensor:
        if not shape:
            raise ValueError("arena tensor shape must not be empty")
        required = shape[0]
        existing = self.buffers.get(name)
        tail = shape[1:]
        compatible = (
            existing is not None
            and existing.dtype == dtype
            and existing.shape[1:] == tail
        )
        if not compatible or existing.shape[0] < required:
            previous = 0 if not compatible else existing.shape[0]
            capacity = max(required, minimum_capacity, int(math.ceil(previous * 1.5)))
            existing = torch.empty(
                (capacity, *tail),
                device=self.device,
                dtype=dtype,
            )
            self.buffers[name] = existing
        return existing[:required]


class HeterogeneousResidentArena:
    """Pack selected systems into alternating reusable compact tensor banks."""

    def __init__(
        self,
        pool: AseGraphBatch,
        *,
        resident_capacity: int,
    ) -> None:
        if resident_capacity <= 0:
            raise ValueError("resident_capacity must be positive")
        if resident_capacity > pool.n_systems:
            raise ValueError("resident_capacity cannot exceed the pool size")
        largest_counts = sorted(pool.counts.tolist(), reverse=True)[
            :resident_capacity
        ]
        self.max_systems = resident_capacity
        self.max_atoms = sum(int(value) for value in largest_counts)
        self._banks = (_ArenaBank(pool.device), _ArenaBank(pool.device))
        self._next_bank = 0
        self._active_bank: _ArenaBank | None = None

    def pack(
        self,
        selections: Sequence[SystemSelection],
    ) -> AseGraphBatch:
        """Return a compact batch without allocating intermediate sub-batches."""

        ordered = [
            (selection.state, int(system_id))
            for selection in selections
            for system_id in selection.system_ids
        ]
        if not ordered:
            raise ValueError("arena selections must contain at least one system")
        if len(ordered) > self.max_systems:
            raise ValueError("arena selection exceeds resident system capacity")
        first = ordered[0][0]
        for source, system_id in ordered:
            if not 0 <= system_id < source.n_systems:
                raise IndexError("arena source system id outside the batch")
            if (
                source.cutoff != first.cutoff
                or source.skin != first.skin
                or source.device != first.device
                or source.dtype != first.dtype
                or source.neighbor_backend != first.neighbor_backend
            ):
                raise ValueError("arena graph settings must match")

        source_metadata: dict[int, tuple[list[int], list[int]]] = {}
        for source, _ in ordered:
            key = id(source)
            if key in source_metadata:
                continue
            atom_ptr = source.ptr.detach().cpu().tolist()
            edge_counts_tensor = (
                torch.bincount(
                    source.system_idx[source.edge_index[0]],
                    minlength=source.n_systems,
                )
                if source.edge_index.shape[1]
                else torch.zeros(
                    source.n_systems,
                    device=source.device,
                    dtype=torch.long,
                )
            )
            edge_counts_cpu = edge_counts_tensor.detach().cpu().tolist()
            edge_ptr = [0]
            for count in edge_counts_cpu:
                edge_ptr.append(edge_ptr[-1] + int(count))
            source_metadata[key] = (atom_ptr, edge_ptr)
        counts = [
            source_metadata[id(source)][0][system_id + 1]
            - source_metadata[id(source)][0][system_id]
            for source, system_id in ordered
        ]
        n_systems = len(ordered)
        n_atoms = sum(counts)
        if n_atoms > self.max_atoms:
            raise ValueError("arena selection exceeds resident atom capacity")
        edge_counts = [
            source_metadata[id(source)][1][system_id + 1]
            - source_metadata[id(source)][1][system_id]
            for source, system_id in ordered
        ]
        n_edges = sum(edge_counts)

        bank = self._banks[self._next_bank]
        self._next_bank = 1 - self._next_bank
        self._active_bank = bank
        z = bank.tensor(
            "z",
            (n_atoms,),
            dtype=torch.long,
            minimum_capacity=self.max_atoms,
        )
        positions = bank.tensor(
            "positions",
            (n_atoms, 3),
            dtype=first.dtype,
            minimum_capacity=self.max_atoms,
        )
        cells = bank.tensor(
            "cells",
            (n_systems, 3, 3),
            dtype=first.dtype,
            minimum_capacity=self.max_systems,
        )
        pbc = bank.tensor(
            "pbc",
            (n_systems, 3),
            dtype=torch.bool,
            minimum_capacity=self.max_systems,
        )
        system_idx = bank.tensor(
            "system_idx",
            (n_atoms,),
            dtype=torch.long,
            minimum_capacity=self.max_atoms,
        )
        ptr = bank.tensor(
            "ptr",
            (n_systems + 1,),
            dtype=torch.long,
            minimum_capacity=self.max_systems + 1,
        )
        masses = bank.tensor(
            "masses",
            (n_atoms,),
            dtype=first.dtype,
            minimum_capacity=self.max_atoms,
        )
        fixed = bank.tensor(
            "fixed",
            (n_atoms,),
            dtype=torch.bool,
            minimum_capacity=self.max_atoms,
        )
        velocities = bank.tensor(
            "velocities",
            (n_atoms, 3),
            dtype=first.dtype,
            minimum_capacity=self.max_atoms,
        )
        edge_index_rows = bank.tensor(
            "edge_index_rows",
            (n_edges, 2),
            dtype=torch.long,
        )
        shifts_int = bank.tensor(
            "shifts_int",
            (n_edges, 3),
            dtype=torch.long,
        )
        reference_positions = bank.tensor(
            "reference_positions",
            (n_atoms, 3),
            dtype=first.dtype,
            minimum_capacity=self.max_atoms,
        )
        reference_cells = bank.tensor(
            "reference_cells",
            (n_systems, 3, 3),
            dtype=first.dtype,
            minimum_capacity=self.max_systems,
        )
        reference_valid = bank.tensor(
            "reference_valid",
            (n_systems,),
            dtype=torch.bool,
            minimum_capacity=self.max_systems,
        )

        templates = []
        atom_offset = 0
        edge_offset = 0
        ptr[0] = 0
        for destination, ((source, system_id), atom_count, edge_count) in enumerate(
            zip(ordered, counts, edge_counts, strict=True)
        ):
            atom_ptr, edge_ptr = source_metadata[id(source)]
            source_atoms = slice(
                atom_ptr[system_id],
                atom_ptr[system_id + 1],
            )
            destination_atoms = slice(atom_offset, atom_offset + atom_count)
            templates.append(source.templates[system_id].copy())
            z[destination_atoms].copy_(source.z[source_atoms])
            positions[destination_atoms].copy_(source.positions[source_atoms])
            cells[destination].copy_(source.cells[system_id])
            pbc[destination].copy_(source.pbc[system_id])
            system_idx[destination_atoms].fill_(destination)
            masses[destination_atoms].copy_(source.masses[source_atoms])
            fixed[destination_atoms].copy_(source.fixed[source_atoms])
            velocities[destination_atoms].copy_(source.velocities[source_atoms])
            ptr[destination + 1] = atom_offset + atom_count

            if source._neighbor_reference_positions is None:
                reference_positions[destination_atoms].copy_(
                    source.positions[source_atoms]
                )
                reference_cells[destination].copy_(source.cells[system_id])
                reference_valid[destination] = False
            else:
                if source._neighbor_reference_cells is None:
                    raise RuntimeError("neighbor reference cells are missing")
                reference_positions[destination_atoms].copy_(
                    source._neighbor_reference_positions[source_atoms]
                )
                reference_cells[destination].copy_(
                    source._neighbor_reference_cells[system_id]
                )
                reference_valid[destination] = (
                    True
                    if source._neighbor_reference_valid is None
                    else source._neighbor_reference_valid[system_id]
                )

            if edge_count:
                source_edge_slice = slice(
                    edge_ptr[system_id],
                    edge_ptr[system_id + 1],
                )
                source_edges = source.edge_index[:, source_edge_slice]
                destination_edges = slice(edge_offset, edge_offset + edge_count)
                edge_index_rows[destination_edges].copy_(
                    (
                        source_edges
                        - source_atoms.start
                        + atom_offset
                    ).transpose(0, 1)
                )
                shifts_int[destination_edges].copy_(
                    source.shifts_int[source_edge_slice]
                )
                edge_offset += edge_count
            atom_offset += atom_count

        packed = AseGraphBatch(
            templates=templates,
            cutoff=first.cutoff,
            skin=first.skin,
            device=first.device,
            dtype=first.dtype,
            neighbor_backend=first.neighbor_backend,
            z=z,
            positions=positions,
            cells=cells,
            pbc=pbc,
            system_idx=system_idx,
            ptr=ptr,
            masses=masses,
            fixed=fixed,
            velocities=velocities,
            edge_index=edge_index_rows.transpose(0, 1),
            shifts_int=shifts_int,
            _neighbor_reference_positions=reference_positions,
            _neighbor_reference_cells=reference_cells,
            _neighbor_reference_valid=reference_valid,
        )
        packed.assert_graph_integrity()
        return packed

    def work_tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        zero: bool = False,
    ) -> torch.Tensor:
        """Return a tensor aligned with the most recently packed bank."""

        if self._active_bank is None:
            raise RuntimeError("pack must be called before requesting work tensors")
        tensor = self._active_bank.tensor(name, shape, dtype=dtype)
        if zero:
            tensor.zero_()
        return tensor
