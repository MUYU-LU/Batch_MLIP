"""Sparse CUDA cell-list construction for fully periodic graph batches."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .dense_neighbors import DEFAULT_MAX_WORK_BYTES


class CellListUnsupportedError(RuntimeError):
    """Raised when sparse cell-list construction cannot safely handle a cell."""


@dataclass(frozen=True)
class _CellMetadata:
    bins: tuple[int, int, int]
    offset_extents: tuple[int, int, int]


def estimate_cell_candidate_reduction(
    cells: torch.Tensor,
    pbc: torch.Tensor,
    *,
    cutoff: float,
    positions: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
) -> float | None:
    """Estimate sparse candidate reduction under uniform bin occupancy."""

    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3):
        raise ValueError("cells must have shape [B, 3, 3]")
    if pbc.shape != (cells.shape[0], 3):
        raise ValueError("pbc must have shape [B, 3]")
    if (positions is None) != (counts is None):
        raise ValueError("positions and counts must be provided together")
    if positions is not None:
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        if counts.shape != (cells.shape[0],):
            raise ValueError("counts must have shape [B]")
        count_values = counts.detach().cpu().tolist()
        if sum(count_values) != positions.shape[0]:
            raise ValueError("counts must sum to the number of positions")
        position_values = (
            positions.detach().cpu().numpy().astype(np.float64, copy=False)
        )
    else:
        count_values = None
        position_values = None
    if not bool(pbc.all()):
        return None
    cell_values = cells.detach().cpu().numpy().astype(np.float64, copy=False)
    reductions = []
    atom_offset = 0
    for system_index, cell in enumerate(cell_values):
        if np.linalg.matrix_rank(cell) != 3:
            return None
        inverse = np.linalg.inv(cell)
        reciprocal_norm = np.linalg.norm(inverse, axis=0)
        fractional_radius = cutoff * reciprocal_norm
        bins = np.maximum(1, np.floor(1.0 / fractional_radius)).astype(np.int64)
        offset_extents = np.ceil(fractional_radius * bins).astype(np.int64)
        if position_values is None:
            span = np.ones(3, dtype=np.float64)
        else:
            atom_count = int(count_values[system_index])
            system_positions = position_values[
                atom_offset : atom_offset + atom_count
            ]
            if atom_count:
                fractional = system_positions @ inverse
                fractional -= np.floor(fractional)
                span = np.ptp(fractional, axis=0)
            else:
                span = np.zeros(3, dtype=np.float64)
            atom_offset += atom_count
        image_extents = np.maximum(
            0,
            np.ceil(fractional_radius + span) - 1,
        ).astype(np.int64)
        cell_candidates = math.prod(2 * offset_extents + 1)
        dense_candidates = math.prod(bins) * math.prod(2 * image_extents + 1)
        reductions.append(1.0 - cell_candidates / dense_candidates)
    return min(reductions, default=None)


def _validate_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    system_ids: Sequence[int],
    cutoff: float,
    max_work_bytes: int,
) -> list[int]:
    ids = [int(value) for value in system_ids]
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if max_work_bytes <= 0:
        raise ValueError("max_work_bytes must be positive")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3):
        raise ValueError("cells must have shape [B, 3, 3]")
    if pbc.shape != (cells.shape[0], 3):
        raise ValueError("pbc must have shape [B, 3]")
    if ptr.shape != (cells.shape[0] + 1,):
        raise ValueError("ptr must have shape [B + 1]")
    if positions.device != cells.device or positions.device != pbc.device:
        raise ValueError("positions, cells, and pbc must use the same device")
    if ptr.device != positions.device:
        raise ValueError("ptr must use the geometry device")
    if len(set(ids)) != len(ids):
        raise ValueError("system_ids must be unique")
    if any(value < 0 or value >= cells.shape[0] for value in ids):
        raise IndexError("system id outside the batch")
    return ids


def _metadata(
    cells: torch.Tensor,
    pbc: torch.Tensor,
    system_ids: list[int],
    cutoff: float,
) -> dict[int, _CellMetadata]:
    graph_ids = torch.as_tensor(system_ids, device=cells.device, dtype=torch.long)
    cell_values = cells[graph_ids].detach().cpu().numpy().astype(np.float64, copy=False)
    periodic_values = pbc[graph_ids].detach().cpu().numpy()
    metadata = {}
    for system_id, cell, periodic in zip(
        system_ids,
        cell_values,
        periodic_values,
        strict=True,
    ):
        if not bool(periodic.all()):
            raise CellListUnsupportedError(
                "cuda_cell currently requires three-dimensional periodic cells"
            )
        if np.linalg.matrix_rank(cell) != 3:
            raise CellListUnsupportedError("periodic cell vectors must be independent")
        reciprocal_norm = np.linalg.norm(np.linalg.inv(cell), axis=0)
        fractional_radius = cutoff * reciprocal_norm
        bins = np.maximum(1, np.floor(1.0 / fractional_radius)).astype(np.int64)
        offset_extents = np.ceil(fractional_radius * bins).astype(np.int64)
        metadata[system_id] = _CellMetadata(
            bins=tuple(int(value) for value in bins),
            offset_extents=tuple(int(value) for value in offset_extents),
        )
    return metadata


def _offset_grid(
    extents: tuple[int, int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    axes = [
        torch.arange(-extent, extent + 1, device=device, dtype=torch.long)
        for extent in extents
    ]
    return torch.cartesian_prod(*axes).reshape(-1, 3)


def _linear_bins(coordinates: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return (
        (coordinates[:, 0] * bins[1] + coordinates[:, 1]) * bins[2]
        + coordinates[:, 2]
    )


def _candidate_chunk_size(
    *,
    offset_count: int,
    max_occupancy: int,
    max_work_bytes: int,
) -> int:
    # Query tensors and expanded candidate tensors are predominantly int64.
    bytes_per_center = offset_count * (96 + 112 * max_occupancy)
    if bytes_per_center > max_work_bytes:
        raise CellListUnsupportedError(
            "one sparse cell-list center exceeds the temporary-work budget"
        )
    return max(1, min(8192, max_work_bytes // max(1, bytes_per_center)))


def _build_group(
    positions: torch.Tensor,
    cells: torch.Tensor,
    ptr_values: list[int],
    system_ids: list[int],
    cell_metadata: _CellMetadata,
    *,
    cutoff: float,
    max_work_bytes: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    device = positions.device
    counts = [
        ptr_values[system_id + 1] - ptr_values[system_id]
        for system_id in system_ids
    ]
    atom_ids = torch.cat(
        [
            torch.arange(
                ptr_values[system_id],
                ptr_values[system_id + 1],
                device=device,
                dtype=torch.long,
            )
            for system_id in system_ids
        ]
    )
    graph_ids = torch.as_tensor(system_ids, device=device, dtype=torch.long)
    owners = torch.repeat_interleave(
        torch.arange(len(system_ids), device=device, dtype=torch.long),
        torch.as_tensor(counts, device=device, dtype=torch.long),
    )
    group_cells = cells[graph_ids].to(torch.float64)
    inverse_cells = torch.linalg.inv(group_cells)
    group_positions = positions[atom_ids].to(torch.float64)
    fractional = torch.bmm(
        group_positions.unsqueeze(1),
        inverse_cells[owners],
    ).squeeze(1)
    wrap_offsets = torch.floor(fractional).to(torch.long)
    wrapped_fractional = fractional - wrap_offsets.to(torch.float64)
    wrapped_positions = group_positions - torch.bmm(
        wrap_offsets.to(torch.float64).unsqueeze(1),
        group_cells[owners],
    ).squeeze(1)

    bins = torch.as_tensor(cell_metadata.bins, device=device, dtype=torch.long)
    bin_coordinates = torch.floor(
        wrapped_fractional * bins.to(torch.float64)
    ).to(torch.long)
    bin_coordinates = torch.minimum(bin_coordinates, bins - 1)
    bins_per_system = math.prod(cell_metadata.bins)
    bin_keys = owners * bins_per_system + _linear_bins(bin_coordinates, bins)
    sorted_keys, atom_order = torch.sort(bin_keys)
    occupancy = torch.bincount(
        bin_keys,
        minlength=len(system_ids) * bins_per_system,
    )
    max_occupancy = int(occupancy.max().item())

    offsets = _offset_grid(cell_metadata.offset_extents, device=device)
    offset_count = offsets.shape[0]
    chunk_size = _candidate_chunk_size(
        offset_count=offset_count,
        max_occupancy=max_occupancy,
        max_work_bytes=max_work_bytes,
    )
    cutoff_squared = cutoff * cutoff
    edge_parts = []
    shift_parts = []
    owner_parts = []
    for start in range(0, atom_ids.shape[0], chunk_size):
        stop = min(start + chunk_size, atom_ids.shape[0])
        center_queries = torch.arange(
            start,
            stop,
            device=device,
            dtype=torch.long,
        ).repeat_interleave(offset_count)
        query_offsets = offsets.repeat(stop - start, 1)
        target_coordinates = bin_coordinates[center_queries] + query_offsets
        image_shifts = torch.div(
            target_coordinates,
            bins,
            rounding_mode="floor",
        )
        target_coordinates = torch.remainder(target_coordinates, bins)
        target_keys = (
            owners[center_queries] * bins_per_system
            + _linear_bins(target_coordinates, bins)
        )
        starts = torch.searchsorted(sorted_keys, target_keys, side="left")
        stops = torch.searchsorted(sorted_keys, target_keys, side="right")
        candidate_counts = stops - starts
        candidate_count = int(candidate_counts.sum().item())
        if candidate_count == 0:
            continue
        query_ids = torch.repeat_interleave(
            torch.arange(
                center_queries.shape[0],
                device=device,
                dtype=torch.long,
            ),
            candidate_counts,
        )
        candidate_offsets = torch.cumsum(candidate_counts, dim=0) - candidate_counts
        within_bin = torch.arange(
            candidate_count,
            device=device,
            dtype=torch.long,
        ) - torch.repeat_interleave(candidate_offsets, candidate_counts)
        neighbor_sorted_ids = starts[query_ids] + within_bin
        neighbor_group_ids = atom_order[neighbor_sorted_ids]
        center_group_ids = center_queries[query_ids]
        candidate_image_shifts = image_shifts[query_ids]
        output_shifts = (
            candidate_image_shifts
            + wrap_offsets[center_group_ids]
            - wrap_offsets[neighbor_group_ids]
        )
        not_self = (center_group_ids != neighbor_group_ids) | torch.any(
            candidate_image_shifts != 0,
            dim=1,
        )
        delta = (
            wrapped_positions[neighbor_group_ids]
            - wrapped_positions[center_group_ids]
            + torch.bmm(
                candidate_image_shifts.to(torch.float64).unsqueeze(1),
                group_cells[owners[center_group_ids]],
            ).squeeze(1)
        )
        physical = not_self & (torch.sum(delta * delta, dim=1) < cutoff_squared)
        if not bool(physical.any()):
            continue
        center_group_ids = center_group_ids[physical]
        neighbor_group_ids = neighbor_group_ids[physical]
        edge_parts.append(
            torch.stack(
                (
                    atom_ids[center_group_ids],
                    atom_ids[neighbor_group_ids],
                )
            )
        )
        shift_parts.append(output_shifts[physical])
        owner_parts.append(graph_ids[owners[center_group_ids]])
    return edge_parts, shift_parts, owner_parts


def _canonical_order(
    edges: torch.Tensor,
    shifts: torch.Tensor,
    *,
    n_atoms: int,
) -> torch.Tensor:
    minimum = shifts.min(dim=0).values.detach().cpu().tolist()
    maximum = shifts.max(dim=0).values.detach().cpu().tolist()
    ranges = [
        int(high) - int(low) + 1
        for low, high in zip(minimum, maximum, strict=True)
    ]
    largest_key = n_atoms * n_atoms
    for radix in ranges:
        largest_key *= radix
    if largest_key < torch.iinfo(torch.long).max:
        key = edges[0] * n_atoms + edges[1]
        for dimension, (low, radix) in enumerate(
            zip(minimum, ranges, strict=True)
        ):
            key = key * radix + shifts[:, dimension] - int(low)
        return torch.argsort(key)

    order = torch.arange(edges.shape[1], device=edges.device)
    values = (
        shifts[:, 2],
        shifts[:, 1],
        shifts[:, 0],
        edges[1],
        edges[0],
    )
    for value in values:
        order = order[torch.argsort(value[order], stable=True)]
    return order


def cell_list_neighbor_blocks(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    system_ids: Sequence[int],
    *,
    cutoff: float,
    max_work_bytes: int = DEFAULT_MAX_WORK_BYTES,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Construct exact directed neighbor blocks using periodic spatial bins."""

    ids = _validate_inputs(
        positions,
        cells,
        pbc,
        ptr,
        system_ids,
        cutoff,
        max_work_bytes,
    )
    if not ids:
        return {}
    ptr_values = ptr.detach().cpu().tolist()
    metadata = _metadata(cells, pbc, ids, cutoff)
    groups: dict[_CellMetadata, list[int]] = defaultdict(list)
    for system_id in ids:
        groups[metadata[system_id]].append(system_id)

    edge_parts = []
    shift_parts = []
    owner_parts = []
    for cell_metadata, group_ids in groups.items():
        group_edges, group_shifts, group_owners = _build_group(
            positions,
            cells,
            ptr_values,
            group_ids,
            cell_metadata,
            cutoff=cutoff,
            max_work_bytes=max_work_bytes,
        )
        edge_parts.extend(group_edges)
        shift_parts.extend(group_shifts)
        owner_parts.extend(group_owners)

    empty_edge = torch.empty((2, 0), device=positions.device, dtype=torch.long)
    empty_shift = torch.empty((0, 3), device=positions.device, dtype=torch.long)
    if not edge_parts:
        return {
            system_id: (empty_edge.clone(), empty_shift.clone())
            for system_id in ids
        }
    edges = torch.cat(edge_parts, dim=1)
    shifts = torch.cat(shift_parts, dim=0)
    owners = torch.cat(owner_parts)
    order = _canonical_order(edges, shifts, n_atoms=positions.shape[0])
    edges = edges[:, order]
    shifts = shifts[order]
    owners = owners[order]
    owner_counts = torch.bincount(owners, minlength=cells.shape[0]).cpu().tolist()
    edge_ptr = [0]
    for count in owner_counts:
        edge_ptr.append(edge_ptr[-1] + int(count))
    return {
        system_id: (
            edges[:, edge_ptr[system_id] : edge_ptr[system_id + 1]],
            shifts[edge_ptr[system_id] : edge_ptr[system_id + 1]],
        )
        for system_id in ids
    }
