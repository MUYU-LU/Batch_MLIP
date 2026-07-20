"""Memory-bounded dense neighbor construction with torch tensors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

DEFAULT_MAX_WORK_BYTES = 512 * 1024**2


class DenseNeighborUnsupportedError(RuntimeError):
    """Raised when dense construction cannot safely represent a cell."""


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


def _image_extents(
    positions: torch.Tensor,
    cells: torch.Tensor,
    periodic: torch.Tensor,
    ptr_values: list[int],
    system_ids: list[int],
    cutoff: float,
) -> list[tuple[int, int, int]]:
    selected_positions = torch.cat(
        [positions[ptr_values[system_id] : ptr_values[system_id + 1]] for system_id in system_ids]
    )
    position_values = selected_positions.detach().cpu().numpy().astype(np.float64, copy=False)
    cell_values = cells.detach().cpu().numpy().astype(np.float64, copy=False)
    periodic_values = periodic.detach().cpu().numpy()
    results = []
    position_offset = 0
    for system_id, cell, mask in zip(system_ids, cell_values, periodic_values, strict=True):
        atom_count = ptr_values[system_id + 1] - ptr_values[system_id]
        periodic_count = int(mask.sum())
        if periodic_count == 0:
            results.append((0, 0, 0))
            position_offset += atom_count
            continue
        if np.linalg.matrix_rank(cell[mask]) != periodic_count:
            raise DenseNeighborUnsupportedError(
                "periodic cell vectors must be linearly independent"
            )
        reciprocal = np.zeros((3, 3), dtype=np.float64)
        reciprocal[:, mask] = (
            np.linalg.inv(cell) if periodic_count == 3 else np.linalg.pinv(cell[mask])
        )
        reciprocal_norm = np.linalg.norm(reciprocal, axis=0)
        atom_slice = slice(position_offset, position_offset + atom_count)
        fractional = position_values[atom_slice] @ reciprocal
        fractional[:, mask] -= np.floor(fractional[:, mask])
        span = np.ptp(fractional, axis=0)
        # For |delta_r| < cutoff, the reciprocal-space component obeys
        # |delta_f[k] + shift[k]| < cutoff * |b[k]|. Wrapped coordinates
        # restrict |delta_f[k]| to the occupied span, tightening the safe
        # integer image range without excluding a possible neighbor.
        extents = np.where(
            mask,
            np.maximum(0, np.ceil(cutoff * reciprocal_norm + span) - 1),
            0,
        ).astype(np.int64)
        results.append(tuple(int(value) for value in extents))
        position_offset += atom_count
    return results


def _shift_grid(extents: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    axes = [
        torch.arange(-extent, extent + 1, device=device, dtype=torch.long) for extent in extents
    ]
    return torch.cartesian_prod(*axes).reshape(-1, 3)


def _microbatch_size(atom_count: int, shift_count: int, max_work_bytes: int) -> int:
    # Float64 base and displacement tensors plus the boolean candidate mask.
    bytes_per_system = atom_count**2 * (2 * 3 * 8 + shift_count)
    return max(1, max_work_bytes // max(1, bytes_per_system))


def _build_compatible_group(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    ptr_values: list[int],
    system_ids: list[int],
    *,
    cutoff: float,
    extents: tuple[int, int, int],
    max_work_bytes: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    device = positions.device
    atom_count = ptr_values[system_ids[0] + 1] - ptr_values[system_ids[0]]
    shifts = _shift_grid(extents, device)
    chunk_size = _microbatch_size(atom_count, shifts.shape[0], max_work_bytes)
    edge_parts: list[torch.Tensor] = []
    shift_parts: list[torch.Tensor] = []
    owner_parts: list[torch.Tensor] = []
    cutoff_squared = cutoff * cutoff
    zero_shift = torch.all(shifts == 0, dim=1)

    for start in range(0, len(system_ids), chunk_size):
        chunk_ids = system_ids[start : start + chunk_size]
        graph_ids = torch.as_tensor(chunk_ids, device=device, dtype=torch.long)
        chunk_cells = cells[graph_ids].to(torch.float64)
        chunk_pbc = pbc[graph_ids]
        atom_ids = ptr[graph_ids, None] + torch.arange(atom_count, device=device, dtype=torch.long)
        chunk_positions = positions[atom_ids].to(torch.float64)

        if bool(chunk_pbc.any()):
            cell_values = chunk_cells.detach().cpu().numpy()
            periodic_values = chunk_pbc.detach().cpu().numpy()
            reciprocal_values = np.zeros_like(cell_values)
            for index, (cell, mask) in enumerate(zip(cell_values, periodic_values, strict=True)):
                reciprocal_values[index][:, mask] = (
                    np.linalg.inv(cell) if bool(mask.all()) else np.linalg.pinv(cell[mask])
                )
            reciprocal = torch.as_tensor(
                reciprocal_values,
                device=device,
                dtype=torch.float64,
            )
            fractional = torch.bmm(chunk_positions, reciprocal)
            wrap_offsets = torch.floor(fractional).to(torch.long)
            wrap_offsets = torch.where(
                chunk_pbc[:, None, :], wrap_offsets, torch.zeros_like(wrap_offsets)
            )
            wrapped_positions = chunk_positions - torch.bmm(
                wrap_offsets.to(torch.float64), chunk_cells
            )
        else:
            wrap_offsets = torch.zeros_like(chunk_positions, dtype=torch.long)
            wrapped_positions = chunk_positions
        base = wrapped_positions[:, None, :, :] - wrapped_positions[:, :, None, :]
        candidates = torch.empty(
            (
                len(chunk_ids),
                atom_count,
                atom_count,
                shifts.shape[0],
            ),
            device=device,
            dtype=torch.bool,
        )
        # Compatible groups share one PBC key, but retaining this mask keeps
        # the kernel correct if grouping changes later.
        shift_valid = torch.all(chunk_pbc[:, None, :] | (shifts[None, :, :] == 0), dim=2)

        for shift_index in range(shifts.shape[0]):
            cartesian_shift = torch.einsum(
                "d,bdk->bk", shifts[shift_index].to(torch.float64), chunk_cells
            )
            delta = base + cartesian_shift[:, None, None, :]
            candidates[..., shift_index] = (
                torch.sum(delta * delta, dim=-1) < cutoff_squared
            ) & shift_valid[:, shift_index, None, None]
        if bool(zero_shift.any()):
            zero_index = int(torch.nonzero(zero_shift, as_tuple=False)[0].item())
            diagonal = torch.arange(atom_count, device=device)
            candidates[:, diagonal, diagonal, zero_index] = False

        entries = torch.nonzero(candidates, as_tuple=False)
        if entries.numel() == 0:
            continue
        batch_ids, centers, neighbors, shift_ids = entries.unbind(dim=1)
        owners = graph_ids[batch_ids]
        atom_offsets = ptr[owners]
        edge_parts.append(torch.stack((centers + atom_offsets, neighbors + atom_offsets)))
        shift_parts.append(
            shifts[shift_ids]
            + wrap_offsets[batch_ids, centers]
            - wrap_offsets[batch_ids, neighbors]
        )
        owner_parts.append(owners)
    return edge_parts, shift_parts, owner_parts


def dense_neighbor_blocks(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    system_ids: Sequence[int],
    *,
    cutoff: float,
    max_work_bytes: int = DEFAULT_MAX_WORK_BYTES,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Construct canonical directed neighbor blocks with device-resident output.

    Topology decisions use float64 regardless of model dtype. Returned edge
    indices are global to ``positions`` and are ordered by center, neighbor,
    then integer shift within each requested system. Only selected coordinates
    and cells are copied to CPU for small reciprocal/image-range metadata.
    """

    ids = _validate_inputs(positions, cells, pbc, ptr, system_ids, cutoff, max_work_bytes)
    empty_edge = torch.empty((2, 0), device=positions.device, dtype=torch.long)
    empty_shift = torch.empty((0, 3), device=positions.device, dtype=torch.long)
    if not ids:
        return {}

    ptr_values = ptr.detach().cpu().tolist()
    counts = [stop - start for start, stop in zip(ptr_values, ptr_values[1:], strict=False)]
    graph_ids = torch.as_tensor(ids, device=positions.device, dtype=torch.long)
    periodic_values = pbc[graph_ids].detach().cpu().tolist()
    extent_values = _image_extents(
        positions,
        cells[graph_ids],
        pbc[graph_ids],
        ptr_values,
        ids,
        cutoff,
    )
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for system_id, periodic, extents in zip(ids, periodic_values, extent_values, strict=True):
        key = (
            int(counts[system_id]),
            *(bool(value) for value in periodic),
            *extents,
        )
        groups[key].append(system_id)

    edge_parts: list[torch.Tensor] = []
    shift_parts: list[torch.Tensor] = []
    owner_parts: list[torch.Tensor] = []
    for key, group_ids in groups.items():
        extents = tuple(int(value) for value in key[-3:])
        group_edges, group_shifts, group_owners = _build_compatible_group(
            positions,
            cells,
            pbc,
            ptr,
            ptr_values,
            group_ids,
            cutoff=cutoff,
            extents=extents,
            max_work_bytes=max_work_bytes,
        )
        edge_parts.extend(group_edges)
        shift_parts.extend(group_shifts)
        owner_parts.extend(group_owners)

    if not edge_parts:
        return {value: (empty_edge.clone(), empty_shift.clone()) for value in ids}
    edges = torch.cat(edge_parts, dim=1)
    shifts_int = torch.cat(shift_parts, dim=0)
    owners = torch.cat(owner_parts)

    # Each system occurs in exactly one compatible group and microbatch. A
    # stable owner sort therefore restores system order without disturbing the
    # canonical center/neighbor/shift order emitted by torch.nonzero.
    order = torch.argsort(owners, stable=True)
    edges = edges[:, order]
    shifts_int = shifts_int[order]
    owners = owners[order]
    counts_by_owner = torch.bincount(owners, minlength=cells.shape[0]).cpu().tolist()
    edge_ptr = [0]
    for count in counts_by_owner:
        edge_ptr.append(edge_ptr[-1] + count)
    return {
        system_id: (
            edges[:, edge_ptr[system_id] : edge_ptr[system_id + 1]],
            shifts_int[edge_ptr[system_id] : edge_ptr[system_id + 1]],
        )
        for system_id in ids
    }
