"""Optional NVIDIA ALCHEMI neighbor-list adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .cell_neighbors import _canonical_order


class NValchemiUnavailableError(ImportError):
    """Raised when the optional NVIDIA neighbor backend is not installed."""


def _load_neighbor_api() -> Callable[..., Any]:
    try:
        from nvalchemiops.torch.neighbors import neighbor_list
    except ImportError as error:
        raise NValchemiUnavailableError(
            "neighbor_backend='nvalchemi' requires nvalchemi-toolkit-ops"
        ) from error
    return neighbor_list


def _validate_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    system_ids: Sequence[int],
    cutoff: float,
) -> list[int]:
    ids = [int(value) for value in system_ids]
    if positions.device.type != "cuda":
        raise ValueError("nvalchemi neighbor construction requires a CUDA device")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3):
        raise ValueError("cells must have shape [B, 3, 3]")
    if pbc.shape != (cells.shape[0], 3):
        raise ValueError("pbc must have shape [B, 3]")
    if ptr.shape != (cells.shape[0] + 1,):
        raise ValueError("ptr must have shape [B + 1]")
    if any(tensor.device != positions.device for tensor in (cells, pbc, ptr)):
        raise ValueError("positions, cells, pbc, and ptr must use the same device")
    if len(set(ids)) != len(ids):
        raise ValueError("system_ids must be unique")
    if any(value < 0 or value >= cells.shape[0] for value in ids):
        raise IndexError("system id outside the batch")
    return ids


def nvalchemi_neighbor_blocks(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    ptr: torch.Tensor,
    system_ids: Sequence[int],
    *,
    cutoff: float,
    method: str | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Build canonical graph blocks with NVIDIA's optional Warp dispatcher."""

    ids = _validate_inputs(positions, cells, pbc, ptr, system_ids, cutoff)
    if not ids:
        return {}

    counts = ptr[1:] - ptr[:-1]
    selected_graph_ids = torch.as_tensor(ids, device=positions.device, dtype=torch.long)
    selected_counts = counts[selected_graph_ids]
    selected_atom_ids = torch.cat(
        [
            torch.arange(
                ptr[system_id],
                ptr[system_id + 1],
                device=positions.device,
                dtype=torch.long,
            )
            for system_id in ids
        ]
    )
    local_ptr = torch.cat(
        (
            torch.zeros(1, device=positions.device, dtype=torch.int32),
            selected_counts.to(torch.int32).cumsum(dim=0, dtype=torch.int32),
        )
    )
    local_batch_idx = torch.repeat_interleave(
        torch.arange(len(ids), device=positions.device, dtype=torch.int32),
        selected_counts,
    )

    neighbor_list = _load_neighbor_api()
    edges, _, shifts = neighbor_list(
        positions=positions[selected_atom_ids].contiguous(),
        cutoff=cutoff,
        cell=cells[selected_graph_ids].contiguous(),
        pbc=pbc[selected_graph_ids].contiguous(),
        batch_idx=local_batch_idx,
        batch_ptr=local_ptr,
        return_neighbor_list=True,
        method=method,
    )
    edges = edges.to(dtype=torch.long)
    shifts = shifts.to(dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise RuntimeError("nvalchemiops returned an invalid edge tensor")
    if shifts.shape != (edges.shape[1], 3):
        raise RuntimeError("nvalchemiops returned invalid integer shifts")

    empty_edge = torch.empty((2, 0), device=positions.device, dtype=torch.long)
    empty_shift = torch.empty((0, 3), device=positions.device, dtype=torch.long)
    if edges.shape[1] == 0:
        return {
            system_id: (empty_edge.clone(), empty_shift.clone())
            for system_id in ids
        }

    center_owner = local_batch_idx[edges[0]].to(torch.long)
    if not torch.equal(center_owner, local_batch_idx[edges[1]].to(torch.long)):
        raise RuntimeError("nvalchemiops returned a cross-system edge")
    global_edges = selected_atom_ids[edges]
    global_owners = selected_graph_ids[center_owner]
    order = _canonical_order(global_edges, shifts, n_atoms=positions.shape[0])
    global_edges = global_edges[:, order]
    shifts = shifts[order]
    global_owners = global_owners[order]

    return {
        system_id: (
            global_edges[:, global_owners == system_id],
            shifts[global_owners == system_id],
        )
        for system_id in ids
    }
