"""Neighbor-list construction for heterogeneous graph batches."""

from __future__ import annotations

from typing import Any, Literal

import torch
from ase.neighborlist import neighbor_list as _ase_neighbor_list

from .cell_neighbors import estimate_cell_candidate_reduction

try:  # matscipy is substantially faster for full-rank cells.
    from matscipy.neighbours import neighbour_list as _matscipy_neighbor_list

    BACKEND = "matscipy"
except ImportError:  # pragma: no cover - environment dependent
    _matscipy_neighbor_list = None
    BACKEND = "ase"

NeighborBackend = Literal[
    "auto",
    "matscipy",
    "cuda_dense",
    "cuda_cell",
    "nvalchemi",
]
AUTO_CUDA_DENSE_LONG_CUTOFF_PAIR_THRESHOLD = 8192
AUTO_CUDA_DENSE_SHORT_CUTOFF_PAIR_THRESHOLD = 32768
AUTO_CUDA_CELL_CANDIDATE_REDUCTION_THRESHOLD = 0.98


def validate_neighbor_backend(backend: str) -> NeighborBackend:
    """Validate and narrow a public neighbor backend name."""

    if backend not in ("auto", "matscipy", "cuda_dense", "cuda_cell", "nvalchemi"):
        raise ValueError(
            "neighbor_backend must be 'auto', 'matscipy', 'cuda_dense', "
            "'cuda_cell', or 'nvalchemi'"
        )
    return backend


def resolve_neighbor_backend(
    backend: NeighborBackend,
    *,
    device: torch.device,
    counts: torch.Tensor,
    cutoff: float,
    cells: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> Literal["matscipy", "cuda_dense", "cuda_cell", "nvalchemi"]:
    """Resolve the requested backend for one rebuild operation."""

    if backend == "matscipy":
        return "matscipy"
    if backend == "cuda_dense":
        if device.type != "cuda":
            raise ValueError("cuda_dense neighbor construction requires a CUDA device")
        return "cuda_dense"
    if backend == "cuda_cell":
        if device.type != "cuda":
            raise ValueError("cuda_cell neighbor construction requires a CUDA device")
        return "cuda_cell"
    if backend == "nvalchemi":
        if device.type != "cuda":
            raise ValueError("nvalchemi neighbor construction requires a CUDA device")
        return "nvalchemi"
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    pair_work = int(torch.sum(counts.to(torch.int64) ** 2).item())
    minimum_systems = 2 if cutoff >= 5.5 else 4
    pair_threshold = (
        AUTO_CUDA_DENSE_LONG_CUTOFF_PAIR_THRESHOLD
        if cutoff >= 5.5
        else AUTO_CUDA_DENSE_SHORT_CUTOFF_PAIR_THRESHOLD
    )
    if device.type == "cuda" and counts.numel() >= minimum_systems and pair_work >= pair_threshold:
        if cells is not None and pbc is not None:
            span_inputs = (
                {}
                if positions is None
                else {"positions": positions, "counts": counts}
            )
            reduction = estimate_cell_candidate_reduction(
                cells,
                pbc,
                cutoff=cutoff,
                **span_inputs,
            )
            if (
                reduction is not None
                and reduction >= AUTO_CUDA_CELL_CANDIDATE_REDUCTION_THRESHOLD
            ):
                return "cuda_cell"
        return "cuda_dense"
    return "matscipy"


def neighbor_list(quantities: str, atoms: Any, cutoff: float, *args, **kwargs):
    """ASE-compatible neighbour-list wrapper."""

    # Matscipy 1.2 mishandles unwrapped coordinates for partial/nonperiodic
    # rank-3 cells, including shifts along nonperiodic axes. Keep it on its
    # fast, validated fully-periodic path and use ASE for all other cells.
    if _matscipy_neighbor_list is not None and atoms.cell.rank == 3 and bool(atoms.pbc.all()):
        return _matscipy_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
    return _ase_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
