"""Neighbor-list construction for heterogeneous graph batches."""

from __future__ import annotations

from dataclasses import dataclass
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
AUTO_WARM_SMALL_EDGE_THRESHOLD = 10_000
AUTO_WARM_DENSE_EDGE_THRESHOLD = 35_000
AUTO_WARM_CELL_EDGE_DENSITY_THRESHOLD = 62.0
AUTO_WARM_CELL_PAIR_RATIO_THRESHOLD = 1.3
AUTO_WARM_CELL_VOLUME_PER_ATOM_THRESHOLD = 10.5
AUTO_WARM_CELL_MAX_SMALL_SYSTEMS = 3

ResolvedNeighborBackend = Literal["matscipy", "cuda_dense", "cuda_cell", "nvalchemi"]


@dataclass(frozen=True)
class NeighborBackendDecision:
    """Observable result of one neighbor-backend selection."""

    backend: ResolvedNeighborBackend
    reason: str
    system_count: int
    pair_work: int | None = None
    candidate_reduction: float | None = None
    candidate_edges: int | None = None
    mean_volume_per_atom: float | None = None


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
    candidate_edges: int | None = None,
    mean_volume_per_atom: float | None = None,
) -> ResolvedNeighborBackend:
    """Resolve the requested backend for one rebuild operation."""

    return resolve_neighbor_backend_decision(
        backend,
        device=device,
        counts=counts,
        cutoff=cutoff,
        cells=cells,
        pbc=pbc,
        positions=positions,
        candidate_edges=candidate_edges,
        mean_volume_per_atom=mean_volume_per_atom,
    ).backend


def resolve_neighbor_backend_decision(
    backend: NeighborBackend,
    *,
    device: torch.device,
    counts: torch.Tensor,
    cutoff: float,
    cells: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    candidate_edges: int | None = None,
    mean_volume_per_atom: float | None = None,
) -> NeighborBackendDecision:
    """Resolve a backend and expose the inputs that caused the decision."""

    system_count = counts.numel()

    if backend == "matscipy":
        return NeighborBackendDecision("matscipy", "explicit backend", system_count)
    if backend == "cuda_dense":
        if device.type != "cuda":
            raise ValueError("cuda_dense neighbor construction requires a CUDA device")
        return NeighborBackendDecision("cuda_dense", "explicit backend", system_count)
    if backend == "cuda_cell":
        if device.type != "cuda":
            raise ValueError("cuda_cell neighbor construction requires a CUDA device")
        return NeighborBackendDecision("cuda_cell", "explicit backend", system_count)
    if backend == "nvalchemi":
        if device.type != "cuda":
            raise ValueError("nvalchemi neighbor construction requires a CUDA device")
        return NeighborBackendDecision("nvalchemi", "explicit backend", system_count)
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    if candidate_edges is not None and candidate_edges < 0:
        raise ValueError("candidate_edges must be non-negative")
    if mean_volume_per_atom is not None and mean_volume_per_atom <= 0.0:
        raise ValueError("mean_volume_per_atom must be positive")
    pair_work = int(torch.sum(counts.to(torch.int64) ** 2).item())
    if (
        device.type == "cuda"
        and candidate_edges is not None
        and mean_volume_per_atom is not None
    ):
        atom_count = int(counts.sum().item())
        edges_per_atom = candidate_edges / atom_count
        edges_per_pair_work = candidate_edges / pair_work
        if candidate_edges <= AUTO_WARM_SMALL_EDGE_THRESHOLD:
            resolved = (
                "cuda_cell"
                if mean_volume_per_atom <= AUTO_WARM_CELL_VOLUME_PER_ATOM_THRESHOLD
                else "matscipy"
            )
        elif candidate_edges <= AUTO_WARM_DENSE_EDGE_THRESHOLD:
            resolved = (
                "cuda_cell"
                if (
                    edges_per_atom > AUTO_WARM_CELL_EDGE_DENSITY_THRESHOLD
                    or system_count <= AUTO_WARM_CELL_MAX_SMALL_SYSTEMS
                )
                else "cuda_dense"
            )
        else:
            resolved = (
                "cuda_cell"
                if edges_per_pair_work > AUTO_WARM_CELL_PAIR_RATIO_THRESHOLD
                else "cuda_dense"
            )
        return NeighborBackendDecision(
            resolved,
            "cached candidate-graph policy",
            system_count,
            pair_work,
            candidate_edges=candidate_edges,
            mean_volume_per_atom=mean_volume_per_atom,
        )
    minimum_systems = 2 if cutoff >= 5.5 else 4
    pair_threshold = (
        AUTO_CUDA_DENSE_LONG_CUTOFF_PAIR_THRESHOLD
        if cutoff >= 5.5
        else AUTO_CUDA_DENSE_SHORT_CUTOFF_PAIR_THRESHOLD
    )
    if device.type == "cuda" and counts.numel() >= minimum_systems and pair_work >= pair_threshold:
        reduction = None
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
                return NeighborBackendDecision(
                    "cuda_cell",
                    "predicted sparse candidate reduction passed",
                    system_count,
                    pair_work,
                    reduction,
                )
        return NeighborBackendDecision(
            "cuda_dense",
            "CUDA pair-work threshold passed",
            system_count,
            pair_work,
            reduction,
        )
    reason = (
        "non-CUDA device"
        if device.type != "cuda"
        else "CUDA pair-work threshold not reached"
    )
    return NeighborBackendDecision(
        "matscipy",
        reason,
        system_count,
        pair_work,
    )


def neighbor_list(quantities: str, atoms: Any, cutoff: float, *args, **kwargs):
    """ASE-compatible neighbour-list wrapper."""

    # Matscipy 1.2 mishandles unwrapped coordinates for partial/nonperiodic
    # rank-3 cells, including shifts along nonperiodic axes. Keep it on its
    # fast, validated fully-periodic path and use ASE for all other cells.
    if _matscipy_neighbor_list is not None and atoms.cell.rank == 3 and bool(atoms.pbc.all()):
        return _matscipy_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
    return _ase_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
