"""Neighbor-list construction for heterogeneous graph batches."""

from __future__ import annotations

from typing import Any

from ase.neighborlist import neighbor_list as _ase_neighbor_list

try:  # matscipy is substantially faster for full-rank cells.
    from matscipy.neighbours import neighbour_list as _matscipy_neighbor_list

    BACKEND = "matscipy"
except ImportError:  # pragma: no cover - environment dependent
    _matscipy_neighbor_list = None
    BACKEND = "ase"


def neighbor_list(quantities: str, atoms: Any, cutoff: float, *args, **kwargs):
    """ASE-compatible neighbour-list wrapper."""

    # matscipy 1.2 attempts to invert the cell even for nonperiodic atoms and
    # therefore fails on ASE's conventional zero cell.
    if _matscipy_neighbor_list is not None and atoms.cell.rank == 3:
        return _matscipy_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
    return _ase_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
