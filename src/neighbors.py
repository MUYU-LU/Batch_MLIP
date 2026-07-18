from __future__ import annotations

from typing import Any

from ase.neighborlist import neighbor_list as _ase_neighbor_list

try:
    from matscipy.neighbours import neighbour_list as _matscipy_neighbor_list
except ImportError:  # pragma: no cover
    _matscipy_neighbor_list = None


def neighbor_list(quantities: str, atoms: Any, cutoff: float, *args, **kwargs):
    """ASE-compatible neighbor list wrapper backed by matscipy when available."""

    if _matscipy_neighbor_list is not None and atoms.cell.rank == 3:
        return _matscipy_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
    return _ase_neighbor_list(quantities, atoms, cutoff, *args, **kwargs)
