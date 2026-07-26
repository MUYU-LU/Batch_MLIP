from __future__ import annotations

import pytest
import torch
from ase import Atoms
from ase.constraints import FixAtoms

from batch_mlip.core import HeterogeneousResidentArena, SystemSelection
from batch_mlip.core.state import AseGraphBatch


def _systems() -> list[Atoms]:
    first = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    first.set_constraint(FixAtoms(indices=[0]))
    second = Atoms(
        "He3",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[7.0, 7.0, 7.0],
        pbc=True,
    )
    third = Atoms(
        "Li",
        positions=[[0.2, 0.3, 0.4]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    fourth = Atoms(
        "Be4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    return [first, second, third, fourth]


def _state(systems, *, build_neighbors: bool) -> AseGraphBatch:
    return AseGraphBatch.from_ase(
        systems,
        cutoff=2.5,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
        build_neighbors=build_neighbors,
    )


def test_arena_matches_compact_concatenation_and_preserves_cache_state():
    systems = _systems()
    pool = _state(systems, build_neighbors=False)
    resident = _state(systems[:3], build_neighbors=True)
    arena = HeterogeneousResidentArena(pool, resident_capacity=3)

    packed = arena.pack(
        [
            SystemSelection(resident, (1, 0)),
            SystemSelection(pool, (3,)),
        ]
    )
    expected = AseGraphBatch.concatenate(
        [
            resident.select_systems([1, 0], rebuild_neighbors=False),
            pool.select_systems([3], rebuild_neighbors=False),
        ]
    )

    assert packed.counts.tolist() == [3, 2, 4]
    assert [atoms.get_chemical_formula() for atoms in packed.templates] == [
        atoms.get_chemical_formula() for atoms in expected.templates
    ]
    for name in (
        "z",
        "positions",
        "cells",
        "pbc",
        "system_idx",
        "ptr",
        "masses",
        "fixed",
        "velocities",
        "edge_index",
        "shifts_int",
    ):
        torch.testing.assert_close(getattr(packed, name), getattr(expected, name))
    torch.testing.assert_close(
        packed._neighbor_reference_positions,
        expected._neighbor_reference_positions,
    )
    torch.testing.assert_close(
        packed._neighbor_reference_cells,
        expected._neighbor_reference_cells,
    )
    assert packed._neighbor_reference_valid.tolist() == [True, True, False]
    packed.assert_graph_integrity()


def test_arena_alternates_banks_and_reuses_work_storage():
    systems = _systems()
    pool = _state(systems, build_neighbors=False)
    arena = HeterogeneousResidentArena(pool, resident_capacity=2)

    first = arena.pack([SystemSelection(pool, (0, 1))])
    first_pointer = first.z.untyped_storage().data_ptr()
    work = arena.work_tensor(
        "optimizer_forces",
        (first.n_atoms, 3),
        dtype=torch.float64,
        zero=True,
    )
    assert not bool(work.any())
    second = arena.pack([SystemSelection(pool, (2, 3))])
    second_pointer = second.z.untyped_storage().data_ptr()
    third = arena.pack([SystemSelection(pool, (1, 0))])

    assert first_pointer != second_pointer
    assert third.z.untyped_storage().data_ptr() == first_pointer
    assert third.counts.tolist() == [3, 2]


def test_arena_capacity_covers_largest_resident_and_rejects_extra_system():
    systems = _systems()
    pool = _state(systems, build_neighbors=False)
    arena = HeterogeneousResidentArena(pool, resident_capacity=1)

    # Capacity is sized for the largest one-system resident set.
    packed = arena.pack([SystemSelection(pool, (3,))])
    assert packed.n_atoms == 4
    with pytest.raises(ValueError, match="system capacity"):
        arena.pack([SystemSelection(pool, (0, 1))])
