from __future__ import annotations

import torch
from ase import Atoms
from batch_mlip.toy_models import QuadraticWellModel

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator


def test_skin_avoids_unnecessary_rebuilds():
    state = AseGraphBatch.from_ase(
        [Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])],
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(), device="cpu", dtype=torch.float64
    )
    assert state.neighbor_rebuild_count == 1
    potential(state)
    assert state.neighbor_rebuild_count == 1

    state.positions[0, 0] += 0.1
    potential(state)
    assert state.neighbor_rebuild_count == 1

    state.positions[0, 0] += 0.11
    potential(state)
    assert state.neighbor_rebuild_count == 2


def test_neighbor_construction_can_be_deferred_until_evaluation():
    state = AseGraphBatch.from_ase(
        [Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])],
        cutoff=2.0,
        device="cpu",
        dtype=torch.float64,
        build_neighbors=False,
    )
    potential = AtomBitBatchCalculator(
        QuadraticWellModel(), device="cpu", dtype=torch.float64
    )

    assert state.neighbor_rebuild_count == 0
    assert state.edge_index.shape == (2, 0)
    potential(state)
    assert state.neighbor_rebuild_count == 1
    assert state.edge_index.shape[1] > 0
