from __future__ import annotations

import torch
from ase import Atoms
from atombit_batch.toy_models import QuadraticWellModel

from atombit_batch import AseGraphBatch, BatchedPotential


def test_skin_avoids_unnecessary_rebuilds():
    state = AseGraphBatch.from_ase(
        [Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])],
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    potential = BatchedPotential(
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
