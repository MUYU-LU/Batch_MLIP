from __future__ import annotations

import torch
from ase import Atoms
from atombit_batch.toy_models import QuadraticWellModel

from atombit_batch import AseGraphBatch, BatchedPotential


def make_systems():
    return [
        Atoms("H2", positions=[[0.1, 0.2, 0.3], [1.0, -0.2, 0.1]]),
        Atoms(
            "H3",
            positions=[[0.4, 0.0, 0.0], [0.0, 0.7, 0.0], [0.1, 0.2, 0.9]],
        ),
    ]


def test_batch_matches_single_system_evaluations():
    systems = make_systems()
    model = QuadraticWellModel(k=1.7)
    potential = BatchedPotential(model, device="cpu", dtype=torch.float64)

    batch = AseGraphBatch.from_ase(
        systems, cutoff=3.0, device="cpu", dtype=torch.float64
    )
    batched = potential(batch)

    energies = []
    forces = []
    for atoms in systems:
        singleton = AseGraphBatch.from_ase(
            [atoms], cutoff=3.0, device="cpu", dtype=torch.float64
        )
        value = potential(singleton)
        energies.append(value.energy)
        forces.append(value.forces)

    torch.testing.assert_close(batched.energy, torch.cat(energies), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(batched.forces, torch.cat(forces), atol=1e-12, rtol=1e-12)


def test_neighbor_edges_never_cross_graphs():
    batch = AseGraphBatch.from_ase(
        make_systems(), cutoff=3.0, device="cpu", dtype=torch.float64
    )
    center, neighbor = batch.edge_index
    assert torch.equal(batch.system_idx[center], batch.system_idx[neighbor])
