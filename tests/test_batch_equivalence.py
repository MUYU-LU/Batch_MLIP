from __future__ import annotations

import torch
from ase import Atoms
from batch_mlip.toy_models import QuadraticWellModel

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator


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
    potential = AtomBitBatchCalculator(model, device="cpu", dtype=torch.float64)

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


def test_float32_model_accumulates_e0_in_float64():
    systems = [Atoms("H46", positions=torch.zeros(46, 3).numpy())]
    state = AseGraphBatch.from_ase(
        systems, cutoff=3.0, device="cpu", dtype=torch.float32, build_neighbors=False
    )
    residual = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.7), device="cpu", dtype=torch.float32, cutoff=3.0
    )(state, neighbor_policy="never")
    e0_value = -13.605693122994
    total = AtomBitBatchCalculator(
        QuadraticWellModel(k=1.7),
        device="cpu",
        dtype=torch.float32,
        cutoff=3.0,
        e0_dict={1: e0_value},
    )(state, neighbor_policy="never")

    assert total.energy.dtype == torch.float64
    expected = residual.energy.to(torch.float64) + 46 * e0_value
    torch.testing.assert_close(total.energy, expected, rtol=0, atol=1e-12)


def test_neighbor_edges_never_cross_graphs():
    batch = AseGraphBatch.from_ase(
        make_systems(), cutoff=3.0, device="cpu", dtype=torch.float64
    )
    center, neighbor = batch.edge_index
    assert torch.equal(batch.system_idx[center], batch.system_idx[neighbor])
