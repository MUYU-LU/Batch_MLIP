from __future__ import annotations

import torch
from ase import Atoms
from batch_mlip.toy_models import PairHarmonicModel, QuadraticWellModel

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator


def _pair_potential() -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        device="cpu",
        dtype=torch.float64,
    )


def _fresh_pair_evaluation(atoms: Atoms):
    calculator = _pair_potential()
    state = calculator.create_state([atoms])
    return calculator(state)


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


def test_skin_candidates_are_filtered_at_physical_cutoff_and_activate_safely():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
    calculator = AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    state = calculator.create_state([atoms])

    assert state.edge_index.shape[1] == 2
    assert state.as_model_data().edge_index.shape[1] == 0

    state.positions[0, 0] += 0.06
    state.positions[1, 0] -= 0.06
    evaluation = calculator(state)
    assert state.neighbor_rebuild_count == 1
    assert state.as_model_data().edge_index.shape[1] == 2

    current = atoms.copy()
    current.positions[:] = state.positions.numpy()
    reference_calculator = _pair_potential()
    reference_state = reference_calculator.create_state([current])
    reference = reference_calculator(reference_state)
    torch.testing.assert_close(
        state.as_model_data().edge_index, reference_state.edge_index
    )
    torch.testing.assert_close(
        state.as_model_data().shifts_int, reference_state.shifts_int
    )
    torch.testing.assert_close(evaluation.energy, reference.energy)
    torch.testing.assert_close(evaluation.forces, reference.forces)


def test_skin_physical_cutoff_matches_ase_for_float32_rounding_boundary():
    separation = [-4.3417268, -2.5232267, 3.2837074]
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], separation])
    cached = AseGraphBatch.from_ase(
        [atoms],
        cutoff=6.0,
        skin=0.5,
        device="cpu",
        dtype=torch.float32,
    )
    fresh = AseGraphBatch.from_ase(
        [atoms],
        cutoff=6.0,
        skin=0.0,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.linalg.vector_norm(cached.positions[1]).item() == 6.0
    assert cached.as_model_data().edge_index.shape[1] == 2
    torch.testing.assert_close(
        cached.as_model_data().edge_index, fresh.as_model_data().edge_index
    )


def test_variable_cell_cache_reuses_safe_affine_compression():
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.2], [0.31, 0.2, 0.2]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    state = calculator.create_state([atoms])
    assert state.as_model_data().edge_index.shape[1] == 0

    compressed = atoms.copy()
    compressed.set_cell([9.4, 10.0, 10.0], scale_atoms=True)
    state.positions[:] = torch.as_tensor(compressed.positions)
    state.cells[0] = torch.as_tensor(compressed.cell.array)

    assert not state.neighbor_list_needs_rebuild()
    evaluation = calculator(state)
    assert state.neighbor_rebuild_count == 1
    assert state.as_model_data().edge_index.shape[1] == 2

    reference = _fresh_pair_evaluation(compressed)
    torch.testing.assert_close(evaluation.energy, reference.energy)
    torch.testing.assert_close(evaluation.forces, reference.forces)


def test_variable_cell_cache_rebuilds_before_missing_pair_enters_cutoff():
    atoms = Atoms(
        "H2",
        scaled_positions=[[0.1, 0.2, 0.2], [0.35, 0.2, 0.2]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    calculator = AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    state = calculator.create_state([atoms])
    assert state.edge_index.shape[1] == 0

    compressed = atoms.copy()
    compressed.set_cell([7.6, 10.0, 10.0], scale_atoms=True)
    state.positions[:] = torch.as_tensor(compressed.positions)
    state.cells[0] = torch.as_tensor(compressed.cell.array)

    assert state.neighbor_list_needs_rebuild()
    evaluation = calculator(state)
    assert state.neighbor_rebuild_count == 2
    assert state.as_model_data().edge_index.shape[1] == 2

    reference = _fresh_pair_evaluation(compressed)
    torch.testing.assert_close(evaluation.energy, reference.energy)
    torch.testing.assert_close(evaluation.forces, reference.forces)


def test_heterogeneous_cache_rebuilds_only_invalid_system():
    systems = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        Atoms("H3", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.9, 0.0]]),
    ]
    state = AseGraphBatch.from_ase(
        systems,
        cutoff=2.0,
        skin=0.4,
        device="cpu",
        dtype=torch.float64,
    )
    second_slice = state.atom_slice(1)
    second_reference = state._neighbor_reference_positions[second_slice].clone()
    second_edges = state.edge_index[
        :, state.system_idx[state.edge_index[0]] == 1
    ].clone()

    state.positions[0, 0] += 0.21
    assert state.neighbor_list_invalid_systems().tolist() == [True, False]
    assert state.ensure_neighbor_list()
    assert state.neighbor_rebuild_count == 2
    torch.testing.assert_close(
        state._neighbor_reference_positions[second_slice], second_reference
    )
    torch.testing.assert_close(
        state.edge_index[:, state.system_idx[state.edge_index[0]] == 1],
        second_edges,
    )
