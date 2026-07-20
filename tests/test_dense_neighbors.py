from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from batch_mlip import AseGraphBatch
from batch_mlip.core.dense_neighbors import dense_neighbor_blocks
from batch_mlip.core.neighbors import neighbor_list, resolve_neighbor_backend


def _systems() -> list[Atoms]:
    return [
        Atoms(
            "H3",
            positions=[[0.2, 0.3, 0.4], [1.4, 0.7, 0.9], [5.7, 5.2, 4.8]],
            cell=[[6.0, 0.0, 0.0], [0.7, 6.5, 0.0], [0.2, 0.4, 7.0]],
            pbc=True,
        ),
        Atoms(
            "H3",
            positions=[[-3.1, 0.1, 0.0], [1.2, 0.4, 0.0], [7.4, 1.1, 0.0]],
            cell=[[3.0, 0.0, 0.0], [0.2, 5.0, 0.0], [0.0, 0.0, 0.0]],
            pbc=[True, False, False],
        ),
        Atoms(
            "H4",
            positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0], [4.0, 0.0, 0.0]],
            pbc=False,
        ),
        Atoms(
            "H2",
            positions=[[0.1, 0.2, 0.3], [1.7, 1.2, 0.8]],
            cell=[[2.2, 0.0, 0.0], [0.3, 2.4, 0.0], [0.2, 0.1, 2.6]],
            pbc=True,
        ),
    ]


def _reference(atoms: Atoms, cutoff: float, offset: int) -> tuple[np.ndarray, np.ndarray]:
    center, neighbor, shifts = neighbor_list("ijS", atoms, cutoff)
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], neighbor, center))
    edges = np.stack((center[order], neighbor[order])).astype(np.int64, copy=False)
    edges += offset
    return edges, np.asarray(shifts[order], dtype=np.int64)


def _dense_for_systems(
    systems: list[Atoms], cutoff: float, device: torch.device
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    counts = np.asarray([len(atoms) for atoms in systems], dtype=np.int64)
    ptr = torch.as_tensor(np.concatenate(([0], np.cumsum(counts))), device=device, dtype=torch.long)
    positions = torch.as_tensor(
        np.concatenate([atoms.positions for atoms in systems]),
        device=device,
        dtype=torch.float64,
    )
    cells = torch.as_tensor(
        np.stack([atoms.cell.array for atoms in systems]),
        device=device,
        dtype=torch.float64,
    )
    pbc = torch.as_tensor(
        np.stack([atoms.pbc for atoms in systems]), device=device, dtype=torch.bool
    )
    return dense_neighbor_blocks(
        positions,
        cells,
        pbc,
        ptr,
        range(len(systems)),
        cutoff=cutoff,
        max_work_bytes=32 * 1024**2,
    )


def test_dense_builder_matches_cpu_for_general_cells_and_ordering():
    systems = _systems()
    cutoff = 4.7
    actual = _dense_for_systems(systems, cutoff, torch.device("cpu"))

    offset = 0
    for system_id, atoms in enumerate(systems):
        expected_edges, expected_shifts = _reference(atoms, cutoff, offset)
        torch.testing.assert_close(
            actual[system_id][0], torch.as_tensor(expected_edges), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[system_id][1], torch.as_tensor(expected_shifts), rtol=0, atol=0
        )
        offset += len(atoms)


def test_dense_builder_matches_randomized_triclinic_and_partial_pbc():
    rng = np.random.default_rng(20260720)
    systems = []
    pbc_patterns = [
        [False, False, False],
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    for index in range(20):
        atom_count = int(rng.integers(1, 7))
        diagonal = rng.uniform(2.2, 7.5, size=3)
        cell = np.diag(diagonal)
        cell[1, 0] = rng.uniform(-0.8, 0.8)
        cell[2, :2] = rng.uniform(-0.8, 0.8, size=2)
        periodic = np.asarray(pbc_patterns[index % len(pbc_patterns)])
        if not periodic[2] and index % 3 == 0:
            cell[2] = 0.0
        fractional = rng.uniform(-1.5, 2.5, size=(atom_count, 3))
        positions = fractional @ cell
        if not periodic.any():
            positions = rng.uniform(-4.0, 4.0, size=(atom_count, 3))
        systems.append(
            Atoms(
                "H" * atom_count,
                positions=positions,
                cell=cell,
                pbc=periodic,
            )
        )

    cutoff = 4.1
    actual = _dense_for_systems(systems, cutoff, torch.device("cpu"))
    offset = 0
    for system_id, atoms in enumerate(systems):
        expected_edges, expected_shifts = _reference(atoms, cutoff, offset)
        assert np.array_equal(actual[system_id][0].numpy(), expected_edges)
        assert np.array_equal(actual[system_id][1].numpy(), expected_shifts)
        offset += len(atoms)


def test_dense_builder_ignores_dependent_nonperiodic_cell_rows():
    atoms = Atoms(
        "H3",
        positions=[[-2.8, 0.2, 0.1], [1.4, 0.7, 0.2], [6.2, 2.3, 1.1]],
        cell=[[3.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=[True, False, False],
    )
    cutoff = 4.1
    actual_edges, actual_shifts = _dense_for_systems([atoms], cutoff, torch.device("cpu"))[0]

    expected = []
    for center in range(len(atoms)):
        for neighbor in range(len(atoms)):
            for shift_x in range(-4, 5):
                if center == neighbor and shift_x == 0:
                    continue
                delta = (
                    atoms.positions[neighbor] - atoms.positions[center] + shift_x * atoms.cell[0]
                )
                if np.dot(delta, delta) < cutoff**2:
                    expected.append((center, neighbor, shift_x, 0, 0))
    expected.sort()
    expected_edges = torch.tensor([[row[0], row[1]] for row in expected]).T
    expected_shifts = torch.tensor([row[2:] for row in expected])

    torch.testing.assert_close(actual_edges, expected_edges, rtol=0, atol=0)
    torch.testing.assert_close(actual_shifts, expected_shifts, rtol=0, atol=0)


def test_dense_builder_supports_selective_empty_rebuild():
    ptr = torch.tensor([0, 1, 3])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    cells = torch.zeros((2, 3, 3))
    pbc = torch.zeros((2, 3), dtype=torch.bool)

    actual = dense_neighbor_blocks(positions, cells, pbc, ptr, [1], cutoff=1.0)

    assert list(actual) == [1]
    assert actual[1][0].shape == (2, 0)
    assert actual[1][1].shape == (0, 3)


def test_neighbor_backend_validation_and_cpu_auto_resolution():
    systems = _systems()
    state = AseGraphBatch.from_ase(
        systems,
        cutoff=4.7,
        device="cpu",
        dtype=torch.float64,
        neighbor_backend="auto",
    )

    assert state.neighbor_backend == "auto"
    assert (
        resolve_neighbor_backend("auto", device=state.device, counts=state.counts, cutoff=4.7)
        == "matscipy"
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_neighbor_backend("cuda_dense", device=state.device, counts=state.counts, cutoff=4.7)
    with pytest.raises(ValueError, match="neighbor_backend"):
        AseGraphBatch.from_ase(systems, cutoff=4.7, neighbor_backend="invalid")

    cuda = torch.device("cuda")
    assert (
        resolve_neighbor_backend("auto", device=cuda, counts=torch.full((8,), 46), cutoff=4.5)
        == "matscipy"
    )
    assert (
        resolve_neighbor_backend("auto", device=cuda, counts=torch.full((16,), 46), cutoff=4.5)
        == "cuda_dense"
    )
    assert (
        resolve_neighbor_backend("auto", device=cuda, counts=torch.full((2,), 276), cutoff=6.0)
        == "cuda_dense"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_dense_state_matches_cpu_backend_exactly():
    systems = _systems()
    cutoff = 4.7
    cpu = AseGraphBatch.from_ase(
        systems,
        cutoff=cutoff,
        device="cuda",
        dtype=torch.float64,
        neighbor_backend="matscipy",
    )
    dense = AseGraphBatch.from_ase(
        systems,
        cutoff=cutoff,
        device="cuda",
        dtype=torch.float64,
        neighbor_backend="cuda_dense",
    )

    torch.testing.assert_close(dense.edge_index, cpu.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(dense.shifts_int, cpu.shifts_int, rtol=0, atol=0)

    atom_slice = dense.atom_slice(1)
    displacement = torch.tensor([0.17, -0.08, 0.03], device="cuda")
    dense.positions[atom_slice] += displacement
    cpu.positions[atom_slice] += displacement
    dense.cells[1] *= 0.93
    cpu.cells[1] *= 0.93
    dense.rebuild_neighbor_list([1])
    cpu.rebuild_neighbor_list([1])

    torch.testing.assert_close(dense.edge_index, cpu.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(dense.shifts_int, cpu.shifts_int, rtol=0, atol=0)
