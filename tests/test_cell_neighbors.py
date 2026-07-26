from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from batch_mlip import AseGraphBatch
from batch_mlip.core.cell_neighbors import (
    CellListUnsupportedError,
    cell_list_neighbor_blocks,
    estimate_cell_candidate_reduction,
)
from batch_mlip.core.neighbors import neighbor_list, resolve_neighbor_backend

CPU_DEVICE = torch.device("cpu")


def _tensors(
    systems: list[Atoms],
    *,
    device: torch.device = CPU_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = np.asarray([len(atoms) for atoms in systems], dtype=np.int64)
    return (
        torch.as_tensor(
            np.concatenate([atoms.positions for atoms in systems]),
            device=device,
            dtype=torch.float64,
        ),
        torch.as_tensor(
            np.stack([atoms.cell.array for atoms in systems]),
            device=device,
            dtype=torch.float64,
        ),
        torch.as_tensor(
            np.stack([atoms.pbc for atoms in systems]),
            device=device,
            dtype=torch.bool,
        ),
        torch.as_tensor(
            np.concatenate(([0], np.cumsum(counts))),
            device=device,
            dtype=torch.long,
        ),
    )


def _reference(
    atoms: Atoms,
    *,
    cutoff: float,
    atom_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    center, neighbor, shifts = neighbor_list("ijS", atoms, cutoff)
    order = np.lexsort(
        (shifts[:, 2], shifts[:, 1], shifts[:, 0], neighbor, center)
    )
    edges = np.stack((center[order], neighbor[order])).astype(np.int64)
    edges += atom_offset
    return torch.as_tensor(edges), torch.as_tensor(shifts[order], dtype=torch.long)


def _assert_matches_reference(
    systems: list[Atoms],
    *,
    cutoff: float,
    device: torch.device = CPU_DEVICE,
) -> None:
    positions, cells, pbc, ptr = _tensors(systems, device=device)
    actual = cell_list_neighbor_blocks(
        positions,
        cells,
        pbc,
        ptr,
        range(len(systems)),
        cutoff=cutoff,
        max_work_bytes=16 * 1024**2,
    )
    offset = 0
    for system_id, atoms in enumerate(systems):
        expected_edges, expected_shifts = _reference(
            atoms,
            cutoff=cutoff,
            atom_offset=offset,
        )
        torch.testing.assert_close(
            actual[system_id][0].cpu(),
            expected_edges,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual[system_id][1].cpu(),
            expected_shifts,
            rtol=0,
            atol=0,
        )
        offset += len(atoms)


def test_cell_list_matches_randomized_triclinic_unwrapped_systems():
    rng = np.random.default_rng(20260726)
    systems = []
    for _ in range(18):
        diagonal = rng.uniform(5.0, 14.0, size=3)
        cell = np.diag(diagonal)
        cell[1, 0] = rng.uniform(-1.5, 1.5)
        cell[2, :2] = rng.uniform(-1.5, 1.5, size=2)
        fractional = rng.uniform(-2.0, 3.0, size=(int(rng.integers(1, 10)), 3))
        systems.append(
            Atoms(
                "H" * len(fractional),
                positions=fractional @ cell,
                cell=cell,
                pbc=True,
            )
        )

    _assert_matches_reference(systems, cutoff=4.7)


def test_cell_list_matches_multiple_images_in_small_cells():
    systems = [
        Atoms(
            "H2",
            positions=[[0.1, 0.2, 0.3], [1.1, 0.8, 0.6]],
            cell=[[2.0, 0.0, 0.0], [0.2, 2.2, 0.0], [0.1, 0.2, 2.4]],
            pbc=True,
        ),
        Atoms(
            "He",
            positions=[[5.3, -2.1, 4.7]],
            cell=[[1.8, 0.0, 0.0], [0.1, 2.0, 0.0], [0.0, 0.2, 2.1]],
            pbc=True,
        ),
    ]

    _assert_matches_reference(systems, cutoff=4.1)


def test_cell_list_selective_heterogeneous_output_order():
    systems = [
        Atoms("H", positions=[[0.2, 0.3, 0.4]], cell=[6.0, 7.0, 8.0], pbc=True),
        Atoms(
            "He2",
            positions=[[0.1, 0.2, 0.3], [2.2, 0.4, 0.5]],
            cell=[9.0, 8.0, 7.0],
            pbc=True,
        ),
        Atoms(
            "Li3",
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]],
            cell=[10.0, 9.0, 8.0],
            pbc=True,
        ),
    ]
    positions, cells, pbc, ptr = _tensors(systems)

    actual = cell_list_neighbor_blocks(
        positions,
        cells,
        pbc,
        ptr,
        [2, 0],
        cutoff=3.2,
    )

    assert list(actual) == [2, 0]
    for system_id in [2, 0]:
        expected = _reference(
            systems[system_id],
            cutoff=3.2,
            atom_offset=int(ptr[system_id]),
        )
        torch.testing.assert_close(actual[system_id][0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[system_id][1], expected[1], rtol=0, atol=0)


def test_cell_list_rejects_partial_periodicity():
    systems = [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[5.0, 6.0, 7.0],
            pbc=[True, True, False],
        )
    ]
    positions, cells, pbc, ptr = _tensors(systems)

    with pytest.raises(CellListUnsupportedError, match="three-dimensional periodic"):
        cell_list_neighbor_blocks(
            positions,
            cells,
            pbc,
            ptr,
            [0],
            cutoff=3.0,
        )


def test_cuda_cell_backend_requires_cuda():
    with pytest.raises(ValueError, match="cuda_cell.*requires a CUDA"):
        resolve_neighbor_backend(
            "cuda_cell",
            device=torch.device("cpu"),
            counts=torch.tensor([4]),
            cutoff=4.0,
        )


def test_auto_cell_policy_requires_large_predicted_candidate_reduction():
    device = torch.device("cuda")
    counts = torch.full((8,), 368)
    large_cells = torch.eye(3).repeat(8, 1, 1) * 24.0
    small_cells = torch.eye(3).repeat(8, 1, 1) * 10.0
    pbc = torch.ones((8, 3), dtype=torch.bool)

    assert estimate_cell_candidate_reduction(
        large_cells,
        pbc,
        cutoff=6.0,
    ) >= 0.98
    assert (
        resolve_neighbor_backend(
            "auto",
            device=device,
            counts=counts,
            cutoff=6.0,
            cells=large_cells,
            pbc=pbc,
        )
        == "cuda_cell"
    )
    assert (
        resolve_neighbor_backend(
            "auto",
            device=device,
            counts=counts,
            cutoff=6.0,
            cells=small_cells,
            pbc=pbc,
        )
        == "cuda_dense"
    )
    pbc[0, 2] = False
    assert (
        resolve_neighbor_backend(
            "auto",
            device=device,
            counts=counts,
            cutoff=6.0,
            cells=large_cells,
            pbc=pbc,
        )
        == "cuda_dense"
    )


def test_cell_policy_uses_occupied_fractional_span():
    cells = torch.eye(3).reshape(1, 3, 3) * 24.0
    pbc = torch.ones((1, 3), dtype=torch.bool)
    counts = torch.tensor([4])
    compact = torch.tensor(
        [
            [10.0, 10.0, 10.0],
            [10.2, 10.0, 10.0],
            [10.0, 10.2, 10.0],
            [10.0, 10.0, 10.2],
        ]
    )
    spanning = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [23.0, 0.1, 0.1],
            [0.1, 23.0, 0.1],
            [0.1, 0.1, 23.0],
        ]
    )

    compact_reduction = estimate_cell_candidate_reduction(
        cells,
        pbc,
        cutoff=6.0,
        positions=compact,
        counts=counts,
    )
    spanning_reduction = estimate_cell_candidate_reduction(
        cells,
        pbc,
        cutoff=6.0,
        positions=spanning,
        counts=counts,
    )

    assert compact_reduction < spanning_reduction


def test_auto_cell_work_guard_falls_back_to_dense(monkeypatch):
    atoms = Atoms(
        "H",
        positions=[[0.1, 0.2, 0.3]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    state = AseGraphBatch.from_ase(
        [atoms],
        cutoff=1.0,
        device="cpu",
        neighbor_backend="auto",
        build_neighbors=False,
    )

    monkeypatch.setattr(
        "batch_mlip.core.state.resolve_neighbor_backend",
        lambda *args, **kwargs: "cuda_cell",
    )

    def reject_cell(*args, **kwargs):
        raise CellListUnsupportedError("synthetic work guard")

    monkeypatch.setattr(
        "batch_mlip.core.state.cell_list_neighbor_blocks",
        reject_cell,
    )

    state.rebuild_neighbor_list()

    assert state.neighbor_rebuild_count == 1
    assert state.edge_index.shape == (2, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_cell_state_matches_matscipy_exactly():
    systems = [
        Atoms(
            "H4",
            positions=[
                [-1.0, 0.2, 0.3],
                [1.2, 1.4, 0.5],
                [5.4, 2.1, 3.3],
                [9.7, -1.3, 4.2],
            ],
            cell=[[7.0, 0.0, 0.0], [0.8, 8.0, 0.0], [0.3, 0.4, 9.0]],
            pbc=True,
        ),
        Atoms(
            "He3",
            positions=[[0.1, 0.2, 0.3], [3.4, 0.8, 1.2], [6.8, 4.3, 2.2]],
            cell=[[8.0, 0.0, 0.0], [-0.4, 7.5, 0.0], [0.2, 0.6, 8.5]],
            pbc=True,
        ),
    ]
    reference = AseGraphBatch.from_ase(
        systems,
        cutoff=4.7,
        device="cuda",
        dtype=torch.float64,
        neighbor_backend="matscipy",
    )
    actual = AseGraphBatch.from_ase(
        systems,
        cutoff=4.7,
        device="cuda",
        dtype=torch.float64,
        neighbor_backend="cuda_cell",
    )

    torch.testing.assert_close(actual.edge_index, reference.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(actual.shifts_int, reference.shifts_int, rtol=0, atol=0)

    actual.positions[:4] += torch.tensor([0.13, -0.07, 0.04], device="cuda")
    reference.positions[:4] += torch.tensor([0.13, -0.07, 0.04], device="cuda")
    actual.cells[0] *= 0.94
    reference.cells[0] *= 0.94
    actual.rebuild_neighbor_list([0])
    reference.rebuild_neighbor_list([0])
    torch.testing.assert_close(actual.edge_index, reference.edge_index, rtol=0, atol=0)
    torch.testing.assert_close(actual.shifts_int, reference.shifts_int, rtol=0, atol=0)
