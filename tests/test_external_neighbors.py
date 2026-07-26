from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from ase import Atoms

from batch_mlip.core.external_neighbors import nvalchemi_neighbor_blocks
from batch_mlip.core.neighbors import neighbor_list, resolve_neighbor_backend


def _reference(
    atoms: Atoms,
    *,
    cutoff: float,
    atom_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    center, neighbor, shifts = neighbor_list("ijS", atoms, cutoff)
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], neighbor, center))
    edges = np.stack((center[order], neighbor[order])).astype(np.int64)
    edges += atom_offset
    return torch.as_tensor(edges), torch.as_tensor(shifts[order], dtype=torch.long)


def test_nvalchemi_backend_requires_cuda():
    with pytest.raises(ValueError, match="nvalchemi.*requires a CUDA"):
        resolve_neighbor_backend(
            "nvalchemi",
            device=torch.device("cpu"),
            counts=torch.tensor([4]),
            cutoff=4.0,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("nvalchemiops") is None,
    reason="nvalchemiops CUDA integration is optional",
)
def test_nvalchemi_matches_matscipy_for_selective_triclinic_batch():
    systems = [
        Atoms(
            "H2",
            positions=[[0.2, 0.3, 0.4], [9.7, 0.4, 0.5]],
            cell=[[10.0, 0.0, 0.0], [0.5, 9.0, 0.0], [0.2, 0.3, 8.0]],
            pbc=True,
        ),
        Atoms(
            "He3",
            positions=[[0.1, 0.2, 0.3], [2.2, 0.4, 0.5], [0.4, 2.3, 0.7]],
            cell=[[8.0, 0.0, 0.0], [-0.4, 9.0, 0.0], [0.3, -0.2, 10.0]],
            pbc=True,
        ),
        Atoms(
            "Li",
            positions=[[4.3, -2.1, 7.4]],
            cell=[[2.0, 0.0, 0.0], [0.1, 2.2, 0.0], [0.2, 0.1, 2.4]],
            pbc=True,
        ),
    ]
    device = torch.device("cuda")
    counts = np.asarray([len(atoms) for atoms in systems], dtype=np.int64)
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
        np.stack([atoms.pbc for atoms in systems]),
        device=device,
        dtype=torch.bool,
    )
    ptr = torch.as_tensor(
        np.concatenate(([0], np.cumsum(counts))),
        device=device,
        dtype=torch.long,
    )

    actual = nvalchemi_neighbor_blocks(
        positions,
        cells,
        pbc,
        ptr,
        [2, 0],
        cutoff=3.1,
    )

    assert list(actual) == [2, 0]
    for system_id in (2, 0):
        expected = _reference(
            systems[system_id],
            cutoff=3.1,
            atom_offset=int(ptr[system_id]),
        )
        torch.testing.assert_close(actual[system_id][0].cpu(), expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[system_id][1].cpu(), expected[1], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("nvalchemiops") is None,
    reason="nvalchemiops CUDA integration is optional",
)
def test_nvalchemi_naive_dispatch_accepts_multiple_systems():
    device = torch.device("cuda")
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 2,
        device=device,
    )
    cells = torch.eye(3, device=device).repeat(2, 1, 1) * 10.0
    pbc = torch.ones((2, 3), device=device, dtype=torch.bool)
    ptr = torch.tensor([0, 2, 4], device=device, dtype=torch.long)

    result = nvalchemi_neighbor_blocks(
        positions,
        cells,
        pbc,
        ptr,
        [0, 1],
        cutoff=2.0,
        method="naive",
    )

    assert [result[system_id][0].shape[1] for system_id in (0, 1)] == [2, 2]
