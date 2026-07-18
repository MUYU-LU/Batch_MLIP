from __future__ import annotations

import torch
from ase import Atoms

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator
from src.model import AtomBitModel
from src.utils import AtomBitConfig


def test_uploaded_model_namespace_runs_as_a_real_batch():
    config = AtomBitConfig(
        cutoff=3.0,
        num_rbf=4,
        hidden_dim=8,
        num_layers=2,
        num_atom_types=1,
        atom_types_map=[1],
        use_L1=False,
        use_L2=False,
        use_gating=False,
        use_direct_force=False,
        active_paths={(0, 0, 0, "prod"): True},
    )
    model = AtomBitModel(config)
    state = AseGraphBatch.from_ase(
        [
            Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]]),
            Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0]]),
        ],
        cutoff=config.cutoff,
        device="cpu",
        dtype=torch.float64,
    )
    potential = AtomBitBatchCalculator(model, device="cpu", dtype=torch.float64)
    evaluation = potential(state)
    assert evaluation.energy.shape == (2,)
    assert evaluation.forces.shape == (5, 3)
    assert torch.isfinite(evaluation.energy).all()
    assert torch.isfinite(evaluation.forces).all()
