from __future__ import annotations

import pytest
import torch
from ase import Atoms

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator
from src.model import AtomBitModel
from src.modules import CartesianDensityBlock
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


@pytest.mark.parametrize(
    ("degree_norm", "expected"),
    [
        ("hard", [3.0**-0.5, 1.0, 1.0]),
        ("smooth_rms", [2.25**-0.5, 2.0**-0.5, 1.0]),
    ],
)
def test_degree_normalization_modes(degree_norm, expected):
    config = AtomBitConfig(
        hidden_dim=4,
        num_layers=1,
        num_atom_types=1,
        atom_types_map=[1],
        use_L1=False,
        use_L2=False,
        use_gating=False,
        degree_norm=degree_norm,
        active_paths={(0, 0, 0, "prod"): True},
    )
    model = AtomBitModel(config)
    center = torch.tensor([0, 0, 0, 1])
    cutoff_weight = torch.tensor([1.0, 0.5, 0.0, 1.0])

    actual = model._compute_inv_sqrt_degree(
        center,
        cutoff_weight,
        3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(actual, torch.tensor(expected))


def test_density_requires_model_supplied_degree_normalization():
    config = AtomBitConfig(
        hidden_dim=4,
        num_atom_types=1,
        atom_types_map=[1],
        use_L1=False,
        use_L2=False,
    )
    density = CartesianDensityBlock(config)

    with pytest.raises(ValueError, match="requires inv_sqrt_deg"):
        density(
            {0: torch.ones((1, 4)), 1: None, 2: None},
            torch.tensor([0]),
            1,
        )


def test_public_checkpoint_constructor_preserves_dtype_and_e0(tmp_path):
    config = AtomBitConfig(
        hidden_dim=4,
        num_layers=1,
        num_atom_types=1,
        atom_types_map=[1],
        use_L1=False,
        use_L2=False,
        use_gating=False,
        degree_norm="smooth_rms",
        active_paths={(0, 0, 0, "prod"): True},
    )
    source = AtomBitModel(config).to(torch.float64)
    checkpoint = tmp_path / "smooth_fp64.pt"
    torch.save(
        {
            "model_config": config,
            "model_state_dict": source.state_dict(),
            "precision_dtype": torch.float64,
        },
        checkpoint,
    )
    e0 = tmp_path / "e0.pt"
    torch.save({"e0_dict": {1: -13.6}}, e0)

    calculator = AtomBitBatchCalculator.from_checkpoint(
        checkpoint,
        e0=e0,
        cutoff=3.0,
    )

    loaded = calculator.model
    floating_state = [
        value for value in loaded.state_dict().values() if value.is_floating_point()
    ]
    assert floating_state
    assert {value.dtype for value in floating_state} == {torch.float64}
    assert calculator.dtype == torch.float64
    assert calculator.cutoff == 3.0
    assert calculator.e0_dict == {1: -13.6}


def test_public_checkpoint_constructor_rejects_incomplete_payload(tmp_path):
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"model_config": {}}, checkpoint)

    with pytest.raises(KeyError, match="model_state_dict"):
        AtomBitBatchCalculator.from_checkpoint(checkpoint)
