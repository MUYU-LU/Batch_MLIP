from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.io import read
from ase.optimize import BFGS, FIRE

from batch_mlip import FrechetCellFilter, MACEBatchCalculator, relax

RUN_MACE_TESTS = os.environ.get("BATCH_MLIP_RUN_MACE_TESTS") == "1"
DEVICE = os.environ.get("BATCH_MLIP_MACE_DEVICE", "cuda:0")
ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.mace,
    pytest.mark.skipif(
        not RUN_MACE_TESTS,
        reason="set BATCH_MLIP_RUN_MACE_TESTS=1 to run MACE integration tests",
    ),
]


@pytest.fixture(scope="module")
def mace_systems() -> list[Any]:
    dataset_dir = ROOT / "data" / "T2_test" / "structures"
    manifest_path = ROOT / "benchmarks" / "t2_fixed_samples.json"
    if not dataset_dir.is_dir():
        pytest.fail(f"MACE integration dataset is missing: {dataset_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest["samples"]["46"][:4]
    systems = [read(dataset_dir / name) for name in names]
    if any(len(atoms) != 46 for atoms in systems):
        pytest.fail("the fixed MACE integration pool must contain 46-atom systems")
    return systems


@pytest.fixture(scope="module")
def mace_calculator() -> MACEBatchCalculator:
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.fail(f"MACE integration requested {device}, but CUDA is unavailable")
    if importlib.util.find_spec("mace") is None:
        pytest.fail("MACE integration was enabled, but mace-torch is not installed")
    return MACEBatchCalculator.from_off(
        model="small",
        device=device,
        dtype=torch.float64,
    )


def _optimizer_options(optimizer: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "fmax": 1e-30,
        "max_steps": 3,
        "max_step": 0.2,
    }
    if optimizer == "fire":
        common.update(dt_start=0.1, dt_max=1.0)
    else:
        common.update(alpha=70.0, optimizer_dtype="float64")
    return common


def _run_ase(
    systems: list[Any],
    calculator: MACEBatchCalculator,
    optimizer_name: str,
) -> list[dict[str, Any]]:
    # Import after from_off initializes CUDA; this is required by legacy MACE
    # checkpoints in the supplied environment.
    from mace.calculators import MACECalculator

    ase_calculator = MACECalculator(
        models=calculator.model,
        device=str(calculator.device),
        default_dtype="float64",
    )
    records = []
    for source in systems:
        atoms = source.copy()
        atoms.calc = ase_calculator
        target = ASEFrechetCellFilter(atoms)
        if optimizer_name == "fire":
            optimizer = FIRE(
                target,
                logfile=None,
                trajectory=None,
                dt=0.1,
                dtmax=1.0,
                maxstep=0.2,
            )
        else:
            optimizer = BFGS(
                target,
                logfile=None,
                trajectory=None,
                alpha=70.0,
                maxstep=0.2,
            )
        optimizer.run(fmax=1e-30, steps=3)
        records.append(
            {
                "steps": optimizer.nsteps,
                "energy": atoms.get_potential_energy(),
                "forces": atoms.get_forces(),
                "stress": atoms.get_stress(voigt=False),
                "positions": atoms.positions.copy(),
                "cell": atoms.cell.array.copy(),
            }
        )
    return records


def _assert_batch_matches_ase(reference, result) -> None:
    energies = result.evaluation.energy.detach().cpu().numpy()
    forces = result.evaluation.forces.detach().cpu().numpy()
    stresses = result.evaluation.stress.detach().cpu().numpy()
    positions = result.state.positions.detach().cpu().numpy()
    cells = result.state.cells.detach().cpu().numpy()
    for system_id, expected in enumerate(reference):
        atom_slice = result.state.atom_slice(system_id)
        assert result.steps == expected["steps"]
        np.testing.assert_allclose(
            energies[system_id], expected["energy"], atol=2e-8, rtol=0.0
        )
        np.testing.assert_allclose(
            forces[atom_slice], expected["forces"], atol=2e-8, rtol=0.0
        )
        np.testing.assert_allclose(
            stresses[system_id], expected["stress"], atol=2e-10, rtol=0.0
        )
        np.testing.assert_allclose(
            positions[atom_slice], expected["positions"], atol=2e-9, rtol=0.0
        )
        np.testing.assert_allclose(
            cells[system_id], expected["cell"], atol=2e-9, rtol=0.0
        )


@pytest.mark.parametrize("optimizer_name", ["fire", "bfgs"])
def test_mace_variable_cell_ase_masked_active_equivalence(
    mace_systems: list[Any],
    mace_calculator: MACEBatchCalculator,
    optimizer_name: str,
) -> None:
    reference = _run_ase(mace_systems, mace_calculator, optimizer_name)
    options = _optimizer_options(optimizer_name)
    masked = relax(
        mace_systems,
        mace_calculator,
        optimizer=optimizer_name,
        cell_filter=FrechetCellFilter(),
        active_compaction=False,
        **options,
    )
    active = relax(
        mace_systems,
        mace_calculator,
        optimizer=optimizer_name,
        cell_filter=FrechetCellFilter(),
        active_compaction=True,
        **options,
    )

    _assert_batch_matches_ase(reference, masked)
    _assert_batch_matches_ase(reference, active)
    torch.testing.assert_close(
        active.state.positions, masked.state.positions, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        active.state.cells, masked.state.cells, atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        active.evaluation.energy,
        masked.evaluation.energy,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        active.evaluation.forces,
        masked.evaluation.forces,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        active.evaluation.stress,
        masked.evaluation.stress,
        atol=1e-12,
        rtol=1e-12,
    )
    assert masked.model_evaluations == active.model_evaluations == 4
    assert masked.graph_evaluations == active.graph_evaluations == 16
    assert all(size == 4 for size in active.active_batch_sizes)
