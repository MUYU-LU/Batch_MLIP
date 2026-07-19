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

from batch_mlip import (
    FrechetCellFilter,
    MACEBatchCalculator,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
    relax,
)

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
        graph_mode="cached",
        skin=0.5,
    )


@pytest.fixture(scope="module")
def mace_rebuild_calculator(
    mace_calculator: MACEBatchCalculator,
) -> MACEBatchCalculator:
    return MACEBatchCalculator(
        mace_calculator.model,
        device=mace_calculator.device,
        dtype=mace_calculator.dtype,
        graph_mode="rebuild",
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_mace_cached_tensor_state_matches_atomic_data(
    mace_systems: list[Any],
    mace_calculator: MACEBatchCalculator,
    mace_rebuild_calculator: MACEBatchCalculator,
    batch_size: int,
) -> None:
    systems = mace_systems[:batch_size]
    cached_state = mace_calculator.create_state(systems)
    cached = mace_calculator(cached_state, compute_stress=True)
    rebuild_count = cached_state.neighbor_rebuild_count
    cached_again = mace_calculator(cached_state, compute_stress=True)

    rebuilt = mace_rebuild_calculator(
        mace_rebuild_calculator.create_state(systems), compute_stress=True
    )

    assert cached_state.neighbor_rebuild_count == rebuild_count == 1
    torch.testing.assert_close(cached.energy, rebuilt.energy, atol=2e-8, rtol=0.0)
    torch.testing.assert_close(cached.forces, rebuilt.forces, atol=2e-8, rtol=0.0)
    torch.testing.assert_close(cached.stress, rebuilt.stress, atol=2e-10, rtol=0.0)
    torch.testing.assert_close(cached_again.energy, cached.energy)
    torch.testing.assert_close(cached_again.forces, cached.forces)
    torch.testing.assert_close(cached_again.stress, cached.stress)


def test_mace_cached_b1_bfgs_matches_ase(
    mace_systems: list[Any],
    mace_calculator: MACEBatchCalculator,
) -> None:
    systems = mace_systems[:1]
    reference = _run_ase(systems, mace_calculator, "bfgs")
    result = relax(
        systems,
        mace_calculator,
        optimizer="bfgs",
        cell_filter=FrechetCellFilter(),
        **_optimizer_options("bfgs"),
    )
    _assert_batch_matches_ase(reference, result)


def test_mace_cached_nve_matches_rebuilt_energy_drift(
    mace_systems: list[Any],
    mace_calculator: MACEBatchCalculator,
    mace_rebuild_calculator: MACEBatchCalculator,
) -> None:
    systems = mace_systems[:2]
    cached_state = mace_calculator.create_state(systems)
    rebuilt_state = mace_rebuild_calculator.create_state(systems)
    initialize_maxwell_boltzmann(
        cached_state,
        300.0,
        seed=17,
        remove_com=True,
        force_exact_temperature=True,
    )
    rebuilt_state.velocities = cached_state.velocities.clone()
    cached_energy = []
    rebuilt_energy = []

    def record_cached(step, state, evaluation, diagnostics):
        cached_energy.append(diagnostics["total_energy"].clone())

    def record_rebuilt(step, state, evaluation, diagnostics):
        rebuilt_energy.append(diagnostics["total_energy"].clone())

    cached = batched_velocity_verlet(
        cached_state,
        mace_calculator,
        timestep_fs=0.1,
        n_steps=4,
        callback=record_cached,
    )
    rebuilt = batched_velocity_verlet(
        rebuilt_state,
        mace_rebuild_calculator,
        timestep_fs=0.1,
        n_steps=4,
        callback=record_rebuilt,
    )

    cached_energy = torch.stack(cached_energy)
    rebuilt_energy = torch.stack(rebuilt_energy)
    cached_drift = cached_energy - cached_energy[0]
    rebuilt_drift = rebuilt_energy - rebuilt_energy[0]
    assert bool(torch.isfinite(cached_drift).all())
    assert float(cached_drift.abs().max()) < 1e-3
    torch.testing.assert_close(cached_drift, rebuilt_drift, atol=2e-9, rtol=0.0)
    torch.testing.assert_close(
        cached.state.positions, rebuilt.state.positions, atol=2e-10, rtol=0.0
    )
    torch.testing.assert_close(
        cached.state.velocities, rebuilt.state.velocities, atol=2e-11, rtol=0.0
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
