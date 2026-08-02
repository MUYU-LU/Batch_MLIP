from __future__ import annotations

import pytest
import torch
from ase import Atoms

from batch_mlip import (
    AutoSchedulerConfig,
    BatchCalculator,
    BatchEvaluation,
    optimize_pool,
)


class PoolQuadraticCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.5, device="cpu", dtype=torch.float64)

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy
        atom_energy = 0.5 * (state.positions * state.positions).sum(dim=-1)
        energy = torch.zeros(
            state.n_systems,
            device=state.device,
            dtype=state.dtype,
        )
        energy.index_add_(0, state.system_idx, atom_energy)
        return BatchEvaluation(
            energy=energy,
            forces=-state.positions.clone(),
            stress=(
                torch.zeros(
                    (state.n_systems, 3, 3),
                    device=state.device,
                    dtype=state.dtype,
                )
                if compute_stress
                else None
            ),
        )


def _config(tmp_path) -> AutoSchedulerConfig:
    return AutoSchedulerConfig(
        cache_path=tmp_path / "optimize-pool.json",
        cache_enabled=False,
        max_batch_size=2,
        multi_gpu_process_cpu_threads=1,
    )


def test_optimize_pool_runs_one_call_probe_fallback_and_closes_workers(tmp_path):
    systems = [
        Atoms("H", positions=[[0.8, 0.0, 0.0]]),
        Atoms("He", positions=[[-0.6, 0.1, 0.0]]),
    ]

    result = optimize_pool(
        systems,
        PoolQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        optimizer="bfgs",
        policy="probe",
        auto_config=_config(tmp_path),
        fmax=1e-5,
        max_steps=100,
    )

    assert bool(result.converged.all())
    assert [atoms.get_chemical_formula() for atoms in result.structures] == [
        "H",
        "He",
    ]
    interface = result.metadata["optimize_pool"]
    assert interface["requested_policy"] == "probe"
    assert interface["capacity_planning"]["mode"] == (
        "representative_probe_fallback"
    )
    assert sorted(
        interface["executor_shutdown"]["acknowledged_worker_ids"]
    ) == [0, 1]


def test_optimize_pool_resolves_frechet_cell_filter(tmp_path):
    system = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )

    result = optimize_pool(
        system,
        PoolQuadraticCalculator(),
        devices=["cpu:0"],
        optimizer="bfgs",
        cell_filter="frechet",
        policy="probe",
        auto_config=_config(tmp_path),
        fmax=1e-5,
        max_steps=2,
    )

    assert bool(result.converged.all())
    assert result.max_stress is not None
    task = result.execution_policy["profile"]["layers"]["task_auxiliary"]
    assert task["variable_cell"]


def test_optimize_pool_rejects_unknown_cell_filter():
    with pytest.raises(ValueError, match="cell_filter"):
        optimize_pool(
            Atoms("H", positions=[[0.0, 0.0, 0.0]]),
            PoolQuadraticCalculator(),
            devices=["cpu:0"],
            cell_filter="unit-cell",  # type: ignore[arg-type]
        )
