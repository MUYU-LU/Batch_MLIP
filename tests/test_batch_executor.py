from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from batch_mlip import (
    AutoSchedulerConfig,
    BatchCalculator,
    BatchEvaluation,
    BatchExecutor,
    relax,
)


class ExecutorQuadraticCalculator(BatchCalculator):
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
        stress = (
            torch.zeros(
                (state.n_systems, 3, 3),
                device=state.device,
                dtype=state.dtype,
            )
            if compute_stress
            else None
        )
        return BatchEvaluation(
            energy=energy,
            forces=-state.positions.clone(),
            stress=stress,
        )


def _systems() -> list[Atoms]:
    return [
        Atoms("H", positions=[[0.8, -0.1, 0.05]]),
        Atoms("H2", positions=[[0.6, 0.0, 0.0], [-0.3, 0.2, 0.0]]),
        Atoms("He", positions=[[-0.7, 0.1, 0.0]]),
        Atoms("Li2", positions=[[0.2, 0.3, 0.0], [-0.5, -0.2, 0.1]]),
    ]


def _config(tmp_path) -> AutoSchedulerConfig:
    return AutoSchedulerConfig(
        cache_path=tmp_path / "executor-cache.json",
        initial_batch_size=1,
        growth_factor=2,
        max_batch_size=2,
        multi_gpu_cold_start_jobs=1,
        multi_gpu_process_cpu_threads=1,
    )


def test_batch_executor_reuses_workers_and_restores_input_order(tmp_path):
    systems = _systems()
    config = _config(tmp_path)
    reference = relax(
        systems,
        ExecutorQuadraticCalculator(),
        optimizer="fire",
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        active_compaction=True,
    )

    with BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as executor:
        first = executor.relax(
            systems,
            optimizer="fire",
            fmax=1e-5,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )
        first_pids = executor.worker_pids
        second = executor.relax(
            systems,
            optimizer="fire",
            fmax=1e-5,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )

        assert bool(first.converged.all())
        assert bool(second.converged.all())
        assert executor.worker_generation == 1
        assert executor.worker_pids == first_pids
        assert len(first_pids) == 2
        assert first.metadata["scheduling"]["worker_startup_seconds_this_call"] > 0
        assert second.metadata["scheduling"]["worker_startup_seconds_this_call"] == 0
        assert second.metadata["scheduling"]["executor_call"] == 2
        assert second.metadata["scheduling"]["decision"] == (
            "persistent_batch_executor"
        )
        assert [
            atoms.get_chemical_formula() for atoms in second.structures
        ] == ["H", "H2", "He", "Li2"]
        torch.testing.assert_close(
            first.evaluation.energy,
            second.evaluation.energy,
        )
        torch.testing.assert_close(
            reference.evaluation.energy,
            second.evaluation.energy,
        )
        torch.testing.assert_close(
            first.evaluation.forces,
            second.evaluation.forces,
        )
        torch.testing.assert_close(
            reference.evaluation.forces,
            second.evaluation.forces,
        )
        np.testing.assert_allclose(
            first.state.positions.cpu(),
            second.state.positions.cpu(),
            atol=1e-12,
        )

    assert executor.closed
    with pytest.raises(RuntimeError, match="closed"):
        executor.relax(systems)


def test_batch_executor_reuses_native_generation_across_optimizers(tmp_path):
    systems = _systems()
    config = _config(tmp_path)

    with BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as executor:
        fire = executor.relax(
            systems,
            optimizer="fire",
            fmax=1e-4,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )
        pids = executor.worker_pids
        bfgs = executor.relax(
            systems,
            optimizer="bfgs",
            fmax=1e-4,
            max_steps=100,
            max_step=0.2,
            alpha=70.0,
            optimizer_dtype="float64",
        )

        assert bool(fire.converged.all())
        assert bool(bfgs.converged.all())
        assert executor.worker_generation == 1
        assert executor.worker_pids == pids
        assert bfgs.metadata["scheduling"]["worker_generation"] == 1


def test_batch_executor_validates_configuration_before_starting(tmp_path):
    systems = _systems()
    executor = BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0"],
        auto_config=AutoSchedulerConfig(
            cache_path=tmp_path / "disabled.json",
            cache_enabled=False,
        ),
    )
    try:
        with pytest.raises(ValueError, match="policy cache"):
            executor.relax(systems)
        assert not executor.started
    finally:
        executor.close()


def test_batch_executor_rejects_thread_backend(tmp_path):
    executor = BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0"],
        auto_config=AutoSchedulerConfig(
            cache_path=tmp_path / "thread.json",
            multi_gpu_worker_backend="thread",
        ),
    )
    try:
        with pytest.raises(ValueError, match="does not accept"):
            executor.relax(_systems())
        assert not executor.started
    finally:
        executor.close()
