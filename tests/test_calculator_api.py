from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from batch_mlip import (
    ASECalculatorAdapter,
    AutoSchedulerConfig,
    BatchCalculator,
    BatchEvaluation,
    BatchPlanner,
    BatchTimingPoint,
    MemoryCoefficients,
    OptimizationPilot,
    PilotRegime,
    evaluate,
    molecular_dynamics,
    relax,
    relax_ase,
)


class QuadraticBatchCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.5, device="cpu", dtype=torch.float64)
        self.calls = 0

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy
        if compute_stress:
            raise NotImplementedError
        self.calls += 1
        atom_energy = 0.5 * (state.positions * state.positions).sum(dim=-1)
        energy = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype)
        energy.index_add_(0, state.system_idx, atom_energy)
        return BatchEvaluation(energy=energy, forces=-state.positions.clone())


class QuadraticASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.5 * float((atoms.positions**2).sum())
        self.results["forces"] = -atoms.positions.copy()


def test_one_generic_batch_calculator_drives_evaluation_fire_and_md():
    systems = [
        Atoms("H", positions=[[0.8, -0.2, 0.1]]),
        Atoms("H2", positions=[[0.3, 0.1, 0.0], [-0.4, 0.2, 0.1]]),
    ]
    calculator = QuadraticBatchCalculator()

    prediction = evaluate(systems, calculator)
    assert prediction.evaluation.energy.shape == (2,)
    assert len(prediction.structures) == 2
    np.testing.assert_allclose(prediction.structures[0].get_potential_energy(), 0.345, atol=1e-15)

    relaxed = relax(
        systems,
        calculator,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        active_compaction=True,
    )
    assert bool(relaxed.converged.all())
    assert len(relaxed.structures) == 2

    dynamics = molecular_dynamics(
        systems,
        calculator,
        ensemble="nve",
        timestep_fs=0.1,
        n_steps=2,
    )
    assert dynamics.steps == 2
    assert len(dynamics.structures) == 2
    assert calculator.calls > 3


def test_ordinary_ase_calculator_adapter_is_compatible_but_sequential():
    systems = [
        Atoms("H", positions=[[1.0, 0.0, 0.0]]),
        Atoms("He2", positions=[[0.0, 0.5, 0.0], [0.0, 0.0, -0.2]]),
    ]
    calculator = ASECalculatorAdapter(QuadraticASECalculator())

    result = evaluate(systems, calculator)

    torch.testing.assert_close(
        result.evaluation.energy,
        torch.tensor([0.5, 0.145], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.evaluation.forces,
        -torch.as_tensor(
            np.concatenate([atoms.positions for atoms in systems]),
            dtype=torch.float64,
        ),
    )


def test_auto_relaxation_chunks_and_restores_heterogeneous_input_order():
    systems = [
        Atoms("H", positions=[[0.8, -0.2, 0.1]]),
        Atoms("He2", positions=[[0.3, 0.1, 0.0], [-0.4, 0.2, 0.1]]),
    ]
    calculator = QuadraticBatchCalculator()
    planner = BatchPlanner(
        MemoryCoefficients(100.0, 1.0, 1.0, 1.0),
        memory_budget_bytes=1_000_000,
        max_batch_size=1,
        max_cost_ratio=100.0,
    )

    result = relax(
        systems,
        calculator,
        scheduling="auto",
        planner=planner,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
        active_compaction=True,
    )

    assert bool(result.converged.all())
    assert [atoms.get_chemical_formula() for atoms in result.structures] == [
        "H",
        "He2",
    ]
    scheduling = result.metadata["scheduling"]
    assert scheduling["decision"] == "memory_safe_planned_queues"
    assert scheduling["batches"] == [
        {
            "system_count": 2,
            "resident_capacity": 1,
            "active_refill": True,
        }
    ]


def test_task_aware_relaxation_executes_selected_tensor_schedule():
    systems = [
        Atoms("H", positions=[[0.8, -0.2, 0.1]]),
        Atoms("H", positions=[[-0.4, 0.2, 0.1]]),
    ]
    calculator = QuadraticBatchCalculator()
    planner = BatchPlanner(
        MemoryCoefficients(100.0, 1.0, 1.0, 1.0),
        memory_budget_bytes=1_000_000,
        max_batch_size=2,
        max_cost_ratio=100.0,
    )
    pilot = OptimizationPilot(
        optimizer="fire",
        regimes=(
            PilotRegime(
                label="one-atom",
                atom_count=1,
                edge_count=0,
                sampled_steps=(10, 10),
                timing_points=(
                    BatchTimingPoint(1, 0.01),
                    BatchTimingPoint(2, 0.012),
                ),
            ),
        ),
    )

    result = relax(
        systems,
        calculator,
        scheduling="auto",
        planner=planner,
        pilot=pilot,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    assert bool(result.converged.all())
    scheduling = result.metadata["scheduling"]
    assert scheduling["decision"] == "task_aware_pilot_policy"
    assert scheduling["recommended_worker_mode"] == "tensor"
    assert scheduling["batches"] == [
        {
            "system_count": 2,
            "resident_capacity": 2,
            "active_refill": False,
            "refill_storage": "slots",
            "predicted_seconds": pytest.approx(0.132),
        }
    ]


def test_online_auto_relaxation_needs_no_explicit_planner_or_pilot(tmp_path):
    systems = [
        Atoms("H", positions=[[0.2 + 0.05 * index, 0.0, 0.0]])
        for index in range(8)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "auto.json",
        initial_batch_size=1,
        growth_factor=2,
        max_batch_size=4,
        refill_min_capacity=4,
    )

    cold = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )
    warm = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    assert bool(cold.converged.all())
    assert bool(warm.converged.all())
    assert [
        atoms.get_chemical_formula() for atoms in cold.structures
    ] == ["H"] * len(systems)
    cold_schedule = cold.metadata["scheduling"]
    warm_schedule = warm.metadata["scheduling"]
    assert cold_schedule["decision"] == "online_autotune"
    assert not cold_schedule["buckets"][0]["cache_hit"]
    assert warm_schedule["buckets"][0]["cache_hit"]
    assert sum(
        batch["system_count"] for batch in cold_schedule["batches"]
    ) == len(systems)
    assert config.resolved_cache_path().exists()


def test_multi_device_auto_relaxation_cold_tunes_then_steals_pending_work(
    tmp_path,
):
    systems = [
        Atoms("H", positions=[[0.2 + 0.05 * index, 0.0, 0.0]])
        for index in range(8)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "multi-auto.json",
        initial_batch_size=1,
        growth_factor=2,
        max_batch_size=4,
        refill_min_capacity=4,
        multi_gpu_cold_start_jobs=2,
        multi_gpu_worker_backend="process",
    )

    result = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    assert bool(result.converged.all())
    schedule = result.metadata["scheduling"]
    assert schedule["decision"] == "online_autotune_multi_gpu"
    assert schedule["gpu_count"] == 2
    assert schedule["active_gpu_count"] == 2
    assert schedule["pending_work_stealing"]
    assert schedule["worker_backend"] == "process"
    assert schedule["worker_backend_fallback_reason"] is None
    assert schedule["allocator"]["selected_policy"] == "native"
    assert not schedule["allocator"]["applied_to_workers"]
    assert all(
        not worker["allocator"]["applied"] for worker in schedule["workers"]
    )
    assert sum(
        record["system_count"] for record in schedule["cold_start"]
    ) == 2
    assert sum(
        chunk["system_count"]
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    ) == 6

    warm = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )
    warm_schedule = warm.metadata["scheduling"]
    assert warm_schedule["cold_start"] == []
    assert sum(
        chunk["system_count"]
        for worker in warm_schedule["workers"]
        for chunk in worker["chunks"]
    ) == len(systems)


def test_single_explicit_device_can_use_process_worker(tmp_path):
    systems = [
        Atoms("H", positions=[[0.2 + 0.05 * index, 0.0, 0.0]])
        for index in range(4)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "single-process-auto.json",
        max_batch_size=2,
        multi_gpu_cold_start_jobs=1,
        multi_gpu_worker_backend="process",
    )

    result = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        devices=["cpu:0"],
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["gpu_count"] == 1
    assert schedule["active_gpu_count"] == 1
    assert schedule["worker_backend"] == "process"
    assert schedule["allocator"]["selected_policy"] == "native"
    assert not schedule["allocator"]["applied_to_workers"]


def test_multi_device_auto_relaxation_accepts_explicit_thread_backend(tmp_path):
    systems = [
        Atoms("H", positions=[[0.2 + 0.05 * index, 0.0, 0.0]])
        for index in range(4)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "thread-auto.json",
        max_batch_size=2,
        multi_gpu_cold_start_jobs=1,
        multi_gpu_worker_backend="thread",
    )

    result = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["worker_backend"] == "thread"
    assert schedule["worker_backend_fallback_reason"] is None


def test_multi_device_auto_uses_threads_for_one_wave_per_device(tmp_path):
    systems = [
        Atoms("H", positions=[[0.2 + 0.05 * index, 0.0, 0.0]])
        for index in range(4)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "one-wave-auto.json",
        max_batch_size=2,
        multi_gpu_cold_start_jobs=1,
    )

    result = relax(
        systems,
        QuadraticBatchCalculator(),
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert schedule["worker_backend"] == "thread"
    assert "8 pending chunks" in schedule["worker_backend_fallback_reason"]


def test_scheduled_relaxation_reassembles_results_on_calculator_device():
    systems = [
        Atoms("H", positions=[[0.8, 0.0, 0.0]]),
        Atoms("He2", positions=[[0.3, 0.0, 0.0], [-0.4, 0.0, 0.0]]),
    ]
    calculator = QuadraticBatchCalculator()
    planner = BatchPlanner(
        MemoryCoefficients(10.0, 1.0, 0.0, 0.0),
        memory_budget_bytes=1_000,
        max_batch_size=1,
        max_cost_ratio=1.0,
    )
    pilot = OptimizationPilot(
        optimizer="fire",
        regimes=(
            PilotRegime(
                label="small",
                atom_count=1,
                edge_count=0,
                sampled_steps=(10,),
                timing_points=(BatchTimingPoint(1, 0.01),),
            ),
        ),
    )

    result = relax(
        systems,
        calculator,
        scheduling="auto",
        planner=planner,
        pilot=pilot,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    assert result.state.device == calculator.device
    assert result.evaluation.energy.device == calculator.device
    assert result.evaluation.forces.device == calculator.device
    assert [atoms.get_chemical_formula() for atoms in result.structures] == [
        "H",
        "He2",
    ]


def test_relax_ase_runs_an_explicit_native_ase_reference():
    systems = [
        Atoms("H", positions=[[0.8, -0.2, 0.1]]),
        Atoms("He", positions=[[-0.4, 0.2, 0.1]]),
    ]

    result = relax_ase(
        systems,
        QuadraticASECalculator(),
        optimizer="bfgs",
        fmax=1e-8,
        max_steps=100,
    )

    assert bool(result.converged.all())
    assert result.metadata["execution"] == "strict_ase_serial"
    assert float(torch.linalg.vector_norm(result.state.positions, dim=1).max()) < 1e-8
