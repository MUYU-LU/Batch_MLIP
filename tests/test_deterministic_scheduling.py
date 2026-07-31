from __future__ import annotations

from dataclasses import replace

import torch
from ase import Atoms

from batch_mlip import (
    AutoSchedulerConfig,
    AutoWorkloadBucket,
    AutoWorkloadPlan,
    BatchCalculator,
    BatchedBFGS,
    BatchedFIRE,
    BatchEvaluation,
    DeterministicMemoryProbe,
    SystemProfile,
    plan_deterministic_relaxation,
    relax,
)
from batch_mlip.interfaces.api import (
    _parallel_deterministic_chunk_policy,
    _parallel_deterministic_chunks,
    _parallel_deterministic_dispatch_order,
    _reserved_incremental_bytes,
)


class QuadraticCalculator(BatchCalculator):
    def __init__(self) -> None:
        super().__init__(cutoff=2.5, device="cpu", dtype=torch.float64)

    def calculate(
        self,
        state,
        *,
        neighbor_policy="auto",
        compute_stress=False,
    ) -> BatchEvaluation:
        del neighbor_policy, compute_stress
        atom_energy = 0.5 * state.positions.square().sum(dim=-1)
        energy = torch.zeros(
            state.n_systems,
            device=state.device,
            dtype=state.dtype,
        )
        energy.index_add_(0, state.system_idx, atom_energy)
        return BatchEvaluation(energy=energy, forces=-state.positions.clone())


def _workload(system_count: int = 3) -> AutoWorkloadPlan:
    profiles = tuple(
        SystemProfile(index=index, atom_count=1, edge_count=0, dof_squared=9)
        for index in range(system_count)
    )
    return AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=tuple(range(system_count)),
                mean_atom_count=1.0,
                mean_edge_count=0.0,
                mean_dof_squared=9.0,
                homogeneous_atom_count=True,
            ),
        ),
        profiling_seconds=0.0,
        fingerprint="test",
        fingerprint_fields={},
    )


def test_deterministic_planner_respects_absolute_memory_budget():
    config = AutoSchedulerConfig(
        memory_budget_bytes=850,
        memory_growth_margin=1.0,
        max_batch_size=10,
    )
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=850,
        baseline_allocated_bytes=100,
        peak_allocated_bytes=356,
        peak_reserved_bytes=400,
        probe_indices=(0,),
        probe_model_work=256,
        model_bytes_per_work=1.0,
    )

    plan = plan_deterministic_relaxation(
        _workload(),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )

    assert [len(chunk.system_indices) for chunk in plan.chunks] == [2, 1]
    assert all(
        chunk.predicted_peak_bytes is not None
        and chunk.predicted_peak_bytes <= 850
        for chunk in plan.chunks
    )
    assert sorted(
        index for chunk in plan.chunks for index in chunk.system_indices
    ) == [0, 1, 2]


def test_probe_incremental_memory_uses_reserved_device_occupancy():
    assert _reserved_incremental_bytes(
        baseline_allocated=100,
        peak_allocated=400,
        peak_reserved=700,
    ) == 600
    assert _reserved_incremental_bytes(
        baseline_allocated=100,
        peak_allocated=400,
        peak_reserved=350,
    ) == 300


def test_parallel_chunks_preserve_full_batches_when_devices_are_occupied():
    config = AutoSchedulerConfig(max_batch_size=2)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    plan = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )

    chunks = _parallel_deterministic_chunks(
        plan,
        device_count=2,
        target_chunks_per_device=2,
    )

    assert len(plan.chunks) == 4
    assert [len(chunk.indices) for chunk in chunks] == [2, 2, 2, 2]
    assert sorted(
        index for chunk in chunks for index in chunk.indices
    ) == list(range(8))


def test_parallel_chunks_add_only_minimum_parts_for_idle_devices():
    config = AutoSchedulerConfig(max_batch_size=4)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    plan = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )

    chunks = _parallel_deterministic_chunks(
        plan,
        device_count=4,
        target_chunks_per_device=1,
    )

    assert len(plan.chunks) == 2
    assert len(chunks) == 4
    assert [len(chunk.indices) for chunk in chunks] == [2, 2, 2, 2]
    assert sorted(
        index for chunk in chunks for index in chunk.indices
    ) == list(range(8))


def test_parallel_chunks_can_preserve_resident_batches_and_idle_devices():
    config = AutoSchedulerConfig(max_batch_size=8)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    plan = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )

    chunks = _parallel_deterministic_chunks(
        plan,
        device_count=4,
        target_chunks_per_device=2,
        dispatch_policy="preserve_resident",
    )

    assert len(plan.chunks) == 1
    assert [chunk.indices for chunk in chunks] == [tuple(range(8))]
    assert _parallel_deterministic_chunk_policy(
        plan,
        device_count=4,
        target_chunks_per_device=2,
        dispatch_policy="preserve_resident",
    ) == "resident_batches_preserved"


def test_parallel_chunks_split_only_enough_to_occupy_devices():
    config = AutoSchedulerConfig(max_batch_size=4)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    homogeneous = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )
    plan = replace(
        homogeneous,
        chunks=(
            homogeneous.chunks[0],
            replace(homogeneous.chunks[1], bucket_index=1),
        ),
    )

    chunks = _parallel_deterministic_chunks(
        plan,
        device_count=4,
        target_chunks_per_device=1,
    )

    assert len(chunks) == 4
    assert [len(chunk.indices) for chunk in chunks] == [2] * 4
    assert sorted(
        index for chunk in chunks for index in chunk.indices
    ) == list(range(8))


def test_parallel_dispatch_can_stratify_only_the_initial_wave():
    config = AutoSchedulerConfig(max_batch_size=2)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    plan = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )
    chunks = _parallel_deterministic_chunks(
        replace(
            plan,
            chunks=(
                replace(plan.chunks[0], bucket_index=0),
                replace(plan.chunks[1], bucket_index=0),
                replace(plan.chunks[2], bucket_index=1),
                replace(plan.chunks[3], bucket_index=2),
            ),
        ),
        device_count=2,
        target_chunks_per_device=2,
    )

    cost_order = _parallel_deterministic_dispatch_order(
        chunks,
        worker_count=3,
        queue_policy="cost_descending",
    )
    stratified = _parallel_deterministic_dispatch_order(
        chunks,
        worker_count=3,
        queue_policy="bucket_stratified",
    )

    assert cost_order == (0, 1, 2, 3)
    assert stratified == (0, 2, 3, 1)
    assert sorted(stratified) == list(range(len(chunks)))


def test_parallel_chunks_preserve_heterogeneous_resident_batches():
    config = AutoSchedulerConfig(max_batch_size=2)
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=None,
        baseline_allocated_bytes=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        probe_indices=(),
        probe_model_work=0,
        model_bytes_per_work=0.0,
    )
    homogeneous = plan_deterministic_relaxation(
        _workload(8),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )
    plan = replace(
        homogeneous,
        chunks=tuple(
            replace(chunk, bucket_index=index % 2)
            for index, chunk in enumerate(homogeneous.chunks)
        ),
    )

    chunks = _parallel_deterministic_chunks(
        plan,
        device_count=4,
        target_chunks_per_device=1,
    )

    assert len(chunks) == 4
    assert [len(chunk.indices) for chunk in chunks] == [2] * 4
    assert sorted(
        index for chunk in chunks for index in chunk.indices
    ) == list(range(8))


def test_dense_bfgs_allowance_changes_deterministic_capacity():
    config = AutoSchedulerConfig(
        memory_budget_bytes=2_000,
        memory_growth_margin=1.0,
        dense_optimizer_tensor_multiplier=16.0,
        max_batch_size=10,
    )
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=2_000,
        baseline_allocated_bytes=100,
        peak_allocated_bytes=100,
        peak_reserved_bytes=100,
        probe_indices=(0,),
        probe_model_work=256,
        model_bytes_per_work=0.0,
    )

    fire = plan_deterministic_relaxation(
        _workload(),
        probe,
        BatchedFIRE(),
        {},
        torch.float64,
        config,
    )
    bfgs = plan_deterministic_relaxation(
        _workload(),
        probe,
        BatchedBFGS(),
        {"optimizer_dtype": "float64"},
        torch.float32,
        config,
    )

    assert [len(chunk.system_indices) for chunk in fire.chunks] == [3]
    assert [len(chunk.system_indices) for chunk in bfgs.chunks] == [1, 1, 1]


def test_auto_relaxation_uses_deterministic_active_drain_without_probe_on_cpu():
    systems = [
        Atoms("H", positions=[[0.8 - 0.1 * index, 0.0, 0.0]])
        for index in range(4)
    ]
    result = relax(
        systems,
        QuadraticCalculator(),
        scheduling="auto",
        auto_config=AutoSchedulerConfig(max_batch_size=2),
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["decision"] == "deterministic_memory_plan"
    assert schedule["memory_fraction"] == 0.85
    assert schedule["probe"]["model_forward_count"] == 0
    assert schedule["active_compaction"]
    assert not schedule["active_refill"]
    assert not schedule["mps"]
    assert [batch["system_count"] for batch in schedule["batches"]] == [2, 2]
    manifest = schedule["policy_manifest"]
    assert result.execution_policy == manifest
    assert manifest["task"]["kind"] == "fixed_cell_relaxation"
    assert manifest["outer_scheduler"]["pool_regime"] == (
        "multiple_resident_waves"
    )
    assert manifest["inner_scheduler"]["queue_policy"] == "active_drain"


def test_ordinary_relaxation_defaults_to_automatic_scheduling():
    systems = [
        Atoms("H", positions=[[0.8 - 0.1 * index, 0.0, 0.0]])
        for index in range(4)
    ]

    result = relax(
        systems,
        QuadraticCalculator(),
        auto_config=AutoSchedulerConfig(max_batch_size=2),
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert schedule["decision"] == "deterministic_memory_plan"
    assert schedule["summary"] == {
        "strategy": "automatic",
        "batch_mode": "active_drain",
        "devices": ["cpu"],
        "device_count": 1,
        "resident_capacities": [2],
        "memory_fraction": 0.85,
        "active_compaction": True,
        "work_stealing": False,
        "refill_reasons": [
            "calculator has no hashable torch model"
        ],
    }


def test_explicit_single_batch_remains_available():
    systems = [
        Atoms("H", positions=[[0.8, 0.0, 0.0]]),
        Atoms("H", positions=[[0.6, 0.0, 0.0]]),
    ]

    result = relax(
        systems,
        QuadraticCalculator(),
        scheduling="single_batch",
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    assert result.schedule == {
        "strategy": "manual",
        "batch_mode": "active_drain",
        "devices": ["cpu"],
        "device_count": 1,
        "resident_capacities": [2],
        "memory_fraction": None,
        "active_compaction": False,
        "work_stealing": False,
        "refill_reasons": [],
    }


def test_multi_device_auto_shards_deterministic_chunks_without_autotuning():
    systems = [
        Atoms("H", positions=[[0.8 - 0.05 * index, 0.0, 0.0]])
        for index in range(8)
    ]
    result = relax(
        systems,
        QuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=AutoSchedulerConfig(
            max_batch_size=8,
            multi_gpu_worker_backend="thread",
        ),
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["decision"] == "deterministic_memory_plan_multi_gpu"
    assert schedule["active_gpu_count"] == 2
    assert schedule["worker_backend"] == "thread"
    assert schedule["parallel_chunk_policy"] == (
        "minimum_parts_for_device_occupancy"
    )
    assert schedule["resident_plan_chunk_count"] == 1
    assert schedule["execution_chunk_count"] == 4
    assert len(schedule["planned_chunks"]) == 4
    assert sum(
        chunk["system_count"] for chunk in schedule["planned_chunks"]
    ) == len(systems)
    assert not schedule["active_refill"]
    assert schedule["summary"] == {
        "strategy": "automatic",
        "batch_mode": "active_drain",
        "devices": ["cpu:0", "cpu:1"],
        "device_count": 2,
        "resident_capacities": [2],
        "memory_fraction": 0.85,
        "active_compaction": True,
        "work_stealing": True,
        "refill_reasons": [
            "multi-GPU refill has no accepted scientific policy"
        ],
    }
    manifest = schedule["policy_manifest"]
    assert manifest["outer_scheduler"]["active_device_count"] == 2
    assert manifest["outer_scheduler"]["execution_chunk_count"] == 4
    assert manifest["inner_scheduler"]["refill"]["evidence_source"] == (
        "policy_exclusion"
    )


def test_multi_device_auto_can_preserve_resident_batch_and_idle_devices():
    systems = [
        Atoms("H", positions=[[0.8 - 0.05 * index, 0.0, 0.0]])
        for index in range(8)
    ]
    result = relax(
        systems,
        QuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=AutoSchedulerConfig(
            max_batch_size=8,
            multi_gpu_worker_backend="thread",
            multi_gpu_dispatch_policy="preserve_resident",
        ),
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["gpu_count"] == 2
    assert schedule["active_gpu_count"] == 1
    assert schedule["multi_gpu_dispatch_policy"] == "preserve_resident"
    assert schedule["parallel_chunk_policy"] == "resident_batches_preserved"
    assert schedule["resident_plan_chunk_count"] == 1
    assert schedule["execution_chunk_count"] == 1
    assert schedule["planned_chunks"][0]["system_count"] == len(systems)
    assert schedule["policy_manifest"]["outer_scheduler"][
        "active_device_count"
    ] == 1


def test_multi_device_auto_supports_persistent_process_workers():
    systems = [
        Atoms("H", positions=[[0.7 - 0.1 * index, 0.0, 0.0]])
        for index in range(4)
    ]
    result = relax(
        systems,
        QuadraticCalculator(),
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=AutoSchedulerConfig(
            max_batch_size=4,
            multi_gpu_worker_backend="process",
        ),
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    schedule = result.metadata["scheduling"]
    assert bool(result.converged.all())
    assert schedule["worker_backend"] == "process"
    assert schedule["active_gpu_count"] == 2
    assert sum(
        chunk["system_count"]
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    ) == len(systems)
