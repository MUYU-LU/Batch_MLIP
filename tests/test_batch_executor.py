from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write

from batch_mlip import (
    AutoSchedulerConfig,
    BatchCalculator,
    BatchedFIRE,
    BatchEvaluation,
    BatchExecutor,
    planning_profile_from_manifest,
    relax,
)
from batch_mlip.workloads import WorkloadJob, WorkloadManifest


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


def _manifest_workload(tmp_path):
    systems = _systems()
    jobs = []
    for index, atoms in enumerate(systems):
        source_path = f"executor-{index}.extxyz"
        path = tmp_path / source_path
        write(path, atoms)
        jobs.append(
            WorkloadJob(
                system_id=f"executor-system-{index}",
                group_id="executor",
                duplicate_group=f"executor-system-{index}",
                order=index,
                dataset_id="executor-test",
                source_path=source_path,
                source_sha256=hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                normalized_structure_sha256=hashlib.sha256(
                    f"executor-normalized-{index}".encode()
                ).hexdigest(),
                frame_index=0,
                atom_count=len(atoms),
                species=tuple(atoms.get_chemical_symbols()),
                chemical_formula=atoms.get_chemical_formula(),
                pbc=(False, False, False),
                cell_A=tuple(float(value) for value in atoms.cell.array.flat),
                volume_A3=0.0,
                constraints=(),
                topology_edge_counts={"active": 0, "candidate": 0},
            )
        )
    manifest = WorkloadManifest(
        workload_id="executor-source-backed-test",
        version=1,
        family="test",
        operation="optimization",
        cell_mode="fixed",
        arrival_mode="closed",
        jobs=tuple(jobs),
        metadata={},
    ).seal()
    profile = planning_profile_from_manifest(
        manifest,
        model_id="executor-quadratic-test",
        cutoff_A=2.5,
        active_edge_key="active",
        candidate_edge_key="candidate",
        force_mode="unspecified",
        model_dtype="torch.float64",
        optimizer=BatchedFIRE(),
        variable_cell=False,
        cell_method=None,
        skin_A=0.0,
        neighbor_backend="auto",
    )
    return systems, manifest, profile


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
            "persistent_deterministic_memory_plan"
        )
        assert second.metadata["scheduling"]["optimization_pilot_runs"] == 0
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
    assert executor.shutdown_metadata is not None
    assert executor.shutdown_metadata["acknowledged_worker_ids"] == [0, 1]
    assert executor.shutdown_metadata["wall_seconds"] < 1.0
    with pytest.raises(RuntimeError, match="closed"):
        executor.relax(systems)


def test_batch_executor_prefetches_manifests_and_reuses_worker_generation(
    tmp_path,
):
    systems, manifest, profile = _manifest_workload(tmp_path)
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "manifest-executor-cache.json",
        cache_enabled=False,
        max_batch_size=2,
        manifest_loader_processes=1,
        manifest_prefetch_chunks_per_worker=1,
        multi_gpu_process_cpu_threads=1,
        multi_gpu_target_chunks_per_device=2,
    )
    options = {
        "fmax": 1e-5,
        "max_steps": 500,
        "dt_start": 0.05,
        "dt_max": 0.5,
    }
    reference = relax(
        systems,
        ExecutorQuadraticCalculator(),
        optimizer="fire",
        **options,
    )

    with BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as executor:
        first = executor.relax_manifest(
            manifest,
            tmp_path,
            profile,
            optimizer="fire",
            **options,
        )
        worker_pids = executor.worker_pids
        second = executor.relax_manifest(
            manifest,
            tmp_path,
            profile,
            optimizer="fire",
            **options,
        )

        assert executor.worker_generation == 1
        assert executor.worker_pids == worker_pids
        assert first.metadata["scheduling"][
            "worker_startup_seconds_this_call"
        ] > 0.0
        schedule = second.metadata["scheduling"]
        assert schedule["decision"] == (
            "persistent_manifest_deterministic_memory_plan"
        )
        assert schedule["worker_startup_seconds_this_call"] == 0.0
        assert schedule["executor_call"] == 2
        materialization = schedule["structure_materialization"]
        assert materialization["mode"] == "manifest_global_prefetch"
        assert materialization["total_loader_processes"] == 2
        assert materialization["maximum_buffered_chunks"] == 4
        assert materialization["materializer_generation"] == 1
        assert not materialization["materializer_generation_restarted"]
        assert materialization["chunk_count"] == 4
        assert materialization["dispatch_wait_seconds"] >= 0.0
        assert all(
            chunk["input"]["mode"] == "manifest_global_prefetch"
            for worker in schedule["workers"]
            for chunk in worker["chunks"]
        )
        torch.testing.assert_close(
            first.evaluation.energy,
            second.evaluation.energy,
        )
        torch.testing.assert_close(
            reference.evaluation.energy,
            second.evaluation.energy,
        )
        torch.testing.assert_close(
            reference.evaluation.forces,
            second.evaluation.forces,
        )


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


def test_batch_executor_deterministic_plan_dispatches_work_to_all_workers(
    tmp_path,
):
    systems = [
        Atoms("H", positions=[[value, 0.0, 0.0]])
        for value in (0.8, 0.6, -0.7, 0.2)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "pilot-cache.json",
        initial_batch_size=2,
        growth_factor=2,
        max_batch_size=2,
        multi_gpu_cold_start_jobs=1,
        multi_gpu_process_cpu_threads=1,
    )
    with BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as executor:
        result = executor.relax(
            systems,
            optimizer="fire",
            fmax=1e-4,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )

    schedule = result.metadata["scheduling"]
    production_workers = {
        worker["worker_id"]
        for worker in schedule["workers"]
        if worker["chunks"]
    }
    production_systems = sum(
        chunk["system_count"]
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    )
    assert schedule["optimization_pilot_runs"] == 0
    assert schedule["parallel_chunk_policy"] == (
        "minimum_parts_for_work_stealing"
    )
    assert schedule["resident_plan_chunk_count"] == 2
    assert schedule["execution_chunk_count"] == 4
    assert sum(
        chunk["system_count"] for chunk in schedule["planned_chunks"]
    ) == 4
    assert production_systems == 4
    assert production_workers == {0, 1}
    assert all(
        "peak_allocated_bytes" in chunk
        and "peak_reserved_bytes" in chunk
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    )
    assert all(
        chunk["peak_allocated_bytes"] is None
        and chunk["peak_reserved_bytes"] is None
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    )
    assert all(
        chunk["runtime_profile"]["schema_version"] == 1
        and chunk["runtime_profile"]["total_seconds"] >= 0.0
        and "samples" not in chunk["runtime_profile"]
        for worker in schedule["workers"]
        for chunk in worker["chunks"]
    )
    assert bool(result.converged.all())


def test_batch_executor_resizes_pool_for_preserved_resident_batches(tmp_path):
    systems = [
        Atoms("H", positions=[[0.8 - 0.05 * index, 0.0, 0.0]])
        for index in range(8)
    ]
    config = AutoSchedulerConfig(
        cache_path=tmp_path / "preserve-resident-cache.json",
        cache_enabled=False,
        max_batch_size=4,
        multi_gpu_dispatch_policy="preserve_resident",
        multi_gpu_process_cpu_threads=1,
    )
    with BatchExecutor(
        ExecutorQuadraticCalculator(),
        devices=["cpu:0", "cpu:1"],
        auto_config=config,
        startup_timeout_seconds=30.0,
        run_timeout_seconds=30.0,
    ) as executor:
        first = executor.relax(
            systems[:4],
            optimizer="fire",
            fmax=1e-4,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )
        assert len(executor.worker_pids) == 1
        assert executor.worker_generation == 1

        second = executor.relax(
            systems,
            optimizer="fire",
            fmax=1e-4,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )
        assert len(executor.worker_pids) == 2
        assert executor.worker_generation == 2

    first_schedule = first.metadata["scheduling"]
    second_schedule = second.metadata["scheduling"]
    assert bool(first.converged.all())
    assert bool(second.converged.all())
    assert first_schedule["multi_gpu_dispatch_policy"] == "preserve_resident"
    assert first_schedule["active_gpu_count"] == 1
    assert first_schedule["execution_chunk_count"] == 1
    assert not first_schedule["worker_generation_restarted"]
    assert second_schedule["active_gpu_count"] == 2
    assert second_schedule["execution_chunk_count"] == 2
    assert second_schedule["worker_generation_restarted"]
    assert [
        atoms.get_chemical_formula() for atoms in second.structures
    ] == ["H"] * len(systems)


def test_batch_executor_does_not_require_timing_policy_cache(tmp_path):
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
        result = executor.relax(
            systems,
            fmax=1e-4,
            max_steps=500,
            dt_start=0.05,
            dt_max=0.5,
        )
        assert bool(result.converged.all())
        assert executor.started
        assert result.metadata["scheduling"]["optimization_pilot_runs"] == 0
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


def test_batch_executor_rejects_invalid_shutdown_timeout():
    with pytest.raises(ValueError, match="shutdown_timeout_seconds"):
        BatchExecutor(
            ExecutorQuadraticCalculator(),
            devices=["cpu:0"],
            shutdown_timeout_seconds=0.0,
        )


def test_manifest_prefetch_depth_must_be_non_negative():
    with pytest.raises(
        ValueError,
        match="manifest_prefetch_chunks_per_worker",
    ):
        AutoSchedulerConfig(manifest_prefetch_chunks_per_worker=-1)
