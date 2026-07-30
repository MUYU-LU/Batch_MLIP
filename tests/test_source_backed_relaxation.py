from __future__ import annotations

import hashlib
from dataclasses import replace

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
    HardwareBoundCostModel,
    HardwareCapacityDecision,
    HardwareCapacityPolicy,
    HardwareCostProfile,
    LayeredCostCoefficients,
    planning_profile_from_manifest,
    relax,
    relax_manifest,
)
from batch_mlip.interfaces.sources import (
    AseManifestStructureProvider,
    AsyncStructureMaterializer,
    StructureMaterializer,
    select_manifest_loader_processes,
)
from batch_mlip.workloads import WorkloadJob, WorkloadManifest


class SourceQuadraticCalculator(BatchCalculator):
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


def _workload(tmp_path):
    systems = [
        Atoms(symbol, positions=[[position, 0.0, 0.0]])
        for symbol, position in zip(
            ("H", "He", "Li", "Be"),
            (0.8, -0.6, 0.4, -0.2),
            strict=True,
        )
    ]
    jobs = []
    for index, atoms in enumerate(systems):
        relative = f"candidate-{index}.extxyz"
        path = tmp_path / relative
        write(path, atoms)
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        jobs.append(
            WorkloadJob(
                system_id=f"system-{index}",
                group_id="test",
                duplicate_group=f"system-{index}",
                order=index,
                dataset_id="test",
                source_path=relative,
                source_sha256=source_sha,
                normalized_structure_sha256=hashlib.sha256(
                    f"normalized-{index}".encode()
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
        workload_id="source-backed-test",
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
        model_id="quadratic-test",
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


def _config(tmp_path) -> AutoSchedulerConfig:
    return AutoSchedulerConfig(
        cache_path=tmp_path / "source-cache.json",
        cache_enabled=False,
        max_batch_size=2,
        multi_gpu_worker_backend="process",
        multi_gpu_process_cpu_threads=1,
        multi_gpu_target_chunks_per_device=2,
    )


def test_manifest_relaxation_matches_eager_plan_and_results(tmp_path):
    systems, manifest, profile = _workload(tmp_path)
    options = {
        "fmax": 1e-5,
        "max_steps": 500,
        "dt_start": 0.05,
        "dt_max": 0.5,
    }
    eager = relax(
        systems,
        SourceQuadraticCalculator(),
        optimizer="fire",
        scheduling="auto",
        devices=["cpu:0", "cpu:1"],
        auto_config=_config(tmp_path),
        **options,
    )
    lazy = relax_manifest(
        manifest,
        tmp_path,
        profile,
        SourceQuadraticCalculator(),
        optimizer="fire",
        devices=["cpu:0", "cpu:1"],
        auto_config=_config(tmp_path),
        **options,
    )

    eager_schedule = eager.metadata["scheduling"]
    lazy_schedule = lazy.metadata["scheduling"]
    assert eager_schedule["planned_chunks"] == lazy_schedule["planned_chunks"]
    materialization = lazy_schedule["structure_materialization"]
    assert materialization["mode"] == "manifest_lazy_worker"
    assert materialization["parent_system_count"] == 0
    assert materialization["worker_system_count"] == 4
    assert materialization["worker_seconds"] >= 0.0
    assert eager_schedule["structure_materialization"]["mode"] == (
        "eager_in_memory"
    )
    assert eager_schedule["structure_materialization"]["parent_system_count"] == 4
    assert lazy_schedule["capacity_planning"]["mode"] == (
        "representative_probe_fallback"
    )
    assert all(
        chunk["materialization_mode"] == "manifest_lazy_worker"
        for worker in lazy_schedule["workers"]
        for chunk in worker["chunks"]
    )
    assert bool(eager.converged.all())
    assert bool(lazy.converged.all())
    assert [atoms.get_chemical_formula() for atoms in lazy.structures] == [
        "H",
        "He",
        "Li",
        "Be",
    ]
    torch.testing.assert_close(eager.evaluation.energy, lazy.evaluation.energy)
    torch.testing.assert_close(eager.evaluation.forces, lazy.evaluation.forces)
    np.testing.assert_allclose(
        eager.state.positions.cpu(),
        lazy.state.positions.cpu(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        eager.state.cells.cpu(),
        lazy.state.cells.cpu(),
        atol=1e-12,
    )


def test_manifest_relaxation_rejects_sidecar_contract_mismatch(tmp_path):
    _, manifest, profile = _workload(tmp_path)
    changed = replace(
        profile,
        workload_manifest_sha256="f" * 64,
        profile_sha256="",
    ).seal()
    with pytest.raises(ValueError, match="manifest hash"):
        relax_manifest(
            manifest,
            tmp_path,
            changed,
            SourceQuadraticCalculator(),
            optimizer="fire",
            devices=["cpu:0"],
            auto_config=_config(tmp_path),
        )


def test_parallel_manifest_materializer_preserves_order_and_identity(tmp_path):
    _, manifest, _ = _workload(tmp_path)
    provider = AseManifestStructureProvider.from_manifest(
        manifest,
        tmp_path,
    )

    with StructureMaterializer(provider, process_count=2) as materializer:
        systems = materializer.materialize((3, 1, 2))
        assert materializer.parallel

    assert [atoms.info["batch_mlip_system_id"] for atoms in systems] == [
        "system-3",
        "system-1",
        "system-2",
    ]
    assert [atoms.get_chemical_formula() for atoms in systems] == [
        "Be",
        "He",
        "Li",
    ]


def test_async_manifest_materializer_reuses_pool_and_preserves_order(tmp_path):
    _, manifest, _ = _workload(tmp_path)
    provider = AseManifestStructureProvider.from_manifest(
        manifest,
        tmp_path,
    )

    with AsyncStructureMaterializer(process_count=2) as materializer:
        first = materializer.submit(provider, (3, 1))
        second = materializer.submit(provider, (2, 0))
        first_systems, first_metadata = materializer.resolve(first)
        second_systems, second_metadata = materializer.resolve(second)

    assert [atoms.info["batch_mlip_system_id"] for atoms in first_systems] == [
        "system-3",
        "system-1",
    ]
    assert [atoms.info["batch_mlip_system_id"] for atoms in second_systems] == [
        "system-2",
        "system-0",
    ]
    assert first_metadata["process_count"] == 2
    assert second_metadata["dispatch_wait_seconds"] >= 0.0


def test_manifest_loader_process_count_must_be_positive():
    with pytest.raises(ValueError, match="manifest_loader_processes"):
        AutoSchedulerConfig(manifest_loader_processes=0)


def test_manifest_loader_auto_policy_selects_processes_only_above_all_gates():
    selected = select_manifest_loader_processes(
        [120] * 2048,
        active_worker_count=7,
        available_cpu_count=128,
    )
    small_pool = select_manifest_loader_processes(
        [500] * 512,
        active_worker_count=7,
        available_cpu_count=128,
    )
    light_pool = select_manifest_loader_processes(
        [77] * 2048,
        active_worker_count=7,
        available_cpu_count=128,
    )
    cpu_constrained = select_manifest_loader_processes(
        [120] * 2048,
        active_worker_count=7,
        available_cpu_count=32,
    )

    assert selected.process_count == 4
    assert selected.atoms_per_worker > 32_000
    assert selected.required_cpu_count == 35
    assert small_pool.process_count == 1
    assert light_pool.process_count == 1
    assert cpu_constrained.process_count == 1


def test_manifest_loader_explicit_override_and_non_manifest_fallback():
    explicit = select_manifest_loader_processes(
        [1] * 8,
        active_worker_count=2,
        requested=3,
        available_cpu_count=4,
    )
    eager = select_manifest_loader_processes(
        [500] * 2048,
        active_worker_count=2,
        requested=4,
        available_cpu_count=128,
        manifest_backed=False,
    )

    assert explicit.process_count == 3
    assert explicit.reason == "explicit manifest-loader process count"
    assert eager.process_count == 1
    assert "non-manifest" in eager.reason


def _synthetic_capacity_policy() -> HardwareCapacityPolicy:
    model = HardwareBoundCostModel(
        contract_id="source-backed-test-capacity-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cpu",
            device_name="test",
            total_memory_bytes=1_000_000,
            memory_safety_fraction=0.85,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(
            fixed=100.0,
            per_atom=10.0,
            per_dense_state_element=1.0,
        ),
    )
    policy = HardwareCapacityPolicy(
        policy_id="source-backed-test-v1",
        source_calibration_sha256="a" * 64,
        model_name="peak_reserved_bytes",
        contract={},
        model=model,
        policy_sha256="",
    )
    return replace(policy, policy_sha256=policy.calculate_sha256())


def test_manifest_relaxation_uses_matched_offline_capacity_without_probe(
    tmp_path,
    monkeypatch,
):
    _, manifest, profile = _workload(tmp_path)
    policy = _synthetic_capacity_policy()
    monkeypatch.setattr(
        "batch_mlip.interfaces.api.select_hardware_capacity_policy",
        lambda *args, **kwargs: HardwareCapacityDecision(
            mode="offline_hardware_model",
            reason="test contract matched",
            policy=policy,
        ),
    )

    result = relax_manifest(
        manifest,
        tmp_path,
        profile,
        SourceQuadraticCalculator(),
        optimizer="fire",
        devices=["cpu:0", "cpu:1"],
        auto_config=_config(tmp_path),
        hardware_capacity_policy=policy,
        fmax=1e-5,
        max_steps=500,
        dt_start=0.05,
        dt_max=0.5,
    )

    scheduling = result.metadata["scheduling"]
    assert scheduling["capacity_planning"] == {
        "mode": "offline_hardware_model",
        "reason": "test contract matched",
        "policy_id": "source-backed-test-v1",
        "policy_sha256": policy.policy_sha256,
        "source_calibration_sha256": "a" * 64,
        "cost_model_contract_id": "source-backed-test-capacity-v1",
        "memory_model": "peak_reserved_bytes",
    }
    assert scheduling["probe"]["system_count"] == 0
    assert scheduling["probe"]["model_forward_count"] == 0
    assert scheduling["structure_materialization"]["parent_system_count"] == 0
    assert all(
        chunk["predicted_peak_bytes"] is not None
        for chunk in scheduling["planned_chunks"]
    )
    assert bool(result.converged.all())
