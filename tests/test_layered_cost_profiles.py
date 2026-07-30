from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from ase import Atoms

from batch_mlip import (
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    AutoWorkloadBucket,
    AutoWorkloadPlan,
    BatchedBFGS,
    FrechetCellFilter,
    HardwareBoundCostModel,
    HardwareCalibratedBatchPlanner,
    HardwareCostProfile,
    LayeredCalibrationObservation,
    LayeredCostCoefficients,
    LayeredCostFeatures,
    SystemProfile,
    TaskAuxiliaryCostProfile,
    fit_hardware_cost_model,
    plan_hardware_calibrated_relaxation,
    planning_profile_from_manifest,
    profile_auto_workload,
    read_planning_profile,
    relax,
    structure_workload_sha256,
    summarize_calibration_error,
    write_planning_profile,
)
from batch_mlip.toy_models import PairHarmonicModel
from batch_mlip.workloads import (
    WorkloadJob,
    WorkloadManifest,
    topology_key,
)


def _manifest() -> WorkloadManifest:
    active_key = topology_key(6.0, 0.0)
    candidate_key = topology_key(6.0, 0.5)
    jobs = tuple(
        WorkloadJob(
            system_id=f"OMC:{index}",
            group_id="family",
            duplicate_group=f"{index + 1:064x}",
            order=index,
            dataset_id="omc",
            source_path=f"family/candidate-{index}.cif",
            source_sha256=f"{index + 11:064x}",
            normalized_structure_sha256=f"{index + 1:064x}",
            frame_index=0,
            atom_count=atom_count,
            species=("C",) * atom_count,
            chemical_formula=f"C{atom_count}",
            pbc=(True, True, True),
            cell_A=(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0),
            volume_A3=1000.0,
            constraints=(),
            topology_edge_counts={
                active_key: active_edges,
                candidate_key: candidate_edges,
            },
        )
        for index, (atom_count, active_edges, candidate_edges) in enumerate(
            ((10, 100, 130), (20, 240, 300))
        )
    )
    return WorkloadManifest(
        workload_id="OPT-OMC-TEST-P2-v1",
        version=1,
        family="variable_horizon_closed",
        operation="optimization",
        cell_mode="variable",
        arrival_mode="closed",
        jobs=jobs,
        metadata={"application": "omc_csp"},
    ).seal()


def test_omc_bfgs_profile_separates_general_and_specific_costs(tmp_path):
    manifest = _manifest()
    profile = planning_profile_from_manifest(
        manifest,
        model_id="AtomBit-smooth-rms",
        cutoff_A=6.0,
        active_edge_key=topology_key(6.0, 0.0),
        candidate_edge_key=topology_key(6.0, 0.5),
        force_mode="autograd",
        model_dtype="torch.float32",
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=0.5,
        neighbor_backend="auto",
    )

    profile.verify()
    first = profile.systems[0]
    assert first.structure.atom_count == 10
    assert first.mlip_graph.active_edge_count == 100
    assert first.mlip_graph.cutoff_A == 6.0
    assert first.graph_execution.candidate_edge_count == 130
    assert first.graph_execution.skin_A == 0.5
    assert first.task_auxiliary.generalized_dimension == 39
    assert first.task_auxiliary.state_dtype == "calculator_state_dtype"
    assert first.task_auxiliary.dense_state_elements == 39**2
    assert first.task_auxiliary.dense_linear_algebra_work == 39**3

    path = tmp_path / "profile.json"
    write_planning_profile(path, profile)
    assert read_planning_profile(path) == profile

    payload = json.loads(path.read_text())
    payload["systems"][0]["structure"]["atom_count"] = 11
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content hash"):
        read_planning_profile(path)


def test_structure_identity_ignores_model_graph_and_task_metadata():
    manifest = _manifest()
    structure_hash = structure_workload_sha256(manifest)
    changed_job = replace(
        manifest.jobs[0],
        topology_edge_counts={
            topology_key(5.0, 0.0): 80,
            topology_key(5.0, 0.5): 110,
        },
    )
    changed = replace(
        manifest,
        jobs=(changed_job, manifest.jobs[1]),
        metadata={"application": "different_task"},
        manifest_sha256="",
    ).seal()

    assert changed.manifest_sha256 != manifest.manifest_sha256
    assert structure_workload_sha256(changed) == structure_hash


def test_scalar_cost_requires_explicit_hardware_bound_coefficients():
    profile = planning_profile_from_manifest(
        _manifest(),
        model_id="AtomBit-smooth-rms",
        cutoff_A=6.0,
        active_edge_key=topology_key(6.0, 0.0),
        candidate_edge_key=topology_key(6.0, 0.5),
        force_mode="autograd",
        model_dtype="torch.float32",
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=0.5,
        neighbor_backend="auto",
    ).systems[0]
    coefficients = LayeredCostCoefficients(
        fixed=1.0,
        per_atom=2.0,
        per_active_edge=3.0,
        per_candidate_edge=5.0,
        per_dense_state_element=7.0,
    )
    model = HardwareBoundCostModel(
        contract_id="atombit-bfgs-h100-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="NVIDIA H100 80GB HBM3",
            total_memory_bytes=85_059_715_072,
            memory_safety_fraction=0.85,
            device_count=1,
        ),
        coefficients=coefficients,
    )

    expected = 1 + 2 * 10 + 3 * 100 + 5 * 130 + 7 * 39**2
    assert model.estimate(profile) == expected


def test_batch_cost_charges_fixed_coefficient_once():
    systems = planning_profile_from_manifest(
        _manifest(),
        model_id="AtomBit-smooth-rms",
        cutoff_A=6.0,
        active_edge_key=topology_key(6.0, 0.0),
        candidate_edge_key=topology_key(6.0, 0.5),
        force_mode="autograd",
        model_dtype="torch.float32",
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=0.5,
        neighbor_backend="auto",
    ).systems
    model = HardwareBoundCostModel(
        contract_id="batch-intercept-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="test",
            total_memory_bytes=1000,
            memory_safety_fraction=0.85,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(fixed=100, per_atom=2),
    )

    assert model.estimate_batch(systems) == 100 + 2 * 30
    assert sum(model.estimate(system) for system in systems) == 200 + 2 * 30


def test_hardware_calibrated_planner_uses_layered_batch_intercept():
    systems = planning_profile_from_manifest(
        _manifest(),
        model_id="AtomBit-smooth-rms",
        cutoff_A=6.0,
        active_edge_key=topology_key(6.0, 0.0),
        candidate_edge_key=topology_key(6.0, 0.5),
        force_mode="autograd",
        model_dtype="torch.float32",
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=0.5,
        neighbor_backend="auto",
    ).systems
    model = HardwareBoundCostModel(
        contract_id="planner-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="test",
            total_memory_bytes=1000,
            memory_safety_fraction=0.5,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(fixed=100, per_atom=10),
    )
    planner = HardwareCalibratedBatchPlanner(model)

    plan = planner.plan_bound_profiles(systems)

    assert plan.memory_budget_bytes == 500
    assert len(plan.buckets) == 1
    assert plan.buckets[0].resident_capacity == 2
    assert plan.buckets[0].predicted_peak_bytes == 400


def test_hardware_calibrated_planner_applies_prediction_margin():
    systems = planning_profile_from_manifest(
        _manifest(),
        model_id="AtomBit-smooth-rms",
        cutoff_A=6.0,
        active_edge_key=topology_key(6.0, 0.0),
        candidate_edge_key=topology_key(6.0, 0.5),
        force_mode="autograd",
        model_dtype="torch.float32",
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
        skin_A=0.5,
        neighbor_backend="auto",
    ).systems
    model = HardwareBoundCostModel(
        contract_id="planner-margin-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="test",
            total_memory_bytes=1000,
            memory_safety_fraction=0.5,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(fixed=100, per_atom=10),
    )
    planner = HardwareCalibratedBatchPlanner(model, prediction_margin=1.10)

    plan = planner.plan_bound_profiles(systems)

    assert planner.estimate_profiles_bytes(plan.profiles) == 441
    assert plan.buckets[0].resident_capacity == 2
    assert plan.buckets[0].predicted_peak_bytes == 441


def test_hardware_calibrated_planner_rejects_invalid_prediction_margin():
    model = HardwareBoundCostModel(
        contract_id="planner-margin-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="test",
            total_memory_bytes=1000,
            memory_safety_fraction=0.5,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(fixed=100),
    )

    with pytest.raises(ValueError, match="prediction_margin"):
        HardwareCalibratedBatchPlanner(model, prediction_margin=0.99)


def test_hardware_calibrated_plan_preserves_outer_workload_buckets():
    profiles = tuple(
        SystemProfile.from_bound_cost(profile)
        for profile in planning_profile_from_manifest(
            _manifest(),
            model_id="AtomBit-smooth-rms",
            cutoff_A=6.0,
            active_edge_key=topology_key(6.0, 0.0),
            candidate_edge_key=topology_key(6.0, 0.5),
            force_mode="autograd",
            model_dtype="torch.float32",
            optimizer=BatchedBFGS(),
            variable_cell=True,
            cell_method=FrechetCellFilter(),
            skin_A=0.5,
            neighbor_backend="auto",
        ).systems
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=(1,),
                mean_atom_count=20.0,
                mean_edge_count=210.0,
                mean_dof_squared=4761.0,
                homogeneous_atom_count=True,
            ),
            AutoWorkloadBucket(
                system_indices=(0,),
                mean_atom_count=10.0,
                mean_edge_count=130.0,
                mean_dof_squared=1521.0,
                homogeneous_atom_count=True,
            ),
        ),
        profiling_seconds=0.0,
        fingerprint="test",
        fingerprint_fields={},
    )
    model = HardwareBoundCostModel(
        contract_id="planner-buckets-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cuda",
            device_name="test",
            total_memory_bytes=1000,
            memory_safety_fraction=0.85,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(fixed=100, per_atom=10),
    )

    plan = plan_hardware_calibrated_relaxation(
        workload,
        HardwareCalibratedBatchPlanner(model),
        memory_fraction=0.85,
    )

    assert [chunk.bucket_index for chunk in plan.chunks] == [0, 1]
    assert [chunk.system_indices for chunk in plan.chunks] == [(1,), (0,)]


def test_public_auto_relaxation_profiles_for_hardware_calibrated_planner():
    systems = [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            cell=[6.0, 6.0, 6.0],
            pbc=True,
        )
        for _ in range(2)
    ]
    calculator = AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        skin=0.5,
        device="cpu",
        dtype=torch.float64,
    )
    model = HardwareBoundCostModel(
        contract_id="public-auto-v1",
        metric="bytes",
        hardware=HardwareCostProfile(
            device_type="cpu",
            device_name="test",
            total_memory_bytes=1_000_000,
            memory_safety_fraction=0.85,
            device_count=1,
        ),
        coefficients=LayeredCostCoefficients(
            fixed=100,
            per_atom=10,
            per_candidate_edge=2,
            per_dense_state_element=1,
        ),
    )
    planner = HardwareCalibratedBatchPlanner(model)

    result = relax(
        systems,
        calculator,
        optimizer=BatchedBFGS(optimizer_dtype="float64"),
        scheduling="auto",
        planner=planner,
        cell_filter=FrechetCellFilter(),
        max_steps=1,
        fmax=1e-12,
    )

    assert result.metadata["scheduling"]["decision"] == (
        "whole_batch_predicted_to_fit"
    )
    assert result.model_evaluations == 2


def test_layered_hardware_calibration_recovers_synthetic_model():
    hardware = HardwareCostProfile(
        device_type="cuda",
        device_name="synthetic",
        total_memory_bytes=1000,
        memory_safety_fraction=0.85,
        device_count=1,
    )
    true = LayeredCostCoefficients(
        fixed=50.0,
        per_atom=2.0,
        per_candidate_edge=3.0,
        per_dense_state_element=5.0,
    )
    observations = []
    for index, (atoms, edges, dense) in enumerate(
        ((5, 10, 20), (10, 12, 40), (20, 30, 45), (30, 50, 80), (15, 40, 90))
    ):
        features = LayeredCostFeatures(
            system_count=1,
            atom_count=atoms,
            active_edge_count=edges // 2,
            candidate_edge_count=edges,
            linear_state_elements=0,
            dense_state_elements=dense,
            dense_linear_algebra_work=dense * 10,
        )
        observations.append(
            LayeredCalibrationObservation(
                observation_id=f"point-{index}",
                split="validation" if index == 4 else "fit",
                workload_id=f"workload-{index}",
                workload_manifest_sha256=f"{index + 1:064x}",
                planning_profile_sha256=f"{index + 11:064x}",
                features=features,
                measured_value=true.estimate_features(features),
            )
        )
    model = fit_hardware_cost_model(
        observations,
        contract_id="synthetic-v1",
        metric="bytes",
        hardware=hardware,
        coefficient_names=(
            "fixed",
            "per_atom",
            "per_candidate_edge",
            "per_dense_state_element",
        ),
    )
    validation = summarize_calibration_error(
        observations,
        model,
        split="validation",
    )

    assert validation.max_absolute_relative_error < 1e-6


def test_task_adapters_distinguish_static_md_and_variable_cell_bfgs():
    static = TaskAuxiliaryCostProfile.static_evaluation(
        index=0,
        atom_count=10,
        stress_required=False,
    )
    md = TaskAuxiliaryCostProfile.molecular_dynamics(
        index=0,
        atom_count=10,
        ensemble="npt_mtk",
        variable_cell=True,
        extended_state_elements=8,
    )
    bfgs = TaskAuxiliaryCostProfile.relaxation(
        index=0,
        atom_count=10,
        optimizer=BatchedBFGS(),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
    )

    assert static.optimizer_state_kind == "none"
    assert static.dense_state_elements == 0
    assert md.optimizer_state_kind == "linear"
    assert md.linear_state_elements == 38
    assert md.stress_required
    assert bfgs.optimizer_state_kind == "dense"
    assert bfgs.generalized_dimension == 39


def test_bfgs_task_adapter_preserves_explicit_optimizer_dtype():
    bfgs = TaskAuxiliaryCostProfile.relaxation(
        index=0,
        atom_count=64,
        optimizer=BatchedBFGS(optimizer_dtype="float64"),
        variable_cell=True,
        cell_method=FrechetCellFilter(),
    )

    assert bfgs.state_dtype == "torch.float64"
    assert bfgs.generalized_dimension == 3 * 64 + 9


def test_automatic_profiler_populates_all_cost_layers():
    systems = [
        Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            cell=[6.0, 6.0, 6.0],
            pbc=True,
        )
    ]
    calculator = AtomBitBatchCalculator(
        PairHarmonicModel(cutoff=2.0),
        cutoff=2.0,
        skin=0.5,
        device="cpu",
        dtype=torch.float64,
    )

    workload = profile_auto_workload(
        systems,
        calculator,
        BatchedBFGS(),
        {"cell_filter": object()},
        AutoSchedulerConfig(),
    )

    profile = workload.profiles[0]
    assert profile.bound_cost is not None
    assert profile.atom_count == profile.bound_cost.structure.atom_count
    assert profile.edge_count == (
        profile.bound_cost.graph_execution.candidate_edge_count
    )
    assert (
        profile.bound_cost.mlip_graph.active_edge_count
        <= profile.edge_count
    )
    assert profile.bound_cost.task_auxiliary.generalized_dimension == 15
    assert profile.dof_squared == 15**2
