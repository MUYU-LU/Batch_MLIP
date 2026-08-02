from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from ase import Atoms

from batch_mlip import (
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    BatchedBFGS,
    RefillPrediction,
    load_refill_policy,
    model_state_sha256,
    relax,
)
from batch_mlip.interfaces.api import _apply_offline_refill_policy
from batch_mlip.planning.auto import (
    AutoWorkloadBucket,
    AutoWorkloadPlan,
)
from batch_mlip.planning.deterministic import (
    DeterministicMemoryProbe,
    DeterministicRelaxationChunk,
    DeterministicRelaxationPlan,
)
from batch_mlip.planning.memory import SystemProfile
from batch_mlip.toy_models import QuadraticWellModel


def _calculator() -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        QuadraticWellModel(),
        cutoff=2.5,
        device="cpu",
        dtype=torch.float64,
    )


def _plan() -> DeterministicRelaxationPlan:
    profiles = tuple(
        SystemProfile(
            index=index,
            atom_count=1,
            edge_count=0,
            dof_squared=9,
        )
        for index in range(4)
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=(0, 1, 2, 3),
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
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=1000,
        baseline_allocated_bytes=0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=100,
        probe_indices=(0,),
        probe_model_work=1,
        model_bytes_per_work=1.0,
    )
    return DeterministicRelaxationPlan(
        workload=workload,
        probe=probe,
        chunks=(
            DeterministicRelaxationChunk(
                system_indices=(2, 0),
                bucket_index=0,
                predicted_peak_bytes=900,
                estimated_cost=2.0,
            ),
            DeterministicRelaxationChunk(
                system_indices=(3, 1),
                bucket_index=0,
                predicted_peak_bytes=850,
                estimated_cost=2.0,
            ),
        ),
        memory_fraction=0.85,
        memory_growth_margin=1.1,
    )


def _mixed_plan() -> DeterministicRelaxationPlan:
    profiles = tuple(
        SystemProfile(
            index=index,
            atom_count=1 if index % 2 == 0 else 2,
            edge_count=0 if index % 2 == 0 else 4,
            dof_squared=9 if index % 2 == 0 else 36,
        )
        for index in range(6)
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=(0, 1, 2, 3, 4, 5),
                mean_atom_count=1.5,
                mean_edge_count=2.0,
                mean_dof_squared=22.5,
                homogeneous_atom_count=False,
            ),
        ),
        profiling_seconds=0.0,
        fingerprint="mixed-test",
        fingerprint_fields={},
    )
    probe = DeterministicMemoryProbe(
        memory_budget_bytes=1000,
        baseline_allocated_bytes=0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=100,
        probe_indices=(0,),
        probe_model_work=1,
        model_bytes_per_work=1.0,
    )
    return DeterministicRelaxationPlan(
        workload=workload,
        probe=probe,
        chunks=tuple(
            DeterministicRelaxationChunk(
                system_indices=(start, start + 1),
                bucket_index=0,
                predicted_peak_bytes=900 - 10 * start,
                estimated_cost=2.0,
            )
            for start in (0, 2, 4)
        ),
        memory_fraction=0.85,
        memory_growth_margin=1.1,
    )


def test_model_state_hash_is_stable_and_value_sensitive():
    model = QuadraticWellModel()
    first = model_state_sha256(model)
    second = model_state_sha256(model)
    assert first == second

    changed = copy.deepcopy(model)
    with torch.no_grad():
        changed.k.add_(1.0)
    assert model_state_sha256(changed) != first


def test_shipped_refill_records_select_only_valid_evidence():
    policy = load_refill_policy()
    assert policy["validation"]["r256_family_holdout"][
        "speed_predictions_correct"
    ] == 9
    assert policy["validation"]["pool_transfer_family_holdout"] == {
        "scientific_gate_failures": 0,
        "speed_predictions_correct": 4,
        "speed_predictions_total": 4,
    }
    assert not policy["validation"]["multi_gpu_transfer"]["accepted"]
    for record in policy["records"]:
        if record["selected_mode"] != "refill":
            continue
        assert record["storage"] == "slots"
        assert record["measured_refill_speedup"] >= 1.05
        assert record["refill_peak_reserved_fraction"] <= 0.85
        assert record["endpoint_gate_passed"]
        assert record["all_jobs_converged"]


def test_offline_refill_policy_merges_memory_waves_in_submitted_order(
    monkeypatch,
):
    prediction = RefillPrediction(
        mode="refill",
        reason="test",
        policy_id="test",
        matched_family="test",
        predicted_speedup=1.2,
        evidence_split="fit",
    )
    monkeypatch.setattr(
        "batch_mlip.interfaces.api.predict_refill",
        lambda *args, **kwargs: prediction,
    )

    selected = _apply_offline_refill_policy(
        _plan(),
        _calculator(),
        BatchedBFGS(),
        {},
        AutoSchedulerConfig(),
    )

    assert len(selected.chunks) == 1
    chunk = selected.chunks[0]
    assert chunk.system_indices == (0, 1, 2, 3)
    assert chunk.resident_capacity == 2
    assert chunk.active_refill
    assert chunk.refill_storage == "slots"
    assert chunk.refill_prediction == prediction.to_dict()


def test_offline_refill_policy_preserves_waves_on_active_prediction(
    monkeypatch,
):
    prediction = RefillPrediction(mode="active", reason="fallback")
    monkeypatch.setattr(
        "batch_mlip.interfaces.api.predict_refill",
        lambda *args, **kwargs: prediction,
    )

    selected = _apply_offline_refill_policy(
        _plan(),
        _calculator(),
        BatchedBFGS(),
        {},
        AutoSchedulerConfig(),
    )

    assert [chunk.system_indices for chunk in selected.chunks] == [
        (2, 0),
        (3, 1),
    ]
    assert all(not chunk.active_refill for chunk in selected.chunks)
    assert all(
        chunk.refill_prediction == prediction.to_dict()
        for chunk in selected.chunks
    )


def test_offline_refill_policy_extracts_only_accepted_exact_shape_group(
    monkeypatch,
):
    def predict(*args, **kwargs):
        if kwargs["homogeneous_atom_count"] and kwargs["mean_atom_count"] == 1.0:
            return RefillPrediction(
                mode="refill",
                reason="validated exact-shape evidence",
                policy_id="test",
                matched_family="small-shape",
                predicted_speedup=1.2,
                evidence_split="fit",
            )
        return RefillPrediction(mode="active", reason="fallback")

    monkeypatch.setattr("batch_mlip.interfaces.api.predict_refill", predict)

    selected = _apply_offline_refill_policy(
        _mixed_plan(),
        _calculator(),
        BatchedBFGS(),
        {},
        AutoSchedulerConfig(),
    )

    assert [chunk.system_indices for chunk in selected.chunks] == [
        (0, 2, 4),
        (1,),
        (3,),
        (5,),
    ]
    refill = selected.chunks[0]
    assert refill.active_refill
    assert refill.resident_capacity == 1
    assert refill.refill_storage == "slots"
    assert refill.refill_prediction["parent_bucket_homogeneous"] is False
    assert refill.refill_prediction["shape_atom_count"] == 1
    assert refill.refill_prediction["shape_dof_squared"] == 9
    assert all(not chunk.active_refill for chunk in selected.chunks[1:])
    scheduled = [index for chunk in selected.chunks for index in chunk.system_indices]
    assert sorted(scheduled) == list(range(6))
    assert len(scheduled) == len(set(scheduled))


def test_offline_refill_policy_preserves_mixed_chunks_when_no_shape_is_accepted(
    monkeypatch,
):
    prediction = RefillPrediction(mode="active", reason="fallback")
    monkeypatch.setattr(
        "batch_mlip.interfaces.api.predict_refill",
        lambda *args, **kwargs: prediction,
    )

    original = _mixed_plan()
    selected = _apply_offline_refill_policy(
        original,
        _calculator(),
        BatchedBFGS(),
        {},
        AutoSchedulerConfig(),
    )

    assert [chunk.system_indices for chunk in selected.chunks] == [
        chunk.system_indices for chunk in original.chunks
    ]
    assert all(not chunk.active_refill for chunk in selected.chunks)


def test_offline_refill_policy_requires_exact_dof_shape_for_whole_bucket(
    monkeypatch,
):
    original = _plan()
    profiles = tuple(
        SystemProfile(
            index=profile.index,
            atom_count=profile.atom_count,
            edge_count=profile.edge_count,
            dof_squared=9 if profile.index % 2 == 0 else 16,
        )
        for profile in original.workload.profiles
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=original.workload.buckets,
        profiling_seconds=0.0,
        fingerprint="dof-mixed-test",
        fingerprint_fields={},
    )
    calls = []

    def predict(*args, **kwargs):
        calls.append(kwargs)
        return RefillPrediction(mode="active", reason="fallback")

    monkeypatch.setattr("batch_mlip.interfaces.api.predict_refill", predict)
    _apply_offline_refill_policy(
        DeterministicRelaxationPlan(
            workload=workload,
            probe=original.probe,
            chunks=(
                DeterministicRelaxationChunk(
                    system_indices=(0, 1),
                    bucket_index=0,
                    predicted_peak_bytes=900,
                    estimated_cost=2.0,
                ),
                DeterministicRelaxationChunk(
                    system_indices=(2, 3),
                    bucket_index=0,
                    predicted_peak_bytes=850,
                    estimated_cost=2.0,
                ),
            ),
            memory_fraction=original.memory_fraction,
            memory_growth_margin=original.memory_growth_margin,
        ),
        _calculator(),
        BatchedBFGS(),
        {},
        AutoSchedulerConfig(),
    )

    assert calls[0]["homogeneous_atom_count"] is False
    assert all(call["homogeneous_atom_count"] is True for call in calls[1:])


def test_manual_refill_arguments_preserve_single_batch_execution():
    result = relax(
        [
            Atoms("H", positions=[[0.2, 0.0, 0.0]]),
            Atoms("H", positions=[[0.1, 0.0, 0.0]]),
        ],
        _calculator(),
        optimizer="bfgs",
        refill_batch_size=1,
        fmax=1e-30,
        max_steps=0,
    )

    summary = result.metadata["scheduling"]["summary"]
    assert summary["strategy"] == "manual"
    assert summary["batch_mode"] == "refill"
    assert summary["resident_capacities"] == [1]


def test_refill_policy_artifact_is_canonical_json():
    policy = load_refill_policy()
    serialized = json.dumps(policy, indent=2, sort_keys=True) + "\n"
    assert serialized == (
        Path("batch_mlip/planning/data/refill_policy_v2.json").read_text(
            encoding="utf-8"
        )
    )


def test_pool_transfer_records_are_single_gpu_only():
    policy = load_refill_policy()
    transferred = [
        record
        for record in policy["records"]
        if record["pool_size"] in (128, 512)
    ]
    assert len(transferred) == 12
    assert {record["resident_capacity"] for record in transferred} == {
        32,
        64,
        128,
    }
    assert all(record["selected_mode"] == "refill" for record in transferred)
