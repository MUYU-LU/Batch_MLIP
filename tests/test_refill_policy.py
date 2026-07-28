from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from batch_mlip import (
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    BatchedBFGS,
    RefillPrediction,
    load_refill_policy,
    model_state_sha256,
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
    assert policy["validation"]["speed_predictions_correct"] == 9
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


def test_refill_policy_artifact_is_canonical_json():
    policy = load_refill_policy()
    serialized = json.dumps(policy, indent=2, sort_keys=True) + "\n"
    assert serialized == (
        Path("batch_mlip/planning/data/refill_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
