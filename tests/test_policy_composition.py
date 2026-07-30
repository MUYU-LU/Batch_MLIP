from __future__ import annotations

import torch

from batch_mlip import (
    AtomBitBatchCalculator,
    AutoWorkloadBucket,
    AutoWorkloadPlan,
    BatchedBFGS,
    DeterministicMemoryProbe,
    DeterministicRelaxationChunk,
    DeterministicRelaxationPlan,
    SystemProfile,
    compose_relaxation_policy_manifest,
)
from batch_mlip.toy_models import QuadraticWellModel


def _calculator(*, skin: float = 0.5) -> AtomBitBatchCalculator:
    return AtomBitBatchCalculator(
        QuadraticWellModel(),
        cutoff=2.5,
        skin=skin,
        device="cpu",
        dtype=torch.float64,
    )


def _probe() -> DeterministicMemoryProbe:
    return DeterministicMemoryProbe(
        memory_budget_bytes=1_000,
        baseline_allocated_bytes=0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=100,
        probe_indices=(0,),
        probe_model_work=1,
        model_bytes_per_work=1.0,
    )


def _homogeneous_refill_plan() -> DeterministicRelaxationPlan:
    profiles = tuple(
        SystemProfile(index=index, atom_count=46, edge_count=400, dof_squared=147**2)
        for index in range(4)
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=(0, 1, 2, 3),
                mean_atom_count=46.0,
                mean_edge_count=400.0,
                mean_dof_squared=float(147**2),
                homogeneous_atom_count=True,
            ),
        ),
        profiling_seconds=0.0,
        fingerprint="test",
        fingerprint_fields={},
    )
    return DeterministicRelaxationPlan(
        workload=workload,
        probe=_probe(),
        chunks=(
            DeterministicRelaxationChunk(
                system_indices=(0, 1, 2, 3),
                bucket_index=0,
                predicted_peak_bytes=900,
                estimated_cost=4.0,
                resident_capacity=2,
                active_refill=True,
                refill_storage="slots",
                refill_prediction={
                    "mode": "refill",
                    "reason": "matched accepted evidence",
                    "matched_family": "H46",
                },
            ),
        ),
        memory_fraction=0.85,
        memory_growth_margin=1.1,
    )


def _mixed_plan() -> DeterministicRelaxationPlan:
    profiles = (
        SystemProfile(index=0, atom_count=46, edge_count=400, dof_squared=147**2),
        SystemProfile(index=1, atom_count=276, edge_count=4_000, dof_squared=837**2),
    )
    workload = AutoWorkloadPlan(
        profiles=profiles,
        buckets=(
            AutoWorkloadBucket(
                system_indices=(1,),
                mean_atom_count=276.0,
                mean_edge_count=4_000.0,
                mean_dof_squared=float(837**2),
                homogeneous_atom_count=True,
            ),
            AutoWorkloadBucket(
                system_indices=(0,),
                mean_atom_count=46.0,
                mean_edge_count=400.0,
                mean_dof_squared=float(147**2),
                homogeneous_atom_count=True,
            ),
        ),
        profiling_seconds=0.0,
        fingerprint="test",
        fingerprint_fields={},
    )
    chunks = tuple(
        DeterministicRelaxationChunk(
            system_indices=(profile.index,),
            bucket_index=bucket_index,
            predicted_peak_bytes=900,
            estimated_cost=float(profile.atom_count),
            resident_capacity=1,
        )
        for bucket_index, profile in enumerate(reversed(profiles))
    )
    return DeterministicRelaxationPlan(
        workload=workload,
        probe=_probe(),
        chunks=chunks,
        memory_fraction=0.85,
        memory_growth_margin=1.1,
    )


def test_manifest_composes_homogeneous_refill_csp_policy():
    manifest = compose_relaxation_policy_manifest(
        _homogeneous_refill_plan(),
        _calculator(),
        BatchedBFGS(),
        {"cell_filter": object(), "active_compaction": True},
        fully_periodic=True,
        available_devices=["cuda:0"],
        active_device_count=1,
        execution_chunk_sizes=[4],
        execution_resident_capacities=[2],
        work_stealing=False,
        observed_converged_steps=[4, 8, 12, -1],
    )

    assert manifest["task"]["kind"] == "periodic_variable_cell_relaxation"
    assert manifest["profile"]["structurally_mixed"] is False
    layers = manifest["profile"]["layers"]
    assert layers["general_structure"]["features"] == ["atom_count"]
    assert layers["mlip_graph"]["features"] == ["active_edge_count"]
    assert layers["task_auxiliary"]["features"] == [
        "generalized_dimension",
        "dense_state_elements",
        "dense_linear_algebra_work",
        "stress_required",
        "variable_cell",
        "cell_method",
        "cell_degrees_of_freedom",
        "state_dtype",
    ]
    assert layers["scalar_cost"]["universal"] is False
    assert layers["scalar_cost"]["legacy_projection_used"] is True
    assert manifest["outer_scheduler"]["pool_regime"] == (
        "multiple_resident_waves"
    )
    assert manifest["outer_scheduler"]["resident_wave_count"] == 2
    assert manifest["inner_scheduler"]["resident_capacities"] == [2]
    assert manifest["inner_scheduler"]["graph"] == {
        "cutoff_A": 2.5,
        "skin_A": 0.5,
        "cache_enabled": True,
        "neighbor_backend": "auto",
        "backend_resolution": "adaptive_per_rebuild",
        "selection_source": "calculator_contract",
    }
    assert manifest["inner_scheduler"]["queue_policy"] == "refill"
    assert manifest["inner_scheduler"]["refill"]["evidence_source"] == (
        "exact_offline_evidence"
    )
    observed = manifest["profile"]["duration_variation"]["observed"]
    assert observed["converged_systems"] == 3
    assert observed["unconverged_systems"] == 1
    assert observed["converged_step"]["mean"] == 8.0


def test_manifest_composes_mixed_multigpu_drain_policy():
    manifest = compose_relaxation_policy_manifest(
        _mixed_plan(),
        _calculator(skin=0.0),
        BatchedBFGS(),
        {"cell_filter": object(), "active_compaction": True},
        fully_periodic=True,
        available_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        active_device_count=2,
        execution_chunk_sizes=[1, 1],
        execution_resident_capacities=[1, 1],
        work_stealing=True,
        refill_fallback_reasons=[
            "multi-GPU refill has no accepted scientific policy"
        ],
    )

    outer = manifest["outer_scheduler"]
    inner = manifest["inner_scheduler"]
    assert manifest["profile"]["structurally_mixed"]
    assert outer["system_mix"] == "mixed"
    assert outer["cost_bucket_count"] == 2
    assert outer["active_device_count"] == 2
    assert outer["assignment"] == "largest_cost_first_work_stealing"
    assert inner["graph"]["cache_enabled"] is False
    assert inner["queue_policy"] == "active_drain"
    assert inner["refill"] == {
        "enabled": False,
        "evidence_source": "policy_exclusion",
        "fallback_reasons": [
            "multi-GPU refill has no accepted scientific policy"
        ],
    }
