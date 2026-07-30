from __future__ import annotations

import json

import pytest

from batch_mlip import (
    AutoSchedulerConfig,
    load_hardware_capacity_policy,
    select_hardware_capacity_policy,
)


def test_packaged_capacity_policy_is_signed_and_uses_accepted_calibration():
    policy = load_hardware_capacity_policy()

    assert policy.policy_id == "omc-csp-atombit-h100-capacity-v1"
    assert policy.source_calibration_sha256 == (
        "d27468326e2b7f464b37c8739d3d3bd219425081bdb96dabf0f250928c78b297"
    )
    assert policy.model.contract_id == (
        "atombit-smooth-rms-fp32-bfgs-f64-frechet-h100-"
        "peak_reserved_bytes-v1"
    )
    assert policy.model.hardware.memory_safety_fraction == 0.85
    assert policy.contract["maximum_validated_batch_size"] == 512


def test_capacity_policy_rejects_modified_content(tmp_path):
    policy = load_hardware_capacity_policy()
    payload = policy.unsigned_dict()
    payload["policy_sha256"] = policy.policy_sha256
    payload["contract"]["cutoff_A"] = 5.0
    path = tmp_path / "modified-capacity-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash"):
        load_hardware_capacity_policy(path)


def test_offline_capacity_switch_requires_bool():
    with pytest.raises(TypeError, match="offline_hardware_capacity_enabled"):
        AutoSchedulerConfig(
            offline_hardware_capacity_enabled=1,  # type: ignore[arg-type]
        )


def test_capacity_policy_falls_back_below_validated_growth_margin():
    policy = load_hardware_capacity_policy()

    decision = select_hardware_capacity_policy(
        policy,
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,
        {},
        (),
        AutoSchedulerConfig(memory_growth_margin=1.0),
        allocator_policy="expandable_segments",
    )

    assert decision.mode == "representative_probe_fallback"
    assert decision.reason == (
        "memory growth margin is below the validated minimum"
    )
