from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import yaml

PROJECT = Path("research/project")


def _load_yaml(name: str):
    return yaml.safe_load((PROJECT / name).read_text(encoding="utf-8"))


def test_evidence_registry_covers_every_experiment_once():
    with (PROJECT / "evidence_registry.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    registered = [row["experiment"] for row in rows]
    experiment_dirs = sorted(
        path.name for path in Path("experiments").iterdir() if path.is_dir()
    )
    assert len(registered) == len(set(registered))
    assert sorted(registered) == experiment_dirs

    assert {row["evidence_status"] for row in rows} <= {
        "accepted_v1_prior",
        "validated_component",
        "benchmark_reference",
        "negative_result",
        "mixed_result",
        "superseded",
        "candidate",
    }
    assert {row["transfer_use"] for row in rows} <= {
        "mechanism_prior",
        "frozen_baseline",
        "action_exclusion",
        "protocol_only",
    }


def test_v1_baseline_binds_the_shipped_refill_policy():
    baseline = _load_yaml("baseline_v1.yaml")
    policy = baseline["provenance"]["refill_policy"]
    payload = Path(policy["path"]).read_bytes()

    assert hashlib.sha256(payload).hexdigest() == policy["sha256"]
    assert baseline["objective"]["memory_safety_fraction"] == 0.85
    assert not baseline["automatic_decisions"]["multi_device"]["refill"]
    assert not baseline["automatic_decisions"]["external_baseline_dispatch"][
        "automatic_mps_fallback"
    ]


def test_chemical_transfer_protocol_uses_parent_level_splits():
    protocol = _load_yaml("chemical_transfer.yaml")
    split = protocol["parent_level_split_per_domain"]

    assert protocol["parent_selection"]["target_per_domain"] == 200
    assert split["training"] + split["validation"] + split["test"]["total"] == 200
    assert (
        split["test"]["interpolation"]
        + split["test"]["chemical_cluster_ood"]
        == split["test"]["total"]
    )
    assert split["descendants_must_remain_with_parent"]
    assert not protocol["fairness_contract"][
        "old_experiment_timing_labels_allowed_for_fit"
    ]
    assert not protocol["planner_modes"]["zero_shot_transfer"][
        "separate_timing_pilot"
    ]
