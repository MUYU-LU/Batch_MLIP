from __future__ import annotations

import json
from pathlib import Path

from benchmarks.benchmark_omc_csp_scheduler_auto import (
    LINEAR_ALGEBRA_BACKENDS,
    TAIL_RECOVERY_MODES,
)
from benchmarks.build_omc_csp_scheduler_epoch import (
    DEVELOPMENT_SELECTED_EXTREME_POOL_FAMILIES,
    EXCLUDED_FAMILIES,
    SPLITS,
    VALIDATION_SELECTED_EXTREME_POOL_FAMILIES,
)
from benchmarks.run_omc_csp_scheduler_development import (
    EXPECTED_WORKLOADS,
    _selected_workloads,
)


def test_scheduler_epoch_split_is_disjoint_and_covers_compatible_families():
    families = [family for members in SPLITS.values() for family in members]

    assert len(families) == 22
    assert len(set(families)) == len(families)
    assert set(EXCLUDED_FAMILIES).isdisjoint(families)
    assert set(DEVELOPMENT_SELECTED_EXTREME_POOL_FAMILIES) <= set(SPLITS["development"])
    assert set(VALIDATION_SELECTED_EXTREME_POOL_FAMILIES) <= set(SPLITS["validation"])
    assert {"JAYDUI", "OBEQIX", "XULDUD", "rof-a", "rof-c"} == set(SPLITS["test"])


def test_scheduler_benchmark_exposes_all_existing_bfgs_backends():
    assert LINEAR_ALGEBRA_BACKENDS == (
        "auto",
        "cholesky",
        "grouped",
        "serial",
    )
    assert TAIL_RECOVERY_MODES == ("none", "ase_bfgs")


def test_scheduler_epoch_uses_frozen_split_matrices():
    root = Path(__file__).resolve().parents[1]
    index = json.loads(
        (
            root
            / "experiments"
            / "omc-csp-scheduler-epoch1"
            / "results"
            / "workload_index.json"
        ).read_text()
    )

    for split, expected in EXPECTED_WORKLOADS.items():
        assert len(_selected_workloads(index, split)) == expected
