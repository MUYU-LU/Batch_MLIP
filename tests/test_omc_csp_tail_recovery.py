from __future__ import annotations

import pytest

from benchmarks.omc_csp_tail_recovery import (
    nonconverged_sources,
    recovery_task_cost,
    replace_nonconverged_records,
    run_ase_bfgs_tail_recovery,
)


def _record(source: str, converged: bool, energy: float) -> dict[str, object]:
    return {
        "source": source,
        "converged": converged,
        "energy_eV": energy,
    }


def test_tail_replacement_is_source_stable_and_unconditional() -> None:
    base = [
        _record("a", True, 1.0),
        _record("b", False, 2.0),
        _record("c", True, 3.0),
        _record("d", False, 4.0),
    ]
    recovery = [
        _record("d", False, 40.0),
        _record("b", True, 20.0),
    ]

    assert nonconverged_sources(base) == ("b", "d")
    merged = replace_nonconverged_records(base, recovery)

    assert [record["source"] for record in merged] == ["a", "b", "c", "d"]
    assert [record["energy_eV"] for record in merged] == [1.0, 20.0, 3.0, 40.0]
    assert merged[3]["converged"] is False


@pytest.mark.parametrize(
    "recovery, match",
    [
        ([_record("b", True, 2.0)], "missing"),
        (
            [_record("b", True, 2.0), _record("d", True, 4.0), _record("x", True, 5.0)],
            "unexpected",
        ),
        (
            [_record("b", True, 2.0), _record("b", True, 2.0)],
            "duplicate",
        ),
    ],
)
def test_tail_replacement_rejects_invalid_coverage(
    recovery: list[dict[str, object]],
    match: str,
) -> None:
    base = [
        _record("a", True, 1.0),
        _record("b", False, 2.0),
        _record("d", False, 4.0),
    ]
    with pytest.raises(ValueError, match=match):
        replace_nonconverged_records(base, recovery)


def test_tail_replacement_rejects_duplicate_base_ids() -> None:
    with pytest.raises(ValueError, match="base.*duplicate"):
        nonconverged_sources(
            [_record("a", True, 1.0), _record("a", False, 2.0)]
        )


def test_empty_tail_has_complete_zero_cost_telemetry(tmp_path) -> None:
    records, telemetry = run_ase_bfgs_tail_recovery(
        [],
        checkpoint=tmp_path / "unused.pt",
        dataset_dir=tmp_path,
        devices=[],
        cutoff=6.0,
        fmax=0.05,
        max_steps=500,
        max_step=0.2,
        alpha=70.0,
    )

    assert records == []
    assert telemetry["attempted_count"] == 0
    assert telemetry["total_seconds"] == 0.0
    assert telemetry["workers"] == []


def test_recovery_cost_includes_variable_cell_bfgs_dimension() -> None:
    expected = 16.0 * (3 * 46 + 9) ** 2 + 256.0 * 46 + 64.0 * 1200
    assert recovery_task_cost(atom_count=46, candidate_edges=1200) == expected
