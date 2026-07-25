from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "summarize_robustness_optimization.py"
)
SPEC = importlib.util.spec_from_file_location("robustness_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(source: str, atom_count: int = 2) -> dict:
    return {
        "source": source,
        "converged": True,
        "steps": 4,
        "energy_eV": 2.0,
        "max_force_eV_per_A": 0.04,
        "stress_eV_per_A3": [[0.0] * 3 for _ in range(3)],
        "positions_A": [[0.0] * 3 for _ in range(atom_count)],
        "cell_A": [[1.0, 0.0, 0.0]] * 3,
    }


def test_validate_matches_signed_jobs_independent_of_record_order() -> None:
    references = [_record("job-0"), _record("job-1", atom_count=4)]
    result = MODULE._validate(references, list(reversed(references)))

    assert result["passed"]
    assert result["max_energy_error_eV_per_atom"] == 0.0


def test_validate_uses_per_structure_atom_count_and_reports_failure() -> None:
    reference = _record("job-0", atom_count=4)
    candidate = copy.deepcopy(reference)
    candidate["energy_eV"] += 8e-4

    result = MODULE._validate([reference], [candidate])

    assert result["max_energy_error_eV_per_atom"] == pytest.approx(2e-4)
    assert result["failed_checks"] == ["max_energy_error_eV_per_atom"]
    assert not result["passed"]


def test_validate_rejects_different_signed_jobs() -> None:
    with pytest.raises(ValueError, match="signed job identifiers differ"):
        MODULE._validate([_record("job-0")], [_record("job-1")])


def test_converged_count_supports_old_and_new_artifacts() -> None:
    records = [_record("job-0"), {**_record("job-1"), "converged": False}]

    assert MODULE._converged({"records": records}) == 1
    assert MODULE._converged({"records": records, "converged": 2}) == 2
