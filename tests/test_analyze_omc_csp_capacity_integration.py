from __future__ import annotations

from benchmarks.analyze_omc_csp_capacity_integration import (
    physical_endpoint_comparison,
)


def _record(
    source: str,
    *,
    energy: float = 0.0,
    position: float = 0.0,
    converged: bool = True,
) -> dict:
    return {
        "source": source,
        "converged": converged,
        "energy_eV": energy,
        "max_force_eV_per_A": 0.01,
        "max_abs_stress_eV_per_A3": 0.001,
        "positions_A": [[position, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "cell_A": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        "stress_eV_per_A3": [
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.001],
        ],
        "steps": 5,
    }


def test_physical_endpoint_comparison_uses_per_atom_energy_and_source_ids():
    candidate = [_record("b"), _record("a", energy=1.5e-4)]
    reference = [_record("a"), _record("b")]

    result = physical_endpoint_comparison(candidate, reference)

    assert result["passed"]
    assert result["maximums"]["max_energy_error_eV_per_atom"] == 7.5e-5
    assert result["failure_counts"]["max_energy_error_eV_per_atom"] == 0


def test_physical_endpoint_comparison_reports_sparse_tolerance_failure():
    candidate = [_record("a"), _record("b", position=0.1)]
    reference = [_record("a"), _record("b")]

    result = physical_endpoint_comparison(candidate, reference)

    assert not result["passed"]
    assert result["failure_counts"]["max_position_rmsd_A"] == 1
    assert result["failed_checks"] == ["max_position_rmsd_A"]
