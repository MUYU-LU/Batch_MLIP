from benchmarks.summarize_fixed_slot_refill import endpoint_comparison


def _payload(step, energy, position):
    return {
        "records": [
            {
                "source": "job",
                "converged": True,
                "steps": step,
                "energy_eV": energy,
                "positions_A": [[position, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "cell_A": [[1.0, 0.0, 0.0]] * 3,
            }
        ]
    }


def test_endpoint_comparison_normalizes_energy_by_atom_count():
    result = endpoint_comparison(
        _payload(10, 4.0, 0.0),
        _payload(12, 5.0, 0.2),
    )

    assert result["convergence_mismatches"] == 0
    assert result["step_mismatches"] == 1
    assert result["max_step_difference"] == 2
    assert result["max_energy_error_eV_per_atom"] == 0.5
    assert result["max_position_rmsd_A"] > 0.0
