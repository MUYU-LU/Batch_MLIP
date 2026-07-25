from benchmarks.summarize_application_mechanisms import (
    _add_atom_throughput,
    _effect,
    _model_key,
    build_summary,
)


def _row(model, task, method, wall, memory=10.0, evaluations=100):
    return {
        "model": model,
        "task": task,
        "method": method,
        "wall_time_s": wall,
        "peak_reserved_GB": memory,
        "peak_device_GB": memory,
        "model_evaluations": evaluations,
        "speedup_vs_mps32": 100.0 / wall,
        "device_memory_fraction_vs_mps32": memory / 20.0,
    }


def test_effect_uses_five_percent_practical_gate():
    assert _effect(1.06) == "wins"
    assert _effect(1.04) == "parity"
    assert _effect(0.96) == "parity"
    assert _effect(0.94) == "loses"


def test_model_names_are_normalized_across_tensor_and_mps_schemas():
    assert _model_key("MACE-OFF-Small") == "mace"
    assert _model_key("AtomBit") == "atombit"


def test_atom_throughput_uses_atom_steps_for_nve(tmp_path):
    manifest = {
        "jobs": [{"atom_count": 10}, {"atom_count": 20}],
        "metadata": {"measured_steps": 4},
    }
    (tmp_path / "nve.json").write_text(__import__("json").dumps(manifest))
    rows = [
        {
            "task": "fixed_horizon_nve",
            "workload_id": "nve",
            "wall_time_s": 2.0,
        }
    ]

    _add_atom_throughput(rows, tmp_path)

    assert rows[0]["atom_throughput_per_s"] == 60.0
    assert rows[0]["atom_throughput_unit"] == "atom_steps_per_second"


def test_summary_attributes_refill_separately_from_compaction():
    rows = []
    for model in ("atombit", "mace"):
        rows.extend(
            [
                _row(model, "static_evaluation", "tensor", 50.0),
                _row(model, "static_evaluation", "mps32", 100.0, 20.0),
                _row(model, "fixed_horizon_nve", "tensor", 80.0),
                _row(model, "fixed_horizon_nve", "mps32", 100.0, 20.0),
                _row(
                    model,
                    "variable_horizon_optimization",
                    "active_drain",
                    100.0,
                    evaluations=200,
                ),
                _row(
                    model,
                    "variable_horizon_optimization",
                    "active_refill",
                    110.0,
                    memory=18.0,
                    evaluations=120,
                ),
                _row(
                    model,
                    "variable_horizon_optimization",
                    "mps32",
                    100.0,
                    20.0,
                ),
            ]
        )

    result = build_summary(rows)
    effects = {
        (item["model"], item["mechanism"]): item
        for item in result["mechanism_effects"]
    }
    refill = effects[("atombit", "active_refill_over_active_drain")]
    assert refill["decision"] == "loses"
    assert refill["model_evaluation_ratio_vs_active_drain"] == 0.6
    assert refill["reserved_memory_ratio_vs_active_drain"] == 1.8
