#!/usr/bin/env python3
"""Summarize matched fresh and persistent executor benchmark records."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _numbers(value: Any) -> Iterator[float]:
    if isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    else:
        yield float(value)


def _max_abs_difference(left: Any, right: Any) -> float:
    left_values = list(_numbers(left))
    right_values = list(_numbers(right))
    if len(left_values) != len(right_values):
        raise ValueError("tensor records have different flattened sizes")
    return max(
        (abs(a - b) for a, b in zip(left_values, right_values, strict=True)),
        default=0.0,
    )


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _timings(record: dict[str, Any]) -> dict[str, float]:
    calls = [float(call["wall_time_s"]) for call in record["call_records"]]
    return {
        "first_call_s": calls[0],
        "post_first_mean_s": _mean(calls[1:]) if len(calls) > 1 else calls[0],
        "fully_warm_last_call_s": calls[-1],
        "session_s": math.fsum(calls),
    }


def _numerical_comparison(
    fresh: dict[str, Any],
    persistent: dict[str, Any],
) -> dict[str, Any] | None:
    comparisons = []
    for call_index, (fresh_call, persistent_call) in enumerate(
        zip(fresh["call_records"], persistent["call_records"], strict=True)
    ):
        if (
            "final_tensors" not in fresh_call
            or "final_tensors" not in persistent_call
        ):
            return None
        fresh_tensors = fresh_call["final_tensors"]
        persistent_tensors = persistent_call["final_tensors"]
        comparisons.append(
            {
                "call": call_index + 1,
                "max_abs_positions_A": _max_abs_difference(
                    fresh_tensors["positions_A"],
                    persistent_tensors["positions_A"],
                ),
                "max_abs_cells_A": _max_abs_difference(
                    fresh_tensors["cells_A"],
                    persistent_tensors["cells_A"],
                ),
                "max_abs_energies_eV": _max_abs_difference(
                    fresh_tensors["energies_eV"],
                    persistent_tensors["energies_eV"],
                ),
                "max_abs_forces_eV_per_A": _max_abs_difference(
                    fresh_tensors["forces_eV_per_A"],
                    persistent_tensors["forces_eV_per_A"],
                ),
                "converged_steps_equal": (
                    fresh_call["converged_steps"]
                    == persistent_call["converged_steps"]
                ),
            }
        )
    return {"calls": comparisons}


def _summarize_pair(
    label: str,
    fresh_path: Path,
    persistent_path: Path,
) -> dict[str, Any]:
    fresh = json.loads(fresh_path.read_text())
    persistent = json.loads(persistent_path.read_text())
    for record in (fresh, persistent):
        record.setdefault(
            "optimizer_sequence",
            [record["optimizer"]] * len(record["call_records"]),
        )
    identity_fields = (
        "mlip",
        "workload_id",
        "jobs",
        "atoms",
        "resident_batch_size",
        "gpu_count",
        "max_steps",
        "optimizer_sequence",
    )
    mismatches = [
        field
        for field in identity_fields
        if fresh.get(field) != persistent.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"{label}: unmatched comparison fields: {', '.join(mismatches)}"
        )
    if len(fresh["call_records"]) != len(persistent["call_records"]):
        raise ValueError(f"{label}: fresh and persistent call counts differ")

    fresh_timings = _timings(fresh)
    persistent_timings = _timings(persistent)
    startup = persistent["call_records"][0]["schedule"].get(
        "worker_startup_seconds_this_call",
        0.0,
    )
    return {
        "label": label,
        "identity": {field: fresh.get(field) for field in identity_fields},
        "fresh": {
            **fresh_timings,
            "peak_reserved_GiB": max(
                call["peak_reserved_bytes"] for call in fresh["call_records"]
            )
            / 2**30,
        },
        "persistent": {
            **persistent_timings,
            "first_generation_startup_s": startup,
            "peak_reserved_GiB": max(
                call["peak_reserved_bytes"]
                for call in persistent["call_records"]
            )
            / 2**30,
        },
        "speedup_fresh_over_persistent": {
            key: fresh_timings[key] / persistent_timings[key]
            for key in fresh_timings
        },
        "numerical": _numerical_comparison(fresh, persistent),
        "source": {
            "fresh": str(fresh_path),
            "persistent": str(persistent_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "FRESH_JSON", "PERSISTENT_JSON"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [
        _summarize_pair(label, Path(fresh), Path(persistent))
        for label, fresh, persistent in args.pair
    ]
    output = {
        "schema_version": 1,
        "experiment": "persistent-batch-executor",
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
