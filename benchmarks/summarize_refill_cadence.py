#!/usr/bin/env python3
"""Summarize immediate and periodic active-refill measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

FILE_PATTERN = re.compile(
    r"(?P<model>atombit|mace)_(?P<optimizer>bfgs|fire)_H46_"
    r"B(?P<batch_size>\d+)_K(?P<interval>\d+)\.json"
)


def _scheduler_seconds(payload: dict[str, Any]) -> float:
    phases = payload["runtime_profile"]["phases"]
    return sum(
        float(phases.get(name, {}).get("total_seconds", 0.0))
        for name in (
            "scheduler.refill_slot_swap",
            "scheduler.refill_repack",
            "scheduler.refill_arena",
        )
    )


def _refill_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in payload["runtime_profile"]["events"]
        if event.get("name") == "refill"
    ]


def _endpoint_difference(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, int, float]:
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    flags_match = all(
        left["converged"] == right["converged"]
        for left, right in zip(reference_records, candidate_records, strict=True)
    )
    max_step_difference = max(
        abs(int(left["steps"]) - int(right["steps"]))
        for left, right in zip(reference_records, candidate_records, strict=True)
    )
    max_energy_difference = max(
        abs(float(left["energy_eV"]) - float(right["energy_eV"]))
        / len(left["positions_A"])
        * 1000.0
        for left, right in zip(reference_records, candidate_records, strict=True)
    )
    return flags_match, max_step_difference, max_energy_difference


def summarize(paths: list[Path]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    for path in paths:
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected refill-cadence filename: {path}")
        key = (
            match.group("model"),
            match.group("optimizer"),
            int(match.group("batch_size")),
        )
        interval = int(match.group("interval"))
        groups.setdefault(key, {})[interval] = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for (model, optimizer, batch_size), points in sorted(groups.items()):
        if set(points) != {1, 2, 5}:
            raise ValueError(f"incomplete cadence group: {(model, optimizer, batch_size)}")
        reference = points[1]
        for interval, payload in sorted(points.items()):
            flags_match, step_difference, energy_difference = _endpoint_difference(
                reference,
                payload,
            )
            events = _refill_events(payload)
            optimizer_steps = int(payload["optimizer_steps_total"])
            frozen_graphs = (
                int(payload["graph_evaluations"])
                - optimizer_steps
                - int(payload["jobs"])
            )
            speedup = float(reference["timing_seconds"]) / float(payload["timing_seconds"])
            passes = (
                interval > 1
                and speedup >= 1.02
                and flags_match
                and step_difference == 0
            )
            rows.append(
                {
                    "model": model,
                    "optimizer": optimizer,
                    "batch_size": batch_size,
                    "refill_interval": interval,
                    "wall_time_s": payload["timing_seconds"],
                    "speedup_vs_interval_1": speedup,
                    "systems_per_second": payload["systems_per_second"],
                    "model_evaluations": payload["model_evaluations"],
                    "graph_evaluations": payload["graph_evaluations"],
                    "frozen_graph_evaluations": frozen_graphs,
                    "refill_events": len(events),
                    "mean_inserted_jobs_per_event": (
                        sum(int(event["inserted"]) for event in events) / len(events)
                        if events
                        else 0.0
                    ),
                    "scheduler_time_s": _scheduler_seconds(payload),
                    "peak_allocated_GiB": payload["peak_allocated_bytes"] / 2**30,
                    "peak_reserved_GiB": payload["peak_reserved_bytes"] / 2**30,
                    "converged_jobs": payload["converged"],
                    "convergence_flags_match_interval_1": flags_match,
                    "max_converged_step_difference": step_difference,
                    "max_endpoint_energy_difference_meV_per_atom": energy_difference,
                    "passes_predeclared_gate": passes,
                }
            )
    return {
        "schema_version": 1,
        "baseline_commit": "28e8aebd0c23e029db2d8ee8acaa7b87edfeca7b",
        "rows": rows,
        "stage_2_run": False,
        "decision": (
            "Keep refill_interval=1 as the default. No delayed candidate passed "
            "both the 1.02x speed gate and exact converged-step gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.inputs)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = summary["rows"]
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
