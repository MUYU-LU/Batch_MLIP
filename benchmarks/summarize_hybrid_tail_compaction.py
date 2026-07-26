#!/usr/bin/env python3
"""Summarize hybrid tail-compaction and CUDA allocator measurements."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

H46_PATTERN = re.compile(
    r"(?P<model>atombit|mace)_(?P<optimizer>bfgs|fire)_H46_"
    r"B(?P<batch_size>64|128)_(?P<mode>immediate|tail75|tail50)\.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    paired = zip(reference["records"], candidate["records"], strict=True)
    comparisons = list(paired)
    flags_match = all(left["converged"] == right["converged"] for left, right in comparisons)
    max_step_difference = max(
        abs(int(left["steps"]) - int(right["steps"])) for left, right in comparisons
    )
    max_energy_difference = max(
        abs(float(left["energy_eV"]) - float(right["energy_eV"]))
        / len(left["positions_A"])
        * 1000.0
        for left, right in comparisons
    )
    return flags_match, max_step_difference, max_energy_difference


def _h46_rows(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for path in paths:
        match = H46_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected H46 filename: {path}")
        key = (
            match.group("model"),
            match.group("optimizer"),
            int(match.group("batch_size")),
        )
        groups.setdefault(key, {})[match.group("mode")] = _load(path)

    rows = []
    for (model, optimizer, batch_size), points in sorted(groups.items()):
        if set(points) != {"immediate", "tail75", "tail50"}:
            raise ValueError(f"incomplete H46 group: {(model, optimizer, batch_size)}")
        reference = points["immediate"]
        for mode in ("immediate", "tail75", "tail50"):
            payload = points[mode]
            flags_match, step_difference, energy_difference = _endpoint_difference(
                reference,
                payload,
            )
            events = _refill_events(payload)
            rows.append(
                {
                    "model": model,
                    "optimizer": optimizer,
                    "batch_size": batch_size,
                    "mode": mode,
                    "wall_time_s": payload["timing_seconds"],
                    "speedup_vs_immediate": (
                        float(reference["timing_seconds"])
                        / float(payload["timing_seconds"])
                    ),
                    "graph_evaluations": payload["graph_evaluations"],
                    "frozen_graph_evaluations": payload["frozen_graph_evaluations"],
                    "refill_events": len(events),
                    "tail_compaction_events": sum(
                        bool(event.get("tail_compaction")) for event in events
                    ),
                    "peak_allocated_GiB": payload["peak_allocated_bytes"] / 2**30,
                    "peak_reserved_GiB": payload["peak_reserved_bytes"] / 2**30,
                    "converged_jobs": payload["converged"],
                    "convergence_flags_match": flags_match,
                    "max_converged_step_difference": step_difference,
                    "max_endpoint_energy_difference_meV_per_atom": energy_difference,
                }
            )
    return rows


def _allocator_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        payload = _load(path)
        metrics = payload.get("allocator_metrics", {})
        rows.append(
            {
                "file": path.name,
                "model": payload["mlip"],
                "workload": payload["workload_id"],
                "batch_size": payload["batch_size"],
                "mode": (
                    "tail"
                    if payload.get("refill_tail_compaction_threshold") is not None
                    else "immediate"
                ),
                "requested_config": metrics.get("cuda_allocator_config"),
                "reported_backend": metrics.get("cuda_allocator_backend"),
                "wall_time_s": payload["timing_seconds"],
                "peak_allocated_GiB": payload["peak_allocated_bytes"] / 2**30,
                "peak_reserved_GiB": payload["peak_reserved_bytes"] / 2**30,
                "current_reserved_GiB": metrics.get("reserved_bytes_current", 0) / 2**30,
                "inactive_split_peak_GiB": (
                    metrics.get("inactive_split_bytes_peak", 0) / 2**30
                ),
                "allocation_retries": metrics.get("allocation_retries"),
                "out_of_memory_count": metrics.get("out_of_memory_count"),
                "effective_control": (
                    "effective" in path.name
                    or (
                        metrics.get("cuda_allocator_config") is None
                        and "native" in path.name
                    )
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h46-dir", type=Path, required=True)
    parser.add_argument("--allocator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    h46_rows = _h46_rows(sorted(args.h46_dir.glob("*.json")))
    allocator_rows = _allocator_rows(sorted(args.allocator_dir.glob("*.json")))
    summary = {
        "schema_version": 1,
        "baseline_commit": "e0b6c561977532d4a61459b83a250dd6f508bd9d",
        "h46_rows": h46_rows,
        "allocator_rows": allocator_rows,
        "decision": {
            "tail_compaction": (
                "Keep immediate tail compaction as the default. The only initial "
                "candidate did not reproduce in allocator-controlled validation."
            ),
            "allocator": (
                "Use effective expandable segments for AtomBit variable-cell "
                "BFGS; retain native allocation for MACE."
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "h46.csv", h46_rows)
    _write_csv(args.output_dir / "allocator.csv", allocator_rows)


if __name__ == "__main__":
    main()
