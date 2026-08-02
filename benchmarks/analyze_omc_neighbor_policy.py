#!/usr/bin/env python3
"""Summarize the frozen OMC neighbor-policy experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def summarize_rebuild_directory(directory: Path) -> dict[str, Any]:
    points = []
    topology_passed = True
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for family in document["families"]:
            for point in family["points"]:
                topology_passed &= all(
                    method["exact_ordered_vs_matscipy"]
                    for method in point["methods"].values()
                )
                auto = point["methods"]["auto"]
                points.append(
                    {
                        "family": family["family"],
                        "cutoff_A": point["candidate_cutoff_A"],
                        "rebuilt_systems": point["rebuilt_systems"],
                        "auto_backend": auto["resolved_backend"],
                        "fastest_backend": point["fastest_explicit_backend"],
                        "regret": auto["regret_vs_fastest"],
                        "selection_seconds": auto["median_selection_seconds"],
                    }
                )
    regrets = [point["regret"] for point in points]
    selection = [point["selection_seconds"] for point in points]
    return {
        "directory": str(directory),
        "families": sorted({point["family"] for point in points}),
        "point_count": len(points),
        "topology_passed": topology_passed,
        "exact_fastest_matches": sum(
            point["auto_backend"] == point["fastest_backend"] for point in points
        ),
        "selected_backends": dict(Counter(point["auto_backend"] for point in points)),
        "median_regret": statistics.median(regrets),
        "mean_regret": statistics.mean(regrets),
        "p90_regret": _percentile(regrets, 0.90),
        "p95_regret": _percentile(regrets, 0.95),
        "maximum_regret": max(regrets),
        "median_selection_seconds": statistics.median(selection),
    }


def summarize_integrated_directory(directory: Path) -> dict[str, Any]:
    points = []
    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        reference = next(iter(document["methods"]))
        methods = document["methods"]
        fastest = min(methods, key=lambda name: methods[name]["median_wall_seconds"])
        auto = methods["auto"]
        differences = auto[f"validation_vs_{reference}"]
        points.append(
            {
                "workload": path.stem,
                "reference": reference,
                "fastest_backend": fastest,
                "auto_seconds": auto["median_wall_seconds"],
                "fastest_seconds": methods[fastest]["median_wall_seconds"],
                "auto_regret": (
                    auto["median_wall_seconds"] / methods[fastest]["median_wall_seconds"] - 1.0
                ),
                "auto_speedup_vs_matscipy": (
                    methods["matscipy"]["median_wall_seconds"]
                    / auto["median_wall_seconds"]
                ),
                "maximum_absolute_endpoint_difference": max(
                    differences.values(),
                    default=0.0,
                ),
                "max_abs_total_energy_drift_eV_per_atom": auto.get(
                    "max_abs_total_energy_drift_eV_per_atom"
                ),
                "auto_peak_reserved_bytes": auto["peak_reserved_bytes"],
            }
        )
    return {
        "directory": str(directory),
        "point_count": len(points),
        "all_endpoints_exact": all(
            point["maximum_absolute_endpoint_difference"] == 0.0 for point in points
        ),
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments/omc-csp-neighbor-policy-v2"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = args.experiment_root / "results"
    summary = {
        "schema_version": 1,
        "negative_tensor_selector": {
            "baseline": summarize_rebuild_directory(results / "baseline"),
            "tensor_selector": summarize_rebuild_directory(results / "tensor-selector"),
        },
        "policy_validation": {
            "old_policy": summarize_rebuild_directory(results / "heldout"),
            "candidate_graph_policy": summarize_rebuild_directory(
                results / "policy-v2-heldout"
            ),
        },
        "integrated_replicated": summarize_integrated_directory(
            results / "integrated-policy-v2"
        ),
        "integrated_unique": summarize_integrated_directory(
            results / "integrated-unique-policy-v2"
        ),
    }
    output = args.output or args.experiment_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
