#!/usr/bin/env python3
"""Evaluate calibrated cost models on untouched OMC-CSP families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch_mlip import (
    LayeredCostFeatures,
    load_hardware_cost_model,
    read_planning_profile,
)

METRICS = (
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "seconds_per_evaluation",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = json.loads((args.raw_dir / "matrix.json").read_text())
    if matrix["status"] != "complete":
        raise ValueError("cannot summarize an incomplete final matrix")
    models = {
        metric: load_hardware_cost_model(
            args.calibration,
            model_name=metric,
        )
        for metric in METRICS
    }
    observations = []
    for point in matrix["points"]:
        raw = json.loads(
            (
                args.raw_dir / f"{point['observation_id']}.json"
            ).read_text()
        )
        profile = read_planning_profile(
            args.profiles_dir / f"{point['workload_id']}.json"
        )
        indices = tuple(int(index) for index in raw["workload_indices"])
        features = LayeredCostFeatures.from_profiles(
            profile.systems[index] for index in indices
        )
        metrics = {}
        for metric, model in models.items():
            measured = float(raw[metric])
            predicted = model.estimate_batch(
                profile.systems[index] for index in indices
            )
            metrics[metric] = {
                "measured": measured,
                "predicted": predicted,
                "predicted_to_measured": predicted / measured,
                "absolute_relative_error": abs(predicted - measured)
                / measured,
            }
        observations.append(
            {
                "observation_id": point["observation_id"],
                "family": point["family"],
                "scale": point["scale"],
                "batch_size": len(indices),
                "features": {
                    "batch_size": features.system_count,
                    "total_atoms": features.atom_count,
                    "total_active_edges": features.active_edge_count,
                    "total_candidate_edges": features.candidate_edge_count,
                    "total_dense_state_elements": (
                        features.dense_state_elements
                    ),
                },
                "metrics": metrics,
            }
        )
    summaries = {}
    for metric in METRICS:
        errors = [
            observation["metrics"][metric]["absolute_relative_error"]
            for observation in observations
        ]
        ratios = [
            observation["metrics"][metric]["predicted_to_measured"]
            for observation in observations
        ]
        summaries[metric] = {
            "count": len(errors),
            "mean_absolute_relative_error": sum(errors) / len(errors),
            "max_absolute_relative_error": max(errors),
            "minimum_predicted_to_measured": min(ratios),
            "maximum_predicted_to_measured": max(ratios),
        }
    output = {
        "schema_version": 1,
        "status": "complete",
        "calibration": str(args.calibration),
        "calibration_sha256": json.loads(
            args.calibration.read_text()
        )["calibration_sha256"],
        "families": sorted(
            {observation["family"] for observation in observations}
        ),
        "observations": observations,
        "error_summary": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
