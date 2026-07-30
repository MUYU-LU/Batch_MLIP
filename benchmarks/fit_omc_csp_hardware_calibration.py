#!/usr/bin/env python3
"""Fit signed layered memory and runtime models from the OMC-CSP matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from batch_mlip import (
    HardwareCostProfile,
    LayeredCalibrationObservation,
    LayeredCostFeatures,
    fit_hardware_cost_model,
    read_planning_profile,
    summarize_calibration_error,
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _model_dict(model) -> dict[str, Any]:
    return {
        "contract_id": model.contract_id,
        "metric": model.metric,
        "hardware": asdict(model.hardware),
        "coefficients": asdict(model.coefficients),
        "safety_factor": model.safety_factor,
    }


def _observations(
    matrix: dict[str, Any],
    *,
    raw_dir: Path,
    profiles_dir: Path,
    measured_key: str,
) -> list[LayeredCalibrationObservation]:
    observations = []
    for point in matrix["points"]:
        raw = json.loads(
            (raw_dir / f"{point['observation_id']}.json").read_text()
        )
        profile = read_planning_profile(
            profiles_dir / f"{point['workload_id']}.json"
        )
        batch_size = int(point["batch_size"])
        indices = tuple(
            int(index)
            for index in raw.get(
                "workload_indices",
                range(batch_size),
            )
        )
        if len(indices) != batch_size:
            raise ValueError(
                f"{point['observation_id']} measured index count differs "
                "from its matrix batch size"
            )
        features = LayeredCostFeatures.from_profiles(
            profile.systems[index] for index in indices
        )
        if measured_key == "seconds_per_evaluation":
            measured = raw["seconds_per_evaluation"]
        else:
            measured = raw[measured_key]
        observations.append(
            LayeredCalibrationObservation(
                observation_id=str(point["observation_id"]),
                split=str(point["split"]),
                workload_id=profile.workload_id,
                workload_manifest_sha256=(
                    profile.workload_manifest_sha256
                ),
                planning_profile_sha256=profile.profile_sha256,
                features=features,
                measured_value=float(measured),
            )
        )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        action="append",
        required=True,
        help="Raw matrix directory. Repeat to combine compatible matrices.",
    )
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-name", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--total-memory-bytes", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    args = parser.parse_args()

    matrices = [
        (
            raw_dir,
            json.loads((raw_dir / "matrix.json").read_text()),
        )
        for raw_dir in args.raw_dir
    ]
    for _, matrix in matrices:
        if matrix["status"] != "complete":
            raise ValueError("cannot fit an incomplete calibration matrix")
    execution_contract = matrices[0][1]["execution_contract"]
    if any(
        matrix["execution_contract"] != execution_contract
        for _, matrix in matrices[1:]
    ):
        raise ValueError("raw matrices have different execution contracts")
    all_points = [
        point
        for _, matrix in matrices
        for point in matrix["points"]
    ]
    observation_ids = [
        str(point["observation_id"]) for point in all_points
    ]
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("raw matrices contain duplicate observation IDs")
    hardware = HardwareCostProfile(
        device_type="cuda",
        device_name=args.device_name,
        total_memory_bytes=args.total_memory_bytes,
        memory_safety_fraction=args.memory_fraction,
        device_count=1,
    )
    contract_base = "atombit-smooth-rms-fp32-bfgs-f64-frechet-h100"
    specifications = {
        "peak_allocated_bytes": {
            "metric": "bytes",
            "terms": (
                "fixed",
                "per_atom",
                "per_active_edge",
                "per_candidate_edge",
                "per_dense_state_element",
            ),
            "conservative": True,
        },
        "peak_reserved_bytes": {
            "metric": "bytes",
            "terms": (
                "fixed",
                "per_atom",
                "per_active_edge",
                "per_candidate_edge",
                "per_dense_state_element",
            ),
            "conservative": True,
        },
        "seconds_per_evaluation": {
            "metric": "seconds",
            "terms": (
                "fixed",
                "per_atom",
                "per_active_edge",
                "per_candidate_edge",
                "per_dense_linear_algebra_work",
            ),
            "conservative": False,
        },
    }
    models = {}
    serialized_observations = {}
    for measured_key, specification in specifications.items():
        observations = [
            observation
            for raw_dir, matrix in matrices
            for observation in _observations(
                matrix,
                raw_dir=raw_dir,
                profiles_dir=args.profiles_dir,
                measured_key=measured_key,
            )
        ]
        model = fit_hardware_cost_model(
            observations,
            contract_id=f"{contract_base}-{measured_key}-v1",
            metric=specification["metric"],
            hardware=hardware,
            coefficient_names=specification["terms"],
            conservative=specification["conservative"],
        )
        models[measured_key] = {
            "model": _model_dict(model),
            "fit_error": asdict(
                summarize_calibration_error(
                    observations,
                    model,
                    split="fit",
                )
            ),
            "validation_error": asdict(
                summarize_calibration_error(
                    observations,
                    model,
                    split="validation",
                )
            ),
        }
        serialized_observations[measured_key] = [
            {
                **asdict(observation),
                "predicted_value": (
                    model.safety_factor
                    * model.coefficients.estimate_features(
                        observation.features
                    )
                ),
            }
            for observation in observations
        ]
    output = {
        "schema_version": 1,
        "status": "fit_complete",
        "contract": execution_contract,
        "hardware": asdict(hardware),
        "matrix": all_points,
        "models": models,
        "observations": serialized_observations,
    }
    output["calibration_sha256"] = _canonical_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration_sha256": output["calibration_sha256"],
                "models": {
                    key: {
                        "fit_mare": value["fit_error"][
                            "mean_absolute_relative_error"
                        ],
                        "validation_mare": value["validation_error"][
                            "mean_absolute_relative_error"
                        ],
                        "validation_max": value["validation_error"][
                            "max_absolute_relative_error"
                        ],
                    }
                    for key, value in models.items()
                },
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
