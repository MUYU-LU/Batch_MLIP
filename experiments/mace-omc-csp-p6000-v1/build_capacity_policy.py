#!/usr/bin/env python3
"""Fit and seal the MACE-OFF23-Small H100 capacity policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    HardwareCapacityPolicy,
    HardwareCostProfile,
    LayeredCalibrationObservation,
    LayeredCostFeatures,
    fit_hardware_cost_model,
    read_planning_profile,
    summarize_calibration_error,
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation(
    *,
    observation_id: str,
    split: str,
    profile_path: Path,
    result_path: Path,
    batch_size: int,
) -> tuple[LayeredCalibrationObservation, dict[str, Any]]:
    profile = read_planning_profile(profile_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") not in {"complete", "passed"} or result.get(
        "error"
    ) is not None:
        raise ValueError(f"capacity point {observation_id} did not complete")
    if int(result["batch_size"]) != batch_size:
        raise ValueError(f"capacity point {observation_id} has the wrong batch size")
    features = LayeredCostFeatures.from_profiles(profile.systems[:batch_size])
    observation = LayeredCalibrationObservation(
        observation_id=observation_id,
        split=split,
        workload_id=profile.workload_id,
        workload_manifest_sha256=profile.workload_manifest_sha256,
        planning_profile_sha256=profile.profile_sha256,
        features=features,
        measured_value=float(result["peak_reserved_bytes"]),
    )
    source = {
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": _file_sha256(profile_path),
        "result_path": str(result_path.resolve()),
        "result_sha256": _file_sha256(result_path),
    }
    return observation, source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-profile", type=Path, required=True)
    parser.add_argument("--cross-results", type=Path, required=True)
    parser.add_argument("--t2-profile", type=Path, required=True)
    parser.add_argument("--t2-result", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specifications = (
        ("cross-mix-B64", "fit", args.cross_profile, args.cross_results / "mace_B64.json", 64),
        ("cross-mix-B128", "fit", args.cross_profile, args.cross_results / "mace_B128.json", 128),
        ("cross-mix-B192", "fit", args.cross_profile, args.cross_results / "mace_B192.json", 192),
        ("t2-wide-B256", "validation", args.t2_profile, args.t2_result, 256),
    )
    observations = []
    sources = {}
    for observation_id, split, profile, result, batch_size in specifications:
        observation, source = _observation(
            observation_id=observation_id,
            split=split,
            profile_path=profile,
            result_path=result,
            batch_size=batch_size,
        )
        observations.append(observation)
        sources[observation_id] = source

    hardware = HardwareCostProfile(
        device_type="cuda",
        device_name="NVIDIA H100 80GB HBM3",
        total_memory_bytes=85017886720,
        memory_safety_fraction=0.85,
        device_count=1,
    )
    model = fit_hardware_cost_model(
        observations,
        contract_id=(
            "mace-off23-small-f64-bfgs-f64-frechet-h100-"
            "peak_reserved_bytes-v1"
        ),
        metric="bytes",
        hardware=hardware,
        coefficient_names=("fixed", "per_candidate_edge"),
        conservative=True,
    )
    serialized_observations = []
    for observation in observations:
        predicted = model.safety_factor * model.coefficients.estimate_features(
            observation.features
        )
        serialized_observations.append(
            {
                **asdict(observation),
                "predicted_value": float(predicted),
                "predicted_to_measured": float(
                    predicted / observation.measured_value
                ),
                "sources": sources[observation.observation_id],
            }
        )
    calibration = {
        "schema_version": 1,
        "status": "fit_complete",
        "model": asdict(model),
        "fit_error": asdict(
            summarize_calibration_error(observations, model, split="fit")
        ),
        "validation_error": asdict(
            summarize_calibration_error(observations, model, split="validation")
        ),
        "observations": serialized_observations,
    }
    calibration["calibration_sha256"] = _canonical_sha256(calibration)

    contract = {
        "calculator_type": "batch_mlip.models.mace.MACEBatchCalculator",
        "calculator_attributes": {
            "energy_units_to_eV": 1.0,
            "graph_mode": "rebuild",
            "head": "Default",
            "length_units_to_A": 1.0,
        },
        "cell_filter_type": "FrechetCellFilter",
        "cpu_threads": 1,
        "cuda": "12.8",
        "cuda_allocator": "native",
        "cutoff_A": 4.5,
        "force_mode": "native_mace",
        "linear_algebra_backend": "auto",
        "maximum_validated_batch_size": 256,
        "model_dtype": "torch.float64",
        "model_id": "MACE-OFF23-Small",
        "model_parameter_count": 694320,
        "model_state_sha256": (
            "2a7e1c92199cc71640d3ddc1a6dda2bbffffa397a7429790edcc3e5a45766526"
        ),
        "model_type": "mace.modules.models.ScaleShiftMACE",
        "neighbor_backend": "auto",
        "optimizer_dtype": "torch.float64",
        "optimizer_type": "BatchedBFGS",
        "skin_A": 0.5,
        "torch": "2.9.1+cu128",
    }
    unsigned = HardwareCapacityPolicy(
        policy_id="omc-csp-mace-off23-small-h100-capacity-v1",
        source_calibration_sha256=calibration["calibration_sha256"],
        model_name="peak_reserved_bytes",
        contract=contract,
        model=model,
        policy_sha256="0" * 64,
    )
    policy = HardwareCapacityPolicy(
        policy_id=unsigned.policy_id,
        source_calibration_sha256=unsigned.source_calibration_sha256,
        model_name=unsigned.model_name,
        contract=unsigned.contract,
        model=unsigned.model,
        policy_sha256=unsigned.calculate_sha256(),
    )
    calibration["policy_id"] = policy.policy_id
    calibration["policy_sha256"] = policy.policy_sha256

    args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_output.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.policy_output.write_text(
        json.dumps(
            {**policy.unsigned_dict(), "policy_sha256": policy.policy_sha256},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "calibration_sha256": calibration["calibration_sha256"],
                "fit_error": calibration["fit_error"],
                "policy_sha256": policy.policy_sha256,
                "validation_error": calibration["validation_error"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
