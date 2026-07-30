"""Offline fitting and validation of hardware-bound layered cost models."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .profiles import (
    HardwareBoundCostModel,
    HardwareCostProfile,
    LayeredCostCoefficients,
    LayeredCostFeatures,
)

CalibrationSplit = Literal["fit", "validation"]

_COEFFICIENT_FEATURES = {
    "fixed": None,
    "per_atom": "atom_count",
    "per_active_edge": "active_edge_count",
    "per_candidate_edge": "candidate_edge_count",
    "per_linear_state_element": "linear_state_elements",
    "per_dense_state_element": "dense_state_elements",
    "per_dense_linear_algebra_work": "dense_linear_algebra_work",
}


def _canonical_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LayeredCalibrationObservation:
    """One measured batch point tied to immutable workload/profile hashes."""

    observation_id: str
    split: CalibrationSplit
    workload_id: str
    workload_manifest_sha256: str
    planning_profile_sha256: str
    features: LayeredCostFeatures
    measured_value: float

    def __post_init__(self) -> None:
        if not self.observation_id or not self.workload_id:
            raise ValueError("calibration observation identity must not be empty")
        if self.split not in ("fit", "validation"):
            raise ValueError("calibration split must be fit or validation")
        if (
            len(self.workload_manifest_sha256) != 64
            or len(self.planning_profile_sha256) != 64
        ):
            raise ValueError("calibration observations require SHA256 identities")
        if (
            not math.isfinite(self.measured_value)
            or self.measured_value <= 0.0
        ):
            raise ValueError("measured calibration value must be positive")


@dataclass(frozen=True)
class CalibrationErrorSummary:
    """Prediction error over a named calibration split."""

    count: int
    mean_absolute_relative_error: float
    max_absolute_relative_error: float
    mean_relative_error: float
    minimum_predicted_to_measured: float
    maximum_predicted_to_measured: float


def _design_matrix(
    observations: Sequence[LayeredCalibrationObservation],
    coefficient_names: Sequence[str],
) -> np.ndarray:
    rows = []
    for observation in observations:
        feature_values = asdict(observation.features)
        rows.append(
            [
                1.0 if _COEFFICIENT_FEATURES[name] is None else float(
                    feature_values[_COEFFICIENT_FEATURES[name]]
                )
                for name in coefficient_names
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _nonnegative_least_squares(
    features: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Solve the small non-negative fit without requiring SciPy."""

    scales = np.linalg.norm(features, axis=0)
    if np.any(scales == 0.0):
        raise ValueError("selected calibration features must vary and be non-zero")
    normalized = features / scales
    coefficients = np.zeros(normalized.shape[1], dtype=np.float64)
    prediction = normalized @ coefficients
    for _ in range(50_000):
        previous = coefficients.copy()
        for column in range(normalized.shape[1]):
            values = normalized[:, column]
            residual = targets - prediction + values * coefficients[column]
            updated = max(0.0, float(values @ residual) / float(values @ values))
            prediction += values * (updated - coefficients[column])
            coefficients[column] = updated
        if np.max(np.abs(coefficients - previous)) <= 1e-11 * max(
            1.0,
            float(np.max(np.abs(coefficients))),
        ):
            break
    return coefficients / scales


def fit_hardware_cost_model(
    observations: Sequence[LayeredCalibrationObservation],
    *,
    contract_id: str,
    metric: Literal["seconds", "bytes"],
    hardware: HardwareCostProfile,
    coefficient_names: Sequence[str],
    conservative: bool = False,
) -> HardwareBoundCostModel:
    """Fit a hardware-bound model using only observations marked ``fit``."""

    rows = tuple(row for row in observations if row.split == "fit")
    names = tuple(coefficient_names)
    if len(rows) < len(names):
        raise ValueError("fit observations must outnumber selected coefficients")
    if not names or len(set(names)) != len(names):
        raise ValueError("coefficient names must be non-empty and unique")
    unknown = set(names) - set(_COEFFICIENT_FEATURES)
    if unknown:
        raise ValueError(f"unknown layered coefficient names: {sorted(unknown)}")
    features = _design_matrix(rows, names)
    targets = np.asarray(
        [row.measured_value for row in rows],
        dtype=np.float64,
    )
    fitted = _nonnegative_least_squares(features, targets)
    values = {name: 0.0 for name in _COEFFICIENT_FEATURES}
    values.update(zip(names, fitted, strict=True))
    coefficients = LayeredCostCoefficients(**values)
    raw_predictions = features @ fitted
    safety_factor = 1.0
    if conservative:
        if np.any(raw_predictions <= 0.0):
            raise ValueError("cannot conservatively scale zero predictions")
        safety_factor = max(
            1.0,
            float(np.max(targets / raw_predictions)),
        )
    return HardwareBoundCostModel(
        contract_id=contract_id,
        metric=metric,
        hardware=hardware,
        coefficients=coefficients,
        safety_factor=safety_factor,
    )


def summarize_calibration_error(
    observations: Sequence[LayeredCalibrationObservation],
    model: HardwareBoundCostModel,
    *,
    split: CalibrationSplit,
) -> CalibrationErrorSummary:
    """Summarize relative error for fit or held-out validation points."""

    rows = tuple(row for row in observations if row.split == split)
    if not rows:
        raise ValueError(f"no {split} calibration observations")
    ratios = np.asarray(
        [
            model.safety_factor
            * model.coefficients.estimate_features(row.features)
            / row.measured_value
            for row in rows
        ],
        dtype=np.float64,
    )
    relative = ratios - 1.0
    return CalibrationErrorSummary(
        count=len(rows),
        mean_absolute_relative_error=float(np.mean(np.abs(relative))),
        max_absolute_relative_error=float(np.max(np.abs(relative))),
        mean_relative_error=float(np.mean(relative)),
        minimum_predicted_to_measured=float(np.min(ratios)),
        maximum_predicted_to_measured=float(np.max(ratios)),
    )


def load_hardware_cost_model(
    path: str | Path,
    *,
    model_name: str,
) -> HardwareBoundCostModel:
    """Load one signed model from a complete hardware-calibration artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_hash = payload.pop("calibration_sha256")
    if _canonical_sha256(payload) != expected_hash:
        raise ValueError("hardware calibration content hash does not match")
    if payload.get("status") != "fit_complete":
        raise ValueError("hardware calibration artifact is incomplete")
    try:
        values = payload["models"][model_name]["model"]
    except KeyError as error:
        raise KeyError(
            f"hardware calibration has no model {model_name!r}"
        ) from error
    return HardwareBoundCostModel(
        contract_id=values["contract_id"],
        metric=values["metric"],
        hardware=HardwareCostProfile(**values["hardware"]),
        coefficients=LayeredCostCoefficients(**values["coefficients"]),
        safety_factor=float(values["safety_factor"]),
    )
