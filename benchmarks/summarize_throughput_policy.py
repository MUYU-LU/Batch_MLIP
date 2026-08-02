#!/usr/bin/env python3
"""Validate controlled curves and build a descriptor-nearest policy table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATOMBIT_SHA256 = "b27c372cb2c2848ae1e54a9ffd7a2aa0b0401d9cc0ed922e3be250ff63e44486"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_path(mlip: str, workload_id: str) -> Path:
    model = "atombit" if mlip == "atombit" else "mace_off_small"
    return ROOT / "benchmarks/workloads/profiles" / model / f"{workload_id}.json"


def _point(
    path: Path,
    *,
    workload: dict[str, Any],
    mlip: str,
    optimizer: str,
    capacity: int,
    max_steps: int,
) -> dict[str, Any]:
    if not path.exists():
        return {"capacity": capacity, "valid": False, "reasons": ["missing"]}
    data = _load(path)
    reasons = []
    expected_manifest = _load(ROOT / workload["manifest"])["manifest_sha256"]
    checks = {
        "status": data.get("status") == "passed",
        "mlip": data.get("mlip") == mlip,
        "optimizer": data.get("optimizer") == optimizer,
        "method": data.get("method") == "active",
        "manifest": data.get("workload_manifest_sha256") == expected_manifest,
        "jobs": data.get("jobs") == 256,
        "capacity": data.get("batch_size") == capacity,
        "fmax": data.get("fmax_eV_per_A") == 0.05,
        "max_steps": data.get("max_steps") == max_steps,
        "deterministic": data.get("deterministic_algorithms") is True,
        "cpu_threads": data.get("cpu_threads") == 1,
        "converged": data.get("converged") == 256,
        "gpu": data.get("environment", {}).get("gpu_name") == "NVIDIA H100 80GB HBM3",
        "model": (
            data.get("checkpoint_sha256") == ATOMBIT_SHA256
            if mlip == "atombit"
            else data.get("model") == "small"
        ),
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    total = data.get("environment", {}).get("gpu_total_memory_bytes")
    allocated = data.get("peak_allocated_bytes")
    reserved = data.get("peak_reserved_bytes")
    if not isinstance(total, int) or not isinstance(allocated, int):
        reasons.append("missing_memory")
    else:
        if allocated > 0.85 * total:
            reasons.append("allocated_above_85_percent")
        if not isinstance(reserved, int) or reserved > 0.85 * total:
            reasons.append("reserved_above_85_percent")
    seconds = data.get("timing_seconds")
    if not isinstance(seconds, int | float) or seconds <= 0:
        reasons.append("invalid_timing")
    return {
        "capacity": capacity,
        "valid": not reasons,
        "reasons": reasons,
        "seconds": seconds,
        "systems_per_second": data.get("systems_per_second"),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "model_evaluations": data.get("model_evaluations"),
        "graph_evaluations": data.get("graph_evaluations"),
        "optimizer_steps_total": data.get("optimizer_steps_total"),
        "sha256": _sha256(path),
        "raw_file": str(path.relative_to(ROOT)),
    }


def _features(profile: dict[str, Any]) -> list[float]:
    dof = 3.0 * profile["atom_count_mean"] + 9.0
    return [
        math.log(profile["atom_count_mean"]),
        math.log(profile["candidate_edges_mean"]),
        math.log(dof * dof),
        float(profile["atom_count_cv"]),
        float(profile["candidate_edges_cv"]),
    ]


def _distance(left: list[float], right: list[float]) -> float:
    weights = (1.0, 1.0, 0.5, 2.0, 2.0)
    return math.sqrt(
        sum(weight * (a - b) ** 2 for weight, a, b in zip(weights, left, right, strict=True))
    )


def _largest_safe_analysis(curves: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for curve in curves:
        valid = [point for point in curve["points"] if point["valid"]]
        if not valid:
            rows.append(
                {
                    "split": curve["split"],
                    "workload_id": curve["workload_id"],
                    "mlip": curve["mlip"],
                    "optimizer": curve["optimizer"],
                    "capacity": None,
                    "throughput_fraction": None,
                    "passed": False,
                    "reason": "no_capacity_passed_convergence_and_memory_gates",
                }
            )
            continue
        selected = max(valid, key=lambda point: point["capacity"])
        fastest = max(point["systems_per_second"] for point in valid)
        fraction = selected["systems_per_second"] / fastest
        rows.append(
            {
                "split": curve["split"],
                "workload_id": curve["workload_id"],
                "mlip": curve["mlip"],
                "optimizer": curve["optimizer"],
                "capacity": selected["capacity"],
                "throughput_fraction": fraction,
                "passed": fraction >= 0.95,
                "reason": None,
            }
        )
    eligible = [row for row in rows if row["capacity"] is not None]
    heldout = [row for row in eligible if row["split"] == "heldout"]
    return {
        "rule": "largest measured capacity passing convergence and 85-percent memory gates",
        "role": (
            "diagnostic support for memory-first planning; measured feasibility "
            "is not an offline runtime predictor"
        ),
        "eligible_curves": len(eligible),
        "passed_curves": sum(row["passed"] for row in eligible),
        "heldout_eligible_curves": len(heldout),
        "heldout_passed_curves": sum(row["passed"] for row in heldout),
        "no_valid_capacity_curves": len(rows) - len(eligible),
        "rows": rows,
    }


def _allocator_diagnostics(raw_dir: Path) -> list[dict[str, Any]]:
    specifications = (
        ("OPT-RB-BOQWIN116-R256-v1", 128),
        ("OPT-RB-BOQWIN116-R256-v1", 256),
        ("OPT-RB-XAFPAY172-R256-v1", 128),
        ("OPT-RB-XAFPAY172-R256-v1", 256),
        ("OPT-RB-ROFB296-R256-v1", 64),
        ("OPT-RB-ROFB296-R256-v1", 128),
    )
    rows = []
    for workload_id, capacity in specifications:
        native_path = raw_dir / (f"heldout_atombit_fire_{workload_id}_B{capacity}.json")
        expandable_path = raw_dir / (
            f"diagnostic_expandable_atombit_fire_{workload_id}_B{capacity}.json"
        )
        if not native_path.exists() or not expandable_path.exists():
            continue
        native = _load(native_path)
        expandable = _load(expandable_path)
        rows.append(
            {
                "workload_id": workload_id,
                "capacity": capacity,
                "native": {
                    "seconds": native.get("timing_seconds"),
                    "systems_per_second": native.get("systems_per_second"),
                    "peak_allocated_bytes": native.get("peak_allocated_bytes"),
                    "peak_reserved_bytes": native.get("peak_reserved_bytes"),
                    "sha256": _sha256(native_path),
                },
                "expandable_segments": {
                    "seconds": expandable.get("timing_seconds"),
                    "systems_per_second": expandable.get("systems_per_second"),
                    "peak_allocated_bytes": expandable.get("peak_allocated_bytes"),
                    "peak_reserved_bytes": expandable.get("peak_reserved_bytes"),
                    "sha256": _sha256(expandable_path),
                },
            }
        )
    return rows


def summarize(matrix_path: Path, raw_dir: Path) -> dict[str, Any]:
    matrix = _load(matrix_path)
    capacities = sorted(
        {int(path.stem.rsplit("_B", 1)[1]) for path in raw_dir.glob("*.json") if "_B" in path.stem}
        | set(matrix["capacities"])
    )
    curves = []
    for workload in matrix["workloads"]:
        for mlip in matrix["mlips"]:
            profile = _load(_profile_path(mlip, workload["workload_id"]))
            for optimizer, config in matrix["optimizers"].items():
                points = []
                for capacity in capacities:
                    evidence_prefix = workload.get(
                        "evidence_prefix",
                        workload["split"],
                    )
                    path = raw_dir / (
                        f"{evidence_prefix}_{mlip}_{optimizer}_"
                        f"{workload['workload_id']}_B{capacity}.json"
                    )
                    points.append(
                        _point(
                            path,
                            workload=workload,
                            mlip=mlip,
                            optimizer=optimizer,
                            capacity=capacity,
                            max_steps=int(config["max_steps"]),
                        )
                    )
                for point in points:
                    throughput = point.get("systems_per_second")
                    if isinstance(throughput, int | float):
                        point["atoms_per_second"] = throughput * profile["atom_count_mean"]
                valid = [point for point in points if point["valid"]]
                selected = None
                if valid:
                    fastest = max(point["systems_per_second"] for point in valid)
                    selected = min(
                        (point for point in valid if point["systems_per_second"] >= 0.95 * fastest),
                        key=lambda point: point["capacity"],
                    )
                curves.append(
                    {
                        "split": workload["split"],
                        "workload_id": workload["workload_id"],
                        "mlip": mlip,
                        "optimizer": optimizer,
                        "profile": profile,
                        "features": _features(profile),
                        "points": points,
                        "selected_capacity": (None if selected is None else selected["capacity"]),
                        "selected_throughput": (
                            None if selected is None else selected["systems_per_second"]
                        ),
                    }
                )
    fit = [curve for curve in curves if curve["split"] == "fit"]
    heldout = [curve for curve in curves if curve["split"] == "heldout"]
    policy_table = [
        {
            "workload_id": curve["workload_id"],
            "mlip": curve["mlip"],
            "optimizer": curve["optimizer"],
            "features": curve["features"],
            "selected_capacity": curve["selected_capacity"],
            "selected_throughput": curve["selected_throughput"],
        }
        for curve in fit
        if curve["selected_capacity"] is not None
    ]
    validation = []
    for target in heldout:
        candidates = [
            curve
            for curve in fit
            if curve["mlip"] == target["mlip"]
            and curve["optimizer"] == target["optimizer"]
            and curve["selected_capacity"] is not None
        ]
        if not candidates:
            validation.append(
                {
                    "workload_id": target["workload_id"],
                    "mlip": target["mlip"],
                    "optimizer": target["optimizer"],
                    "nearest_fit_workload": None,
                    "predicted_capacity": None,
                    "measured_selected_capacity": target["selected_capacity"],
                    "throughput_fraction": None,
                    "passed": False,
                    "reason": "no_valid_fit_curve",
                }
            )
            continue
        nearest = min(
            candidates,
            key=lambda curve: _distance(curve["features"], target["features"]),
        )
        predicted = nearest["selected_capacity"]
        point = next(
            (item for item in target["points"] if item["capacity"] == predicted),
            None,
        )
        valid_points = [item for item in target["points"] if item["valid"]]
        fastest = max(
            (item["systems_per_second"] for item in valid_points),
            default=None,
        )
        fraction = (
            None
            if point is None or not point["valid"] or fastest is None
            else point["systems_per_second"] / fastest
        )
        validation.append(
            {
                "workload_id": target["workload_id"],
                "mlip": target["mlip"],
                "optimizer": target["optimizer"],
                "nearest_fit_workload": nearest["workload_id"],
                "predicted_capacity": predicted,
                "measured_selected_capacity": target["selected_capacity"],
                "throughput_fraction": fraction,
                "passed": fraction is not None and fraction >= 0.95,
            }
        )
    static_policy_passed = bool(validation) and all(row["passed"] for row in validation)
    return {
        "schema_version": 1,
        "status": ("passed" if static_policy_passed else "static_policy_rejected"),
        "decision": (
            "ship_static_throughput_policy"
            if static_policy_passed
            else "retain_deterministic_85_percent_memory_planner"
        ),
        "selection_rule": "smallest valid capacity within 5 percent of fastest",
        "memory_fraction_limit": 0.85,
        "evidence": {
            "fit_workloads": len(matrix["workloads"])
            - sum(item["split"] == "heldout" for item in matrix["workloads"]),
            "heldout_workloads": sum(item["split"] == "heldout" for item in matrix["workloads"]),
            "curves": len(curves),
            "capacity_points": sum(len(curve["points"]) for curve in curves),
            "valid_capacity_points": sum(
                point["valid"] for curve in curves for point in curve["points"]
            ),
        },
        "fit_policy_table": policy_table,
        "curves": curves,
        "heldout_validation": validation,
        "largest_memory_safe_analysis": _largest_safe_analysis(curves),
        "allocator_diagnostics": _allocator_diagnostics(raw_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/matrix.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/results.json",
    )
    args = parser.parse_args()
    result = summarize(args.matrix, args.raw_dir)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
