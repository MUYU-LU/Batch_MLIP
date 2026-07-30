#!/usr/bin/env python3
"""Build deterministic nested selections for expanded OMC-CSP calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from batch_mlip import (
    HardwareCalibratedBatchPlanner,
    load_hardware_cost_model,
    read_planning_profile,
)

FIT_WORKLOADS = (
    "OPT-OMC-SOXLEX-P512-INTRA-NARROW-v1",
    "OPT-OMC-JAYDUI-P512-INTRA-NARROW-v1",
    "OPT-OMC-NACJAF-P512-INTRA-NARROW-v1",
    "OPT-OMC-KONTIQ-P512-INTRA-NARROW-v1",
    "OPT-OMC-HAMTIZ-P512-INTRA-NARROW-v1",
    "OPT-OMC-BOQWIN-P512-INTRA-NARROW-v1",
    "OPT-OMC-WICZUF-P512-INTRA-NARROW-v1",
    "OPT-OMC-XAFPAY-P512-INTRA-WIDE-v1",
)
VALIDATION_WORKLOADS = (
    "OPT-OMC-PAHYON-P512-INTRA-NARROW-v1",
    "OPT-OMC-OBEQUJ-P512-INTRA-NARROW-v1",
    "OPT-OMC-ROF-C-P512-INTRA-WIDE-v1",
)
FINAL_WORKLOADS = (
    "OPT-OMC-UJIRIO-P512-INTRA-NARROW-v1",
    "OPT-OMC-WIDBAO-P512-INTRA-NARROW-v1",
    "OPT-OMC-XAFQIH-P512-INTRA-NARROW-v1",
    "OPT-OMC-XULDUD-P512-INTRA-WIDE-v1",
)

EXECUTION_CONTRACT = {
    "mlip": "AtomBit-smooth-rms-fp32",
    "optimizer": "BatchedBFGS",
    "optimizer_dtype": "torch.float64",
    "cell_filter": "FrechetCellFilter",
    "cutoff_A": 6.0,
    "skin_A": 0.5,
    "force_mode": "autograd",
    "neighbor_backend": "auto",
    "max_steps": 3,
    "warm_executions": 1,
    "measured_executions": 1,
    "scheduling": "single_batch",
    "cuda_allocator": "expandable_segments",
    "deterministic": True,
}


def _family(workload_id: str) -> str:
    return workload_id.removeprefix("OPT-OMC-").split("-P512-", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-output", type=Path, required=True)
    parser.add_argument("--max-cost-ratio", type=float, default=2.0)
    parser.add_argument(
        "--design",
        choices=("expanded", "final"),
        default="expanded",
        help="Expanded fit/validation matrix or untouched final-family tests.",
    )
    args = parser.parse_args()

    model = load_hardware_cost_model(
        args.calibration,
        model_name="peak_reserved_bytes",
    )
    source_calibration_sha256 = json.loads(
        args.calibration.read_text(encoding="utf-8")
    )["calibration_sha256"]
    planner = HardwareCalibratedBatchPlanner(
        model,
        max_batch_size=512,
        max_cost_ratio=args.max_cost_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = []
    summaries = {}
    split_workloads = (
        (
            ("fit", FIT_WORKLOADS),
            ("validation", VALIDATION_WORKLOADS),
        )
        if args.design == "expanded"
        else (("final", FINAL_WORKLOADS),)
    )
    for split, workload_ids in split_workloads:
        for workload_id in workload_ids:
            profile = read_planning_profile(
                args.profiles_dir / f"{workload_id}.json"
            )
            plan = planner.plan_bound_profiles(profile.systems)
            target = max(
                plan.buckets,
                key=lambda bucket: bucket.predicted_peak_bytes,
            )
            by_index = {system.index: system for system in profile.systems}
            ranked = sorted(
                target.system_indices,
                key=lambda index: model.estimate(by_index[index]),
                reverse=True,
            )
            capacity = target.resident_capacity
            if capacity < 128:
                raise ValueError(
                    f"{workload_id} capacity {capacity} is below B128"
                )
            family = _family(workload_id)
            sizes = (
                (16, 128, capacity)
                if args.design == "expanded"
                else (128, capacity)
            )
            if len(set(sizes)) != len(sizes):
                raise ValueError(f"duplicate batch sizes for {workload_id}")
            summaries[workload_id] = {
                "family": family,
                "split": split,
                "bucket_count": len(plan.buckets),
                "resident_capacity": capacity,
                "memory_budget_bytes": plan.memory_budget_bytes,
            }
            scales = (
                ("small", "knee", "near_capacity")
                if args.design == "expanded"
                else ("knee", "near_capacity")
            )
            for scale, batch_size in zip(scales, sizes, strict=True):
                indices = ranked[:batch_size]
                observation_id = (
                    f"expanded-{split}-{family}-{scale}-B{batch_size}"
                )
                selection_name = f"{observation_id}.json"
                selection_path = args.output_dir / selection_name
                selection_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "workload_id": workload_id,
                            "workload_manifest_sha256": (
                                profile.workload_manifest_sha256
                            ),
                            "planning_profile_sha256": profile.profile_sha256,
                            "cost_model_contract_id": model.contract_id,
                            "selection_rule": (
                                "nested descending predicted per-system "
                                "reserved-memory cost within the planner's "
                                "highest predicted-peak bucket"
                            ),
                            "scale": scale,
                            "indices": indices,
                            "predicted_peak_bytes": math.ceil(
                                model.estimate_batch(
                                    by_index[index] for index in indices
                                )
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                points.append(
                    {
                        "split": split,
                        "family": family,
                        "mixing": (
                            "INTRA-WIDE"
                            if "-INTRA-WIDE-" in workload_id
                            else "INTRA-NARROW"
                        ),
                        "scale": scale,
                        "batch_size": batch_size,
                        "pool_size": 512,
                        "workload_id": workload_id,
                        "observation_id": observation_id,
                        "indices_json": os.path.relpath(
                            selection_path,
                            start=args.matrix_output.parent,
                        ),
                    }
                )
    matrix = {
        "schema_version": 1,
        "status": "planned",
        "execution_contract": EXECUTION_CONTRACT,
        "source_calibration": str(args.calibration),
        "source_calibration_sha256": source_calibration_sha256,
        "selection_design": {
            "design": args.design,
            "scales": (
                ["B16", "B128", "model_predicted_capacity"]
                if args.design == "expanded"
                else ["B128", "model_predicted_capacity"]
            ),
            "nested": True,
            "single_measurement_per_point": True,
        },
        "families": summaries,
        "points": points,
    }
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fit_families": (
                    len(FIT_WORKLOADS)
                    if args.design == "expanded"
                    else 0
                ),
                "validation_families": (
                    len(VALIDATION_WORKLOADS)
                    if args.design == "expanded"
                    else 0
                ),
                "final_families": (
                    len(FINAL_WORKLOADS)
                    if args.design == "final"
                    else 0
                ),
                "points": len(points),
                "output": str(args.matrix_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
