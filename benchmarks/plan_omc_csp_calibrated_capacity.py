#!/usr/bin/env python3
"""Apply a signed reserved-memory calibration to held-out OMC P512 pools."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from batch_mlip import (
    HardwareCalibratedBatchPlanner,
    load_hardware_cost_model,
    read_planning_profile,
)

DEFAULT_WORKLOADS = (
    "OPT-OMC-AXOSOW-P512-INTRA-NARROW-v1",
    "OPT-OMC-BOQQUT-P512-INTRA-WIDE-v1",
    "OPT-OMC-ROF-A-P512-INTRA-WIDE-v1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-index-dir", type=Path)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--max-cost-ratio", type=float, default=2.0)
    parser.add_argument("--workloads", nargs="*", default=DEFAULT_WORKLOADS)
    args = parser.parse_args()

    model = load_hardware_cost_model(
        args.calibration,
        model_name="peak_reserved_bytes",
    )
    planner = HardwareCalibratedBatchPlanner(
        model,
        max_batch_size=args.max_batch_size,
        max_cost_ratio=args.max_cost_ratio,
    )
    workloads: dict[str, Any] = {}
    for workload_id in args.workloads:
        profile = read_planning_profile(
            args.profiles_dir / f"{workload_id}.json"
        )
        plan = planner.plan_bound_profiles(profile.systems)
        by_index = {system.index: system for system in profile.systems}
        validation_indices = None
        validation_path = None
        if args.validation_index_dir is not None:
            target = max(
                plan.buckets,
                key=lambda bucket: bucket.predicted_peak_bytes,
            )
            ranked = sorted(
                target.system_indices,
                key=lambda index: planner.hardware_model.estimate(
                    by_index[index]
                ),
                reverse=True,
            )
            validation_indices = ranked[: target.resident_capacity]
            args.validation_index_dir.mkdir(parents=True, exist_ok=True)
            validation_path = (
                args.validation_index_dir / f"{workload_id}.json"
            )
            validation_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workload_id": workload_id,
                        "workload_manifest_sha256": (
                            profile.workload_manifest_sha256
                        ),
                        "planning_profile_sha256": profile.profile_sha256,
                        "cost_model_contract_id": model.contract_id,
                        "indices": validation_indices,
                        "predicted_peak_bytes": math.ceil(
                            model.estimate_batch(
                                by_index[index]
                                for index in validation_indices
                            )
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        workloads[workload_id] = {
            "profile_sha256": profile.profile_sha256,
            "systems": len(profile.systems),
            "bucket_count": len(plan.buckets),
            "capacity_validation_indices": (
                None if validation_path is None else str(validation_path)
            ),
            "buckets": [
                {
                    "system_count": len(bucket.system_indices),
                    "resident_capacity": bucket.resident_capacity,
                    "predicted_peak_bytes": bucket.predicted_peak_bytes,
                    "predicted_budget_fraction": (
                        bucket.predicted_peak_bytes
                        / plan.memory_budget_bytes
                    ),
                    "atom_count_min": min(
                        by_index[index].structure.atom_count
                        for index in bucket.system_indices
                    ),
                    "atom_count_max": max(
                        by_index[index].structure.atom_count
                        for index in bucket.system_indices
                    ),
                }
                for bucket in plan.buckets
            ],
        }
    output = {
        "schema_version": 1,
        "status": "complete",
        "calibration": str(args.calibration),
        "cost_model_contract_id": model.contract_id,
        "memory_budget_bytes": planner.memory_budget_bytes,
        "memory_safety_fraction": model.hardware.memory_safety_fraction,
        "max_batch_size": args.max_batch_size,
        "max_cost_ratio": args.max_cost_ratio,
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
