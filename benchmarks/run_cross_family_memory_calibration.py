#!/usr/bin/env python3
"""Run one-step automatic BFGS memory calibration across six GPU families."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

FAMILY_LABELS = (
    "GUFJOG44",
    "SOXLEX48",
    "XATMOV88",
    "OBEQIX220",
    "ROFA-MIX",
    "ROFB296",
)


def summarize_family_result(result: dict) -> dict:
    """Extract comparable memory-planning metrics from one benchmark result."""

    call = result["call_records"][0]
    schedule = call["schedule"]
    predicted = max(
        chunk["predicted_peak_bytes"]
        for chunk in schedule["resident_plan_chunks"]
        if chunk["predicted_peak_bytes"] is not None
    )
    reserved = call["peak_reserved_bytes"]
    return {
        "workload_id": result["workload_id"],
        "jobs": result["jobs"],
        "atoms": result["atoms"],
        "api_wall_seconds": call["wall_time_s"],
        "production_wall_seconds": schedule["production_run_seconds"],
        "systems_per_second": call["systems_per_s"],
        "converged": call["converged"],
        "resident_plan_chunks": schedule["resident_plan_chunk_count"],
        "execution_chunks": schedule["execution_chunk_count"],
        "execution_batch_sizes": [
            chunk["system_count"] for chunk in schedule["planned_chunks"]
        ],
        "predicted_peak_bytes": predicted,
        "peak_allocated_bytes": call["peak_allocated_bytes"],
        "peak_reserved_bytes": reserved,
        "observed_reserved_to_predicted": reserved / predicted,
        "memory_budget_bytes": schedule["memory_budget_bytes_per_gpu"],
        "observed_reserved_to_budget": (
            reserved / schedule["memory_budget_bytes_per_gpu"]
        ),
        "parallel_chunk_policy": schedule["parallel_chunk_policy"],
        "model_evaluations": call["model_evaluations"],
        "graph_evaluations": call["graph_evaluations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5")
    parser.add_argument("--memory-growth-margin", type=float, default=1.25)
    args = parser.parse_args()

    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) != len(FAMILY_LABELS):
        parser.error(f"--devices must provide {len(FAMILY_LABELS)} devices")
    if args.memory_growth_margin < 1.0:
        parser.error("--memory-growth-margin must be at least one")
    manifests = {
        label: args.manifest_dir / f"OPT-RB-{label}-R512-v1.json"
        for label in FAMILY_LABELS
    }
    missing = [str(path) for path in manifests.values() if not path.is_file()]
    if missing:
        parser.error(f"missing calibration manifests: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    processes: list[tuple[str, subprocess.Popen, object]] = []
    for label, device in zip(FAMILY_LABELS, devices, strict=True):
        output = args.output_dir / f"memory_{label}.json"
        log_path = args.output_dir / f"memory_{label}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [
            str(args.python),
            "benchmarks/benchmark_persistent_executor.py",
            "--mlip",
            "atombit",
            "--mode",
            "persistent",
            "--optimizer",
            "bfgs",
            "--devices",
            device,
            "--calls",
            "1",
            "--automatic-capacity",
            "--memory-growth-margin",
            str(args.memory_growth_margin),
            "--max-steps",
            "1",
            "--fmax",
            "0.05",
            "--deterministic",
            "--workload-manifest",
            str(manifests[label]),
            "--dataset-dir",
            str(args.dataset_dir),
            "--cache-path",
            str(args.output_dir / f"cache_{label}.json"),
            "--clear-cache",
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(output),
        ]
        processes.append(
            (
                label,
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
                log,
            )
        )

    failures = []
    for label, process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append((label, return_code))
    if failures:
        raise RuntimeError(f"family calibration failures: {failures}")

    families = {}
    for label in FAMILY_LABELS:
        result_path = args.output_dir / f"memory_{label}.json"
        families[label] = summarize_family_result(
            json.loads(result_path.read_text())
        )
    aggregate = {
        "schema_version": 1,
        "phase": "one_step_family_memory_calibration",
        "memory_growth_margin": args.memory_growth_margin,
        "families": families,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
