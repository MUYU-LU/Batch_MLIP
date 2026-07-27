#!/usr/bin/env python3
"""Run four ASE worker pools through one multi-GPU CUDA MPS daemon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def terminal_summary(aggregate: dict) -> dict:
    """Exclude per-structure records from terminal output."""

    return {
        key: value
        for key, value in aggregate.items()
        if key != "gpu_results"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = sorted(args.manifest_dir.glob("*-MPS-G*.json"))
    if len(manifests) != 4:
        parser.error("manifest directory must contain four MPS GPU shards")
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime_dir / "shared"
    pipe = runtime / "pipe"
    log = runtime / "log"
    pipe.mkdir(parents=True, exist_ok=True)
    log.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "CUDA_MPS_PIPE_DIRECTORY": str(pipe),
        "CUDA_MPS_LOG_DIRECTORY": str(log),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    processes = []
    logs = []
    session_started = time.perf_counter()
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=environment,
            check=True,
        )

        for gpu_index, manifest in enumerate(manifests):
            output = args.output_dir / f"mps_gpu_{gpu_index}.json"
            log_path = args.output_dir / f"mps_gpu_{gpu_index}.log"
            handle = log_path.open("w", encoding="utf-8")
            logs.append(handle)
            command = [
                str(args.python),
                "benchmarks/benchmark_mps_ase_pool.py",
                "--mlip",
                "atombit",
                "--task",
                "optimization",
                "--optimizer",
                "bfgs",
                "--pool-size",
                "750",
                "--workers",
                "4",
                "--device",
                f"cuda:{gpu_index}",
                "--gpu-index",
                str(gpu_index),
                "--workload-manifest",
                str(manifest),
                "--dataset-dir",
                str(args.dataset_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--fmax",
                "0.05",
                "--max-steps",
                "500",
                "--alpha",
                "70.0",
                "--max-step",
                "0.2",
                "--optimizer-dtype",
                "float64",
                "--cpu-threads-per-worker",
                "1",
                "--deterministic",
                "--output",
                str(output),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        return_codes = [process.wait() for process in processes]
        if any(return_codes):
            raise RuntimeError(f"MPS harness failures: {return_codes}")
    finally:
        for handle in logs:
            handle.close()
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            input="quit\n",
            text=True,
            env=environment,
            check=False,
        )

    session_seconds = time.perf_counter() - session_started
    results = [
        json.loads((args.output_dir / f"mps_gpu_{index}.json").read_text())
        for index in range(4)
    ]
    production_seconds = max(
        result["timing"]["wall_seconds"] for result in results
    )
    job_count = sum(result["pool_size"] for result in results)
    aggregate = {
        "schema_version": 1,
        "method": "ase_cuda_mps_4_workers_per_gpu",
        "gpu_count": 4,
        "workers_per_gpu": 4,
        "jobs": job_count,
        "production_wall_seconds": production_seconds,
        "total_session_wall_seconds": session_seconds,
        "systems_per_second": job_count / production_seconds,
        "converged": sum(result["converged"] for result in results),
        "optimizer_steps_total": sum(
            result["optimizer_steps_total"] for result in results
        ),
        "model_evaluations_total": sum(
            result["model_evaluations_total"] for result in results
        ),
        "peak_gpu_memory_bytes": [
            result["peak_gpu_memory_bytes_nvidia_smi"]
            for result in results
        ],
        "gpu_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            terminal_summary(aggregate),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
