#!/usr/bin/env python3
"""Run a cost-balanced ASE/BFGS CUDA-MPS baseline for one OMC-CSP workload."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_production import sha256_file  # noqa: E402

from batch_mlip.workloads import (  # noqa: E402
    WorkloadManifest,
    read_workload_manifest,
    write_workload_manifest,
)


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative CUDA indices")
    return gpus


def _job_cost(job: Any) -> float:
    candidates = max(job.topology_edge_counts.values(), default=0)
    dimension = 3 * job.atom_count + 9
    # Static, task-aware proxy: BFGS state + graph and per-atom model work.
    return 16.0 * dimension * dimension + 256.0 * job.atom_count + 64.0 * candidates


def _lpt_shards(manifest: WorkloadManifest, gpus: list[int]) -> list[list[int]]:
    shards = [[] for _ in gpus]
    costs = [0.0 for _ in gpus]
    for index in sorted(
        range(len(manifest.jobs)),
        key=lambda item: (-_job_cost(manifest.jobs[item]), item),
    ):
        target = min(range(len(gpus)), key=lambda item: (costs[item], item))
        shards[target].append(index)
        costs[target] += _job_cost(manifest.jobs[index])
    for shard in shards:
        shard.sort()
    return shards


def _write_shard(
    manifest: WorkloadManifest,
    indices: list[int],
    *,
    gpu: int,
    path: Path,
) -> WorkloadManifest:
    jobs = tuple(replace(manifest.jobs[index], order=order) for order, index in enumerate(indices))
    shard = replace(
        manifest,
        workload_id=f"{manifest.workload_id}-ASE-MPS-GPU{gpu}",
        jobs=jobs,
        metadata={
            **manifest.metadata,
            "baseline": "ase_cuda_mps_static_lpt",
            "source_workload_id": manifest.workload_id,
            "source_workload_manifest_sha256": manifest.manifest_sha256,
            "source_indices": indices,
            "physical_gpu": gpu,
        },
        manifest_sha256="",
    ).seal()
    write_workload_manifest(path, shard)
    return shard


def _mps_environment(
    *,
    visible_devices: list[int],
    pipe: Path,
    log: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in visible_devices),
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe),
            "CUDA_MPS_LOG_DIRECTORY": str(log),
            "PYTHONHASHSEED": "20260729",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "BATCH_MLIP_REPRODUCIBILITY_SEED": "20260729",
            "BATCH_MLIP_DETERMINISTIC_ALGORITHMS": "1",
            "BATCH_MLIP_DETERMINISTIC_WARN_ONLY": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            # Spawned ASE workers execute a script under benchmarks/ and need
            # the repository root to import the local batch_mlip package.
            "PYTHONPATH": str(ROOT),
        }
    )
    return environment


def _start_mps(environment: dict[str, str]) -> None:
    completed = subprocess.run(
        ["nvidia-cuda-mps-control", "-d"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    control_socket = Path(environment["CUDA_MPS_PIPE_DIRECTORY"]) / "control"
    # Some driver releases report a non-zero detached-launch status even after
    # the control daemon has created its socket. The socket is the usable API.
    if completed.returncode and not control_socket.exists():
        raise RuntimeError(
            "could not start CUDA MPS: " f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _stop_mps(environment: dict[str, str]) -> None:
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--mps-session-config",
        type=Path,
        help="Per-physical-GPU external MPS environments; never starts MPS.",
    )
    parser.add_argument(
        "--reuse-active-mps",
        action="store_true",
        help="Use the MPS session supplied through the current environment.",
    )
    args = parser.parse_args()
    if args.workers_per_gpu <= 0 or args.fmax <= 0.0 or args.max_steps <= 0:
        parser.error("workers, fmax, and max steps must be positive")

    started = time.perf_counter()
    manifest = read_workload_manifest(args.manifest)
    gpus = _parse_gpus(args.gpus)
    if len(gpus) > len(manifest.jobs):
        parser.error("the workload has fewer jobs than requested GPUs")
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.runtime_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    shards = _lpt_shards(manifest, gpus)
    shard_specs = []
    for gpu, indices in zip(gpus, shards, strict=True):
        path = shard_dir / f"gpu-{gpu}.json"
        shard = _write_shard(manifest, indices, gpu=gpu, path=path)
        shard_specs.append((gpu, indices, path, shard))

    processes: list[tuple[int, subprocess.Popen[str], Path]] = []
    owns_mps_session = not (args.reuse_active_mps or args.mps_session_config is not None)
    per_gpu_environment: dict[int, dict[str, str]] | None = None
    if args.mps_session_config is not None:
        raw = json.loads(args.mps_session_config.read_text(encoding="utf-8"))
        entries = raw.get("gpus", raw)
        per_gpu_environment = {}
        for gpu in gpus:
            entry = entries.get(str(gpu))
            if not isinstance(entry, dict):
                parser.error(f"MPS session config has no GPU {gpu} entry")
            required = ("cuda_visible_devices", "pipe_directory", "log_directory")
            if any(not entry.get(name) for name in required):
                parser.error(f"MPS session config GPU {gpu} is incomplete")
            per_gpu_environment[gpu] = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(entry["cuda_visible_devices"]),
                "CUDA_MPS_PIPE_DIRECTORY": str(entry["pipe_directory"]),
                "CUDA_MPS_LOG_DIRECTORY": str(entry["log_directory"]),
                "PYTHONHASHSEED": "20260729",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "BATCH_MLIP_REPRODUCIBILITY_SEED": "20260729",
                "BATCH_MLIP_DETERMINISTIC_ALGORITHMS": "1",
                "BATCH_MLIP_DETERMINISTIC_WARN_ONLY": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": str(ROOT),
            }
    elif owns_mps_session:
        # MPS creates UNIX sockets. Keep them on node-local storage rather
        # than the shared project filesystem, where socket creation is not
        # portable.
        runtime = Path("/tmp") / f"batch-mlip-mps-{os.getuid()}-{os.getpid()}"
        pipe = runtime / "pipe"
        log = runtime / "log"
        pipe.mkdir(parents=True, exist_ok=True)
        log.mkdir(parents=True, exist_ok=True)
        environment = _mps_environment(
            visible_devices=gpus,
            pipe=pipe,
            log=log,
        )
    else:
        required = (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            parser.error("--reuse-active-mps requires environment variables: " + ", ".join(missing))
        environment = dict(os.environ)
    try:
        if owns_mps_session:
            _start_mps(environment)
        for logical_gpu, (gpu, indices, shard_path, _) in enumerate(shard_specs):
            worker_output = args.runtime_dir / f"gpu-{gpu}.result.json"
            command = [
                sys.executable,
                str(ROOT / "benchmarks" / "benchmark_mps_ase_pool.py"),
                "--mlip",
                "atombit",
                "--task",
                "optimization",
                "--optimizer",
                "bfgs",
                "--pool-size",
                str(len(indices)),
                "--workers",
                str(args.workers_per_gpu),
                "--device",
                "cuda:0" if per_gpu_environment is not None else f"cuda:{logical_gpu}",
                "--gpu-index",
                str(gpu),
                "--cutoff",
                "6.0",
                "--fmax",
                str(args.fmax),
                "--max-steps",
                str(args.max_steps),
                "--max-step",
                "0.2",
                "--alpha",
                "70.0",
                "--deterministic",
                "--seed",
                "20260729",
                "--model-dtype",
                "float32",
                "--optimizer-dtype",
                "float64",
                "--dataset-dir",
                str(args.dataset_dir),
                "--workload-manifest",
                str(shard_path),
                "--checkpoint",
                str(args.checkpoint),
                "--cpu-threads-per-worker",
                "1",
                "--output",
                str(worker_output),
            ]
            worker_environment = (
                per_gpu_environment[gpu] if per_gpu_environment is not None else environment
            )
            processes.append(
                (gpu, subprocess.Popen(command, env=worker_environment, text=True), worker_output)
            )
        failures = []
        for gpu, process, output_path in processes:
            returncode = process.wait()
            if returncode:
                failures.append((gpu, returncode, output_path))
        if failures:
            raise RuntimeError(f"MPS benchmark subprocess failures: {failures}")
    finally:
        if owns_mps_session:
            _stop_mps(environment)

    worker_outputs = []
    for gpu, indices, _, shard in shard_specs:
        result_path = args.runtime_dir / f"gpu-{gpu}.result.json"
        result = json.loads(result_path.read_text())
        worker_outputs.append(
            {
                "gpu": gpu,
                "source_indices": indices,
                "shard_manifest_sha256": shard.manifest_sha256,
                "result": result,
            }
        )
    records = [
        record
        for worker in worker_outputs
        for result in worker["result"]["worker_results"]
        for record in result["records"]
    ]
    expected = {job.system_id for job in manifest.jobs}
    if {record["source"] for record in records} != expected or len(records) != len(expected):
        raise RuntimeError("ASE/MPS baseline did not preserve exact job coverage")
    output = {
        "schema_version": 1,
        "status": "complete",
        "method": "ase_cuda_mps_static_lpt",
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "pool_size": len(manifest.jobs),
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "mps_session": (
            "external_per_gpu"
            if args.mps_session_config is not None
            else "reused"
            if args.reuse_active_mps
            else "owned"
        ),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "contract": {
            "optimizer": "ASE BFGS",
            "cell_filter": "ASE FrechetCellFilter",
            "cutoff_A": 6.0,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "max_step_A": 0.2,
            "alpha": 70.0,
            "model_dtype": "float32",
            "optimizer_dtype": "float64",
            "deterministic": True,
        },
        "timing": {
            "script_seconds": time.perf_counter() - started,
            "production_makespan_seconds": max(
                item["result"]["timing"]["wall_seconds"] for item in worker_outputs
            ),
        },
        "static_lpt": {
            "cost_per_gpu": [
                sum(_job_cost(manifest.jobs[index]) for index in indices) for indices in shards
            ],
            "jobs_per_gpu": [len(indices) for indices in shards],
        },
        "peak_gpu_memory_bytes_nvidia_smi": max(
            (item["result"]["peak_gpu_memory_bytes_nvidia_smi"] or 0 for item in worker_outputs),
            default=None,
        ),
        "model_evaluations": sum(
            item["result"]["model_evaluations_total"] for item in worker_outputs
        ),
        "optimizer_steps": sum(item["result"]["optimizer_steps_total"] for item in worker_outputs),
        "converged_count": sum(int(item["result"]["converged"] or 0) for item in worker_outputs),
        "workers": worker_outputs,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pool_size": len(manifest.jobs),
                "production_makespan_seconds": output["timing"]["production_makespan_seconds"],
                "converged_count": output["converged_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
