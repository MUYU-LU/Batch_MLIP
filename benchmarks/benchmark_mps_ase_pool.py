#!/usr/bin/env python3
"""Benchmark an ASE optimization pool with concurrent CUDA MPS workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument(
        "--optimizer",
        choices=("fire", "bfgs", "bfgslinesearch"),
        required=True,
    )
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--optimizer-dtype",
        choices=("state", "float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/t2_fixed_samples.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--worker-start-interval", type=float, default=0.0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pool_size <= 0 or args.workers <= 0 or args.cpu_threads_per_worker <= 0:
        parser.error("pool size, worker count, and CPU thread count must be positive")
    if args.worker_start_interval < 0:
        parser.error("worker start interval must be non-negative")
    if args.pool_size % args.workers:
        parser.error("pool size must be divisible by worker count")
    for variable in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        if not os.environ.get(variable):
            parser.error(f"{variable} must identify the active MPS daemon")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_impl(
    worker_id: int,
    args: argparse.Namespace,
    barrier: Any,
    result_path: Path,
) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"logm result may be inaccurate.*",
        category=RuntimeWarning,
    )
    import torch
    from ase.calculators.calculator import all_changes
    from ase.io import read

    benchmark_dir = Path(__file__).resolve().parent
    repository_root = benchmark_dir.parent
    sys.path[:0] = [str(repository_root), str(benchmark_dir)]
    from benchmark_production import load_manifest, load_production_model, synchronize
    from benchmark_variable_cell_scaling import (
        AtomBitBatchCalculator,
        AtomBitCalculator,
        run_ase,
    )

    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(args.cpu_threads_per_worker)
    torch.set_num_interop_threads(1)
    shard_size = args.pool_size // args.workers
    manifest = load_manifest(args.manifest, min(args.pool_size, 32))
    available = manifest["samples"][str(args.atom_count)]
    base_names = available[: min(args.pool_size, len(available))]
    global_names = [base_names[index % len(base_names)] for index in range(args.pool_size)]
    start = worker_id * shard_size
    names = global_names[start : start + shard_size]

    systems = []
    for name in names:
        atoms = read(args.dataset_dir / name)
        if len(atoms) != args.atom_count:
            raise ValueError(f"{name} has {len(atoms)} atoms")
        atoms.info["benchmark_source"] = name
        systems.append(atoms)

    device = torch.device(args.device)
    model_dtype = getattr(torch, args.model_dtype)
    optimizer_dtype = (
        None if args.optimizer_dtype == "state" else args.optimizer_dtype
    )
    model, _ = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=model_dtype).eval()

    # Exercise the same model, force, stress, and autograd paths before the barrier.
    warm_batch = AtomBitBatchCalculator(
        model,
        cutoff=args.cutoff,
        device=device,
        dtype=model_dtype,
        force_mode="autograd",
    )
    warm_batch(warm_batch.create_state([systems[0]]), compute_stress=True)
    warm_ase = AtomBitCalculator(
        model,
        cutoff=args.cutoff,
        device=device,
        dtype=model_dtype,
        enable_stress=True,
        add_e0=False,
    )
    warm_ase.calculate(
        systems[0],
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    )
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    barrier.wait(timeout=600)
    started = time.perf_counter()
    output = run_ase(
        model,
        systems,
        device=device,
        cutoff=args.cutoff,
        fmax=args.fmax,
        max_steps=args.max_steps,
        dt_start=args.dt_start,
        dt_max=args.dt_max,
        max_step=args.max_step,
        optimizer_name=args.optimizer,
        alpha=args.alpha,
        optimizer_dtype=optimizer_dtype,
        model_dtype=model_dtype,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "worker_id": worker_id,
        "status": "passed",
        "sample_files": names,
        "elapsed_seconds": elapsed,
        "systems_per_second": shard_size / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
        **output,
    }
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")


def _worker(
    worker_id: int,
    args: argparse.Namespace,
    barrier: Any,
    result_path: Path,
) -> None:
    try:
        _worker_impl(worker_id, args, barrier, result_path)
    except BaseException as exc:
        result_path.write_text(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        barrier.abort()
        raise


def _gpu_memory_monitor(
    gpu_index: int,
    interval: float,
    stop: threading.Event,
    samples: list[int],
) -> None:
    while not stop.is_set():
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            try:
                samples.append(int(completed.stdout.strip()) * 1024 * 1024)
            except ValueError:
                pass
        stop.wait(interval)


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    thread_count = str(args.cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.workers + 1)
    worker_paths = [
        args.output.with_suffix(f".worker-{worker_id}.json")
        for worker_id in range(args.workers)
    ]
    for worker_path in worker_paths:
        worker_path.unlink(missing_ok=True)
    processes = [
        context.Process(
            target=_worker,
            args=(worker_id, args, barrier, worker_paths[worker_id]),
            name=f"mps-ase-worker-{worker_id}",
        )
        for worker_id in range(args.workers)
    ]
    for process in processes:
        process.start()
        if args.worker_start_interval:
            time.sleep(args.worker_start_interval)

    try:
        barrier.wait(timeout=600)
    except threading.BrokenBarrierError:
        for process in processes:
            process.join(timeout=1)
        failures = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode != 0
        }
        raise RuntimeError(f"MPS workers failed during warmup: {failures}") from None

    memory_samples: list[int] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_gpu_memory_monitor,
        args=(
            args.gpu_index,
            args.memory_sample_interval,
            stop_monitor,
            memory_samples,
        ),
        daemon=True,
    )
    monitor.start()
    started = time.perf_counter()
    while not all(worker_path.exists() for worker_path in worker_paths):
        if all(process.exitcode is not None for process in processes):
            break
        time.sleep(0.01)
    elapsed = time.perf_counter() - started
    stop_monitor.set()
    monitor.join()
    for process in processes:
        process.join()

    worker_results = [
        json.loads(worker_path.read_text(encoding="utf-8"))
        for worker_path in worker_paths
        if worker_path.exists()
    ]
    worker_results.sort(key=lambda item: item["worker_id"])
    failures = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode != 0
    }
    if failures or len(worker_results) != args.workers:
        raise RuntimeError(
            f"MPS worker failure: exit_codes={failures}, "
            f"results={len(worker_results)}/{args.workers}"
        )
    failed_results = [
        worker_result
        for worker_result in worker_results
        if worker_result["status"] != "passed"
    ]
    if failed_results:
        raise RuntimeError(f"MPS workers reported failures: {failed_results}")

    records = [
        record
        for worker_result in worker_results
        for record in worker_result["records"]
    ]
    result = {
        "schema_version": 1,
        "status": "complete",
        "method": "ase_cuda_mps",
        "optimizer": args.optimizer,
        "atom_count": args.atom_count,
        "pool_size": args.pool_size,
        "workers": args.workers,
        "systems_per_worker": args.pool_size // args.workers,
        "mps": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pipe_directory": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
            "log_directory": os.environ["CUDA_MPS_LOG_DIRECTORY"],
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "parameters": {
            "cutoff_A": args.cutoff,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "dt_start": args.dt_start,
            "dt_max": args.dt_max,
            "max_step_A": args.max_step,
            "bfgs_alpha_eV_per_A2": args.alpha,
            "model_dtype": args.model_dtype,
            "optimizer_dtype": args.optimizer_dtype,
            "cpu_threads_per_worker": args.cpu_threads_per_worker,
            "worker_start_interval_seconds": args.worker_start_interval,
            "deterministic_algorithms": args.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cell_filter": "ASE FrechetCellFilter",
        },
        "timing": {
            "wall_seconds": elapsed,
            "systems_per_second": args.pool_size / elapsed,
            "worker_seconds": [
                worker_result["elapsed_seconds"] for worker_result in worker_results
            ],
        },
        "peak_gpu_memory_bytes_nvidia_smi": max(memory_samples, default=None),
        "gpu_memory_samples": len(memory_samples),
        "converged": sum(bool(record["converged"]) for record in records),
        "optimizer_steps_total": sum(
            worker_result["optimizer_steps_total"]
            for worker_result in worker_results
        ),
        "model_evaluations_total": sum(
            worker_result["model_evaluations"]
            for worker_result in worker_results
        ),
        "worker_results": worker_results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for worker_path in worker_paths:
        worker_path.unlink()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "wall_seconds": elapsed,
                "systems_per_second": args.pool_size / elapsed,
                "converged": result["converged"],
                "peak_gpu_memory_bytes_nvidia_smi": result[
                    "peak_gpu_memory_bytes_nvidia_smi"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
