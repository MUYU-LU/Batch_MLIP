#!/usr/bin/env python3
"""Compare fresh and persistent automatic relaxation workers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_mixed_scheduling import load_signed_systems  # noqa: E402
from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    synchronize,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    BatchExecutor,
    FrechetCellFilter,
    MACEBatchCalculator,
    relax,
)


def _schedule_batches(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    batches = list(schedule.get("batches", ()))
    for key in ("cold_start", "workers"):
        for record in schedule.get(key, ()):
            if "schedule" in record:
                batches.extend(_schedule_batches(record["schedule"]))
            for chunk in record.get("chunks", ()):
                batches.extend(_schedule_batches(chunk["schedule"]))
    return batches


def _worker_chunks(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = []
    for worker in schedule.get("workers", ()):
        chunks.extend(worker.get("chunks", ()))
    return chunks


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _result_record(
    result,
    *,
    wall_seconds: float,
    jobs: int,
    atoms: int,
    store_final_tensors: bool,
) -> dict[str, Any]:
    schedule = result.metadata["scheduling"]
    batches = _schedule_batches(schedule)
    worker_chunks = _worker_chunks(schedule)
    memory_records = [*batches, *worker_chunks]
    record = {
        "wall_time_s": wall_seconds,
        "systems_per_s": jobs / wall_seconds,
        "atoms_per_s": atoms / wall_seconds,
        "peak_allocated_bytes": max(
            (batch.get("peak_allocated_bytes") or 0 for batch in memory_records),
            default=0,
        ),
        "peak_reserved_bytes": max(
            (batch.get("peak_reserved_bytes") or 0 for batch in memory_records),
            default=0,
        ),
        "converged": int(result.converged.sum().item()),
        "converged_steps": result.converged_step.detach().cpu().tolist(),
        "model_evaluations": result.model_evaluations,
        "graph_evaluations": result.graph_evaluations,
        "state_sha256": {
            "positions": _tensor_sha256(result.state.positions),
            "cells": _tensor_sha256(result.state.cells),
            "energies": _tensor_sha256(result.evaluation.energy),
            "forces": _tensor_sha256(result.evaluation.forces),
            "converged": _tensor_sha256(result.converged),
            "converged_step": _tensor_sha256(result.converged_step),
        },
        "schedule": schedule,
    }
    if store_final_tensors:
        record["final_tensors"] = {
            "positions_A": result.state.positions.detach().cpu().tolist(),
            "cells_A": result.state.cells.detach().cpu().tolist(),
            "energies_eV": result.evaluation.energy.detach().cpu().tolist(),
            "forces_eV_per_A": (
                result.evaluation.forces.detach().cpu().tolist()
            ),
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument("--mode", choices=("fresh", "persistent"), required=True)
    parser.add_argument("--optimizer", choices=("bfgs", "fire"), default="bfgs")
    parser.add_argument(
        "--optimizer-sequence",
        help="comma-separated bfgs/fire sequence with one entry per call",
    )
    parser.add_argument("--devices", required=True)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument(
        "--job-limit",
        type=int,
        help="Run the first N signed jobs; omit to run the complete workload.",
    )
    parser.add_argument("--resident-batch-size", type=int, default=128)
    parser.add_argument(
        "--automatic-capacity",
        action="store_true",
        help="Use unmodified AutoSchedulerConfig capacity defaults.",
    )
    parser.add_argument(
        "--memory-growth-margin",
        type=float,
        help="Override the analytical memory-growth margin for calibration.",
    )
    parser.add_argument(
        "--target-chunks-per-device",
        type=int,
        help="Override the pending work-stealing depth for calibration.",
    )
    parser.add_argument(
        "--multi-gpu-dispatch-policy",
        choices=("subdivide", "preserve_resident"),
        default="subdivide",
    )
    parser.add_argument(
        "--multi-gpu-queue-policy",
        choices=("cost_descending", "bucket_stratified"),
        default=AutoSchedulerConfig().multi_gpu_queue_policy,
    )
    parser.add_argument(
        "--cold-start-jobs",
        type=int,
        default=AutoSchedulerConfig().multi_gpu_cold_start_jobs,
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--store-final-tensors", action="store_true")
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/"
            "smooth_rms_finetune/"
            "AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
        ),
    )
    parser.add_argument("--mace-model", default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calls <= 0 or args.resident_batch_size <= 0:
        parser.error("calls and resident batch size must be positive")
    if args.job_limit is not None and args.job_limit <= 0:
        parser.error("job limit must be positive")
    if args.cold_start_jobs <= 0:
        parser.error("cold-start jobs must be positive")
    if (
        args.target_chunks_per_device is not None
        and args.target_chunks_per_device <= 0
    ):
        parser.error("target chunks per device must be positive")
    if (
        args.memory_growth_margin is not None
        and args.memory_growth_margin < 1.0
    ):
        parser.error("memory-growth margin must be at least one")
    if args.clear_cache:
        args.cache_path.unlink(missing_ok=True)

    devices = [
        value.strip() for value in args.devices.split(",") if value.strip()
    ]
    if not devices:
        parser.error("devices must contain at least one device")
    optimizer_sequence = (
        [args.optimizer] * args.calls
        if args.optimizer_sequence is None
        else [
            value.strip().lower()
            for value in args.optimizer_sequence.split(",")
            if value.strip()
        ]
    )
    if len(optimizer_sequence) != args.calls:
        parser.error("optimizer sequence must contain one entry per call")
    if any(value not in ("bfgs", "fire") for value in optimizer_sequence):
        parser.error("optimizer sequence entries must be bfgs or fire")
    max_steps = args.max_steps or (
        2_000 if "fire" in optimizer_sequence else 500
    )
    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, systems = load_signed_systems(
        args.workload_manifest,
        args.dataset_dir,
        job_limit=args.job_limit,
    )
    jobs = len(systems)
    atoms = sum(len(system) for system in systems)
    primary_device = torch.device(devices[0])
    if args.mlip == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        calculator = AtomBitBatchCalculator(
            model.to(device=primary_device, dtype=torch.float32).eval(),
            cutoff=6.0,
            skin=0.5,
            device=primary_device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
        model_info = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_metadata": model_metadata,
        }
    else:
        calculator = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=primary_device,
            dtype=torch.float64,
            graph_mode="rebuild",
            skin=0.0,
            neighbor_backend="auto",
        )
        model_info = {"model": args.mace_model}

    calculator(
        calculator.create_state([systems[0]]),
        compute_stress=True,
    )
    synchronize(primary_device)
    config_options: dict[str, Any] = {
        "cache_path": args.cache_path,
        "multi_gpu_cold_start_jobs": args.cold_start_jobs,
        "multi_gpu_worker_backend": "process",
        "multi_gpu_process_cpu_threads": 1,
        "multi_gpu_dispatch_policy": args.multi_gpu_dispatch_policy,
        "multi_gpu_queue_policy": args.multi_gpu_queue_policy,
    }
    if not args.automatic_capacity:
        config_options.update(
            {
                "initial_batch_size": args.resident_batch_size,
                "max_batch_size": args.resident_batch_size,
            }
        )
    if args.memory_growth_margin is not None:
        config_options["memory_growth_margin"] = args.memory_growth_margin
    if args.target_chunks_per_device is not None:
        config_options["multi_gpu_target_chunks_per_device"] = (
            args.target_chunks_per_device
        )
    config = AutoSchedulerConfig(
        **config_options,
    )

    def optimizer_options(name: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "cell_filter": FrechetCellFilter(),
            "fmax": args.fmax,
            "smax": None,
            "max_steps": max_steps,
            "max_step": 0.2,
            "callback_interval": max_steps + 1,
        }
        if name == "bfgs":
            options.update(
                {
                    "alpha": 70.0,
                    "optimizer_dtype": "float64",
                    "linear_algebra_backend": "auto",
                }
            )
        else:
            options.update({"dt_start": 0.1, "dt_max": 1.0})
        return options

    executor = (
        BatchExecutor(
            calculator,
            devices=devices,
            auto_config=config,
        )
        if args.mode == "persistent"
        else None
    )
    call_records = []
    session_started = time.perf_counter()
    executor_shutdown = None
    try:
        for current_optimizer in optimizer_sequence:
            options = optimizer_options(current_optimizer)
            gc.collect()
            synchronize(primary_device)
            started = time.perf_counter()
            if executor is None:
                result = relax(
                    systems,
                    calculator,
                    optimizer=current_optimizer,
                    scheduling="auto",
                    devices=devices,
                    auto_config=config,
                    **options,
                )
            else:
                result = executor.relax(
                    systems,
                    optimizer=current_optimizer,
                    **options,
                )
            synchronize(primary_device)
            record = _result_record(
                result,
                wall_seconds=time.perf_counter() - started,
                jobs=jobs,
                atoms=atoms,
                store_final_tensors=args.store_final_tensors,
            )
            record["optimizer"] = current_optimizer
            call_records.append(record)
    finally:
        if executor is not None:
            executor.close()
            executor_shutdown = executor.shutdown_metadata
    session_seconds = time.perf_counter() - session_started

    steady_records = call_records[1:] or call_records
    output = {
        "schema_version": 1,
        "status": "complete",
        "mlip": args.mlip,
        "mode": args.mode,
        "optimizer": (
            optimizer_sequence[0]
            if len(set(optimizer_sequence)) == 1
            else "mixed"
        ),
        "optimizer_sequence": optimizer_sequence,
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.workload_manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "jobs": jobs,
        "workload_jobs": len(manifest.jobs),
        "job_limit": args.job_limit,
        "atoms": atoms,
        "devices": devices,
        "gpu_count": len(devices),
        "calls": args.calls,
        "resident_batch_size": (
            None if args.automatic_capacity else args.resident_batch_size
        ),
        "automatic_capacity": args.automatic_capacity,
        "memory_growth_margin_override": args.memory_growth_margin,
        "multi_gpu_dispatch_policy": args.multi_gpu_dispatch_policy,
        "multi_gpu_queue_policy": args.multi_gpu_queue_policy,
        "target_chunks_per_device_override": (
            args.target_chunks_per_device
        ),
        "cold_start_jobs": args.cold_start_jobs,
        "fmax_eV_per_A": args.fmax,
        "max_steps": max_steps,
        "deterministic_algorithms": args.deterministic,
        "call_records": call_records,
        "first_call_wall_time_s": call_records[0]["wall_time_s"],
        "steady_state_mean_wall_time_s": sum(
            record["wall_time_s"] for record in steady_records
        )
        / len(steady_records),
        "session_wall_time_s": session_seconds,
        "executor_shutdown": executor_shutdown,
        "environment": environment_metadata(primary_device),
        **model_info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": args.mode,
                "first_call_s": output["first_call_wall_time_s"],
                "steady_mean_s": output["steady_state_mean_wall_time_s"],
                "calls": [
                    round(record["wall_time_s"], 6)
                    for record in call_records
                ],
                "peak_reserved_GiB": [
                    record["peak_reserved_bytes"] / 2**30
                    for record in call_records
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
