#!/usr/bin/env python3
"""Benchmark plug-and-play online relaxation on a signed workload."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

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
    FrechetCellFilter,
    MACEBatchCalculator,
    relax,
)


def schedule_batches(schedule: dict[str, object]) -> list[dict[str, object]]:
    if "batches" in schedule:
        return list(schedule["batches"])
    batches = []
    for record in schedule.get("cold_start", []):
        batches.extend(schedule_batches(record["schedule"]))
    for worker in schedule.get("workers", []):
        for chunk in worker["chunks"]:
            batches.extend(schedule_batches(chunk["schedule"]))
    return batches


def schedule_cache_hits(schedule: dict[str, object]) -> list[bool]:
    hits = [
        bool(bucket["cache_hit"])
        for bucket in schedule.get("buckets", [])
    ]
    for record in schedule.get("cold_start", []):
        hits.extend(schedule_cache_hits(record["schedule"]))
    for worker in schedule.get("workers", []):
        for chunk in worker["chunks"]:
            hits.extend(schedule_cache_hits(chunk["schedule"]))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument("--optimizer", choices=("bfgs", "fire"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--devices",
        help="comma-separated devices for automatic multi-GPU execution",
    )
    parser.add_argument(
        "--worker-backend",
        choices=("auto", "process", "thread"),
        default="auto",
        help="multi-GPU execution backend",
    )
    parser.add_argument("--worker-cpu-threads", type=int, default=1)
    parser.add_argument("--process-min-chunks-per-device", type=int, default=8)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--initial-batch-size", type=int, default=1)
    parser.add_argument("--growth-factor", type=int, default=4)
    parser.add_argument("--memory-safety-fraction", type=float, default=0.85)
    parser.add_argument("--deterministic", action="store_true")
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
    max_steps = args.max_steps or (2_000 if args.optimizer == "fire" else 500)
    if args.clear_cache:
        args.cache_path.unlink(missing_ok=True)

    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, systems = load_signed_systems(
        args.workload_manifest,
        args.dataset_dir,
    )
    device = torch.device(args.device)
    devices = (
        None
        if args.devices is None
        else [
            torch.device(value.strip())
            for value in args.devices.split(",")
            if value.strip()
        ]
    )
    if args.mlip == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=0.0,
            device=device,
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
            device=device,
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
    synchronize(device)
    options = {
        "cell_filter": FrechetCellFilter(),
        "fmax": args.fmax,
        "smax": None,
        "max_steps": max_steps,
        "max_step": 0.2,
        "callback_interval": max_steps + 1,
    }
    if args.optimizer == "bfgs":
        options.update(
            {
                "alpha": 70.0,
                "optimizer_dtype": "float64",
                "linear_algebra_backend": "auto",
            }
        )
    else:
        options.update({"dt_start": 0.1, "dt_max": 1.0})
    config = AutoSchedulerConfig(
        cache_path=args.cache_path,
        initial_batch_size=args.initial_batch_size,
        growth_factor=args.growth_factor,
        max_batch_size=args.max_batch_size,
        memory_safety_fraction=args.memory_safety_fraction,
        multi_gpu_worker_backend=args.worker_backend,
        multi_gpu_process_cpu_threads=args.worker_cpu_threads,
        multi_gpu_process_min_chunks_per_device=(
            args.process_min_chunks_per_device
        ),
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = relax(
        systems,
        calculator,
        optimizer=args.optimizer,
        scheduling="auto",
        auto_config=config,
        devices=devices,
        **options,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    schedule = result.metadata["scheduling"]
    measured_batches = schedule_batches(schedule)
    peak_allocated_bytes = max(
        batch["peak_allocated_bytes"] or 0
        for batch in measured_batches
    )
    peak_reserved_bytes = max(
        batch["peak_reserved_bytes"] or 0
        for batch in measured_batches
    )
    output = {
        "schema_version": 1,
        "status": "complete",
        "mlip": args.mlip,
        "optimizer": args.optimizer,
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.workload_manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "jobs": len(systems),
        "parameters": {
            "fmax_eV_per_A": args.fmax,
            "max_steps": max_steps,
            "max_batch_size": args.max_batch_size,
            "initial_batch_size": args.initial_batch_size,
            "growth_factor": args.growth_factor,
            "memory_safety_fraction": args.memory_safety_fraction,
            "multi_gpu_worker_backend": args.worker_backend,
            "multi_gpu_process_cpu_threads": args.worker_cpu_threads,
            "multi_gpu_process_min_chunks_per_device": (
                args.process_min_chunks_per_device
            ),
            "deterministic_algorithms": args.deterministic,
        },
        "schedule": schedule,
        "wall_time_s": elapsed,
        "systems_per_second": len(systems) / elapsed,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "model_evaluations": result.model_evaluations,
        "graph_evaluations": result.graph_evaluations,
        "neighbor_rebuilds": result.state.neighbor_rebuild_count,
        "converged": int(result.converged.sum().item()),
        "converged_steps": result.converged_step.detach().cpu().tolist(),
        "energies_eV": result.evaluation.energy.detach().cpu().tolist(),
        "environment": environment_metadata(device),
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
                "seconds": elapsed,
                "converged": output["converged"],
                "cache_hits": schedule_cache_hits(schedule),
                "batch_sizes": [
                    batch["system_count"] for batch in measured_batches
                ],
                "resident_capacities": [
                    batch["resident_capacity"]
                    for batch in measured_batches
                ],
                "peak_reserved_GiB": output["peak_reserved_bytes"] / 2**30,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
