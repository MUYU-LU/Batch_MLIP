#!/usr/bin/env python3
"""Benchmark active drain versus per-worker refill on signed GPU shards."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_multi_gpu_sharding import (  # noqa: E402
    combine_outputs,
    gpu_metadata,
)
from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    synchronize,
    write_result,
)
from benchmark_robustness_optimization import _systems  # noqa: E402
from benchmark_variable_cell_scaling import run_batch  # noqa: E402

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    RuntimeProfiler,
    WorkerShard,
    balance_work,
    run_parallel_workers,
)


def _devices(value: str) -> list[str]:
    devices = [
        f"cuda:{item.strip()}" if item.strip().isdigit() else item.strip()
        for item in value.split(",")
        if item.strip()
    ]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


@dataclass
class Runner:
    model: torch.nn.Module
    systems: list[Any]
    device: torch.device
    resident_capacity: int
    refill: bool

    def __call__(self) -> dict[str, Any]:
        synchronize(self.device)
        with RuntimeProfiler(device=self.device) as profiler:
            result = run_batch(
                self.model,
                self.systems,
                batch_size=min(self.resident_capacity, len(self.systems)),
                active_compaction=True,
                device=self.device,
                cutoff=6.0,
                skin=0.5,
                fmax=0.05,
                max_steps=500,
                dt_start=0.1,
                dt_max=1.0,
                max_step=0.2,
                optimizer_name="bfgs",
                alpha=70.0,
                optimizer_dtype="float64",
                model_dtype=torch.float32,
                neighbor_backend="auto",
                refill=self.refill,
                refill_policy="immediate",
                refill_storage="slots",
                refill_min_chunk=1 if self.refill else None,
                refill_interval=1,
                convergence_check_interval=1,
                linear_algebra_backend="auto",
            )
        synchronize(self.device)
        result["runtime_profile"] = profiler.summary()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            self.device
        )
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            self.device
        )
        result["peak_memory_bytes"] = result["peak_allocated_bytes"]
        result["device_metadata"] = gpu_metadata(self.device)
        return result


@dataclass
class Preparer:
    systems: list[Any]
    checkpoint: str
    resident_capacity: int
    refill: bool

    def __call__(self, shard: WorkerShard) -> Runner:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device(shard.device)
        torch.cuda.set_device(device)
        selected = [self.systems[index] for index in shard.system_indices]
        model, _ = load_production_model(Path(self.checkpoint))
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=0.5,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
        calculator(
            calculator.create_state([selected[0]]),
            compute_stress=True,
        )
        synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return Runner(
            model,
            selected,
            device,
            self.resident_capacity,
            self.refill,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("active", "refill"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", type=_devices, required=True)
    parser.add_argument("--resident-capacity", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.resident_capacity <= 0:
        parser.error("resident capacity must be positive")
    manifest, systems = _systems(args.manifest, args.dataset_dir, None)
    if len(systems) % len(args.devices):
        parser.error("pool size must divide evenly across workers")
    worker_pool_size = len(systems) // len(args.devices)
    if worker_pool_size <= args.resident_capacity:
        parser.error("worker pool must exceed resident capacity")
    if any(len(atoms) != len(systems[0]) for atoms in systems):
        parser.error("slot refill transfer requires homogeneous atom counts")

    costs = [float((3 * len(atoms) + 9) ** 2) for atoms in systems]
    shards = balance_work(costs, args.devices)
    if any(len(shard.system_indices) != worker_pool_size for shard in shards):
        raise RuntimeError("static sharding did not produce equal worker pools")
    result = {
        "schema_version": 1,
        "status": "running",
        "method": args.method,
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "pool_size": len(systems),
        "unique_structure_count": len(
            {
                job.normalized_structure_sha256
                for job in manifest.jobs
            }
        ),
        "gpu_count": len(args.devices),
        "worker_pool_size": worker_pool_size,
        "resident_capacity": args.resident_capacity,
        "devices": args.devices,
        "worker_system_indices": [
            list(shard.system_indices) for shard in shards
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "execution_contract": {
            "model_dtype": "torch.float32",
            "optimizer_dtype": "torch.float64",
            "optimizer": "BatchedBFGS",
            "cell_filter": "BatchedFrechetCellFilter",
            "active_compaction": True,
            "refill_policy": (
                "immediate" if args.method == "refill" else None
            ),
            "refill_storage": (
                "slots" if args.method == "refill" else None
            ),
            "cutoff_A": 6.0,
            "skin_A": 0.5,
            "fmax_eV_per_A": 0.05,
            "smax_eV_per_A3": None,
            "max_steps": 500,
            "linear_algebra_backend": "auto",
            "deterministic_algorithms": True,
            "cpu_threads_per_worker": 1,
            "allocator_environment": {
                "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF"
                ),
            },
        },
        "environment": environment_metadata(torch.device("cpu")),
    }
    write_result(args.output, result)
    execution = run_parallel_workers(
        shards,
        Preparer(
            systems,
            str(args.checkpoint),
            args.resident_capacity,
            args.method == "refill",
        ),
    )
    combined = combine_outputs(execution, workload_size=len(systems))
    peaks_allocated = [
        worker.payload["peak_allocated_bytes"]
        for worker in execution.worker_results
    ]
    peaks_reserved = [
        worker.payload["peak_reserved_bytes"]
        for worker in execution.worker_results
    ]
    result.update(
        {
            "status": "passed",
            "startup_wall_seconds": execution.startup_wall_seconds,
            "optimization_wall_seconds": execution.run_wall_seconds,
            "end_to_end_wall_seconds": execution.end_to_end_wall_seconds,
            "systems_per_second": len(systems) / execution.run_wall_seconds,
            "atoms_per_second": sum(len(atoms) for atoms in systems)
            / execution.run_wall_seconds,
            "all_converged": all(
                record["converged"] for record in combined["records"]
            ),
            "converged": sum(
                record["converged"] for record in combined["records"]
            ),
            "peak_allocated_bytes_max_per_worker": max(peaks_allocated),
            "peak_reserved_bytes_max_per_worker": max(peaks_reserved),
            **combined,
        }
    )
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
