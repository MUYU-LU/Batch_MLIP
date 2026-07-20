#!/usr/bin/env python3
"""Benchmark independent-process multi-GPU BFGS workload sharding."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_manifest,
    load_production_model,
    synchronize,
    write_result,
)
from benchmark_variable_cell_scaling import run_batch as run_atombit_batch  # noqa: E402

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    MACEBatchCalculator,
    RuntimeProfiler,
    WorkerShard,
    balance_work,
    run_parallel_workers,
)


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("expected at least one device")
    return devices


def load_group(
    *,
    atom_count: int,
    count: int,
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> list[Atoms]:
    names = manifest["samples"][str(atom_count)]
    systems = []
    for index in range(count):
        name = names[index % len(names)]
        atoms = read(dataset_dir / name)
        if len(atoms) != atom_count:
            raise ValueError(f"{name} has {len(atoms)} atoms, expected {atom_count}")
        atoms.info["benchmark_source"] = name
        systems.append(atoms)
    return systems


def load_workload(
    name: str,
    *,
    size: int,
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> list[Atoms]:
    if name == "homogeneous-276":
        return load_group(
            atom_count=276,
            count=size,
            manifest=manifest,
            dataset_dir=dataset_dir,
        )
    if name != "mixed-46-276":
        raise ValueError(f"unsupported workload {name!r}")
    small_count = size // 2
    large_count = size - small_count
    small = load_group(
        atom_count=46,
        count=small_count,
        manifest=manifest,
        dataset_dir=dataset_dir,
    )
    large = load_group(
        atom_count=276,
        count=large_count,
        manifest=manifest,
        dataset_dir=dataset_dir,
    )
    systems = []
    for index in range(max(small_count, large_count)):
        if index < small_count:
            systems.append(small[index])
        if index < large_count:
            systems.append(large[index])
    return systems


def synchronize_and_peak(device: torch.device) -> int | None:
    synchronize(device)
    return (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )


def gpu_metadata(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device)}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_version": torch.version.cuda,
    }


@dataclass
class AtomBitRunner:
    model: torch.nn.Module
    systems: list[Atoms]
    device: torch.device
    batch_size: int

    def __call__(self) -> dict[str, Any]:
        synchronize(self.device)
        with RuntimeProfiler(device=self.device) as profiler:
            output = run_atombit_batch(
                self.model,
                self.systems,
                batch_size=min(self.batch_size, len(self.systems)),
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
                refill=True,
                refill_policy="immediate",
            )
        output["runtime_profile"] = profiler.summary()
        output["peak_memory_bytes"] = synchronize_and_peak(self.device)
        output["device_metadata"] = gpu_metadata(self.device)
        return output


@dataclass
class AtomBitPreparer:
    systems: list[Atoms]
    checkpoint: str
    batch_size: int
    deterministic: bool

    def __call__(self, shard: WorkerShard) -> AtomBitRunner:
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(self.deterministic)
        device = torch.device(shard.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        selected = [self.systems[index] for index in shard.system_indices]
        model, _ = load_production_model(Path(self.checkpoint))
        model = model.to(device=device, dtype=torch.float32).eval()
        warm = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=0.5,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
        )
        warm(warm.create_state([selected[0]]), compute_stress=True)
        synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        return AtomBitRunner(model, selected, device, self.batch_size)


@dataclass
class MACERunner:
    calculator: MACEBatchCalculator
    systems: list[Atoms]
    device: torch.device
    batch_size: int

    def __call__(self) -> dict[str, Any]:
        from benchmark_mace_variable_cell_scaling import run_batch as run_mace_batch

        synchronize(self.device)
        with RuntimeProfiler(device=self.device) as profiler:
            output = run_mace_batch(
                self.calculator,
                self.systems,
                batch_size=min(self.batch_size, len(self.systems)),
                active_compaction=True,
                fmax=0.05,
                max_steps=500,
                dt_start=0.1,
                dt_max=1.0,
                max_step=0.2,
                optimizer_name="bfgs",
                alpha=70.0,
                refill=True,
                refill_policy="immediate",
            )
        output["runtime_profile"] = profiler.summary()
        output["peak_memory_bytes"] = synchronize_and_peak(self.device)
        output["device_metadata"] = gpu_metadata(self.device)
        return output


@dataclass
class MACEPreparer:
    systems: list[Atoms]
    batch_size: int
    deterministic: bool

    def __call__(self, shard: WorkerShard) -> MACERunner:
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(self.deterministic)
        device = torch.device(shard.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        selected = [self.systems[index] for index in shard.system_indices]
        calculator = MACEBatchCalculator.from_off(
            model="small",
            device=device,
            dtype=torch.float64,
            graph_mode="cached",
            skin=0.5,
        )
        calculator(
            calculator.create_state([selected[0]]), compute_stress=True
        )
        synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        return MACERunner(calculator, selected, device, self.batch_size)


def combine_outputs(execution, *, workload_size: int) -> dict[str, Any]:
    records: list[dict[str, Any] | None] = [None] * workload_size
    totals = {
        "model_evaluations": 0,
        "graph_evaluations": 0,
        "uncompacted_graph_evaluations": 0,
        "avoided_graph_evaluations": 0,
        "neighbor_rebuilds": 0,
        "optimizer_steps_total": 0,
    }
    worker_data = []
    for worker_result in execution.worker_results:
        output = worker_result.payload
        indices = worker_result.shard.system_indices
        for index, record in zip(indices, output["records"], strict=True):
            if records[index] is not None:
                raise RuntimeError(f"duplicate output for input index {index}")
            records[index] = record
        for key in totals:
            totals[key] += int(output[key])
        worker_data.append(
            {
                "worker_id": worker_result.shard.worker_id,
                "device": worker_result.shard.device,
                "system_indices": list(indices),
                "system_count": len(indices),
                "estimated_cost": worker_result.shard.estimated_cost,
                "startup_seconds": worker_result.startup_seconds,
                "run_seconds": worker_result.run_seconds,
                "peak_memory_bytes": output["peak_memory_bytes"],
                "device_metadata": output["device_metadata"],
                "runtime_profile": output["runtime_profile"],
                "active_batch_sizes": output["active_batch_sizes"],
                "model_evaluations": output["model_evaluations"],
                "neighbor_rebuilds": output["neighbor_rebuilds"],
            }
        )
    if any(record is None for record in records):
        missing = [index for index, record in enumerate(records) if record is None]
        raise RuntimeError(f"missing outputs for indices {missing}")
    return {"records": records, "workers": worker_data, **totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("atombit", "mace"), required=True)
    parser.add_argument(
        "--workload",
        choices=("homogeneous-276", "mixed-46-276"),
        required=True,
    )
    parser.add_argument("--workload-size", type=int, default=256)
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workload_size <= 0:
        raise ValueError("workload size must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(args.devices) > args.workload_size:
        raise ValueError("worker count cannot exceed workload size")

    manifest = load_manifest(args.manifest, min(args.workload_size, 32))
    systems = load_workload(
        args.workload,
        size=args.workload_size,
        manifest=manifest,
        dataset_dir=args.dataset_dir,
    )
    costs = [float((3 * len(atoms) + 9) ** 2) for atoms in systems]
    shards = balance_work(costs, args.devices)
    result = {
        "schema_version": 1,
        "status": "running",
        "model": args.model,
        "workload": args.workload,
        "workload_size": len(systems),
        "atom_count_distribution": dict(Counter(len(atoms) for atoms in systems)),
        "total_atoms": sum(len(atoms) for atoms in systems),
        "worker_count": len(shards),
        "devices": args.devices,
        "parameters": {
            "optimizer": "bfgs",
            "cell_filter": "FrechetCellFilter",
            "resident_batch_size": args.batch_size,
            "active_compaction": True,
            "refill_policy": "immediate",
            "graph_mode": "cached",
            "skin_A": 0.5,
            "deterministic": args.deterministic,
            "cost_estimate": "(3 * atom_count + 9)^2",
        },
        "environment": environment_metadata(torch.device("cpu")),
    }
    write_result(args.output, result)
    if args.model == "atombit":
        preparer: Any = AtomBitPreparer(
            systems, str(args.checkpoint), args.batch_size, args.deterministic
        )
    else:
        preparer = MACEPreparer(systems, args.batch_size, args.deterministic)

    execution = run_parallel_workers(shards, preparer)
    combined = combine_outputs(execution, workload_size=len(systems))
    result.update(
        {
            "status": "complete",
            "startup_wall_seconds": execution.startup_wall_seconds,
            "optimization_wall_seconds": execution.run_wall_seconds,
            "end_to_end_wall_seconds": execution.end_to_end_wall_seconds,
            "systems_per_second": len(systems) / execution.run_wall_seconds,
            "atoms_per_second": sum(len(atoms) for atoms in systems)
            / execution.run_wall_seconds,
            "end_to_end_systems_per_second": len(systems)
            / execution.end_to_end_wall_seconds,
            "all_converged": all(
                record["converged"] for record in combined["records"]
            ),
            **combined,
        }
    )
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
