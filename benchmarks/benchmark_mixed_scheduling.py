#!/usr/bin/env python3
"""Benchmark FIFO, bucketed, and memory-planned signed mixed workloads."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_mace_variable_cell_scaling import (  # noqa: E402
    run_batch as run_mace_batch,
)
from benchmark_memory_planner import (  # noqa: E402
    calibrate,
    combine_bucket_outputs,
    serialize_plan,
)
from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_manifest,
    load_production_model,
    sha256_file,
    synchronize,
)
from benchmark_variable_cell_scaling import (  # noqa: E402
    run_batch as run_atombit_batch,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    BatchPlanner,
    MACEBatchCalculator,
    RuntimeProfiler,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def load_signed_systems(
    manifest_path: Path,
    dataset_dir: Path,
    *,
    job_limit: int | None = None,
):
    manifest = read_workload_manifest(manifest_path)
    systems = []
    jobs = manifest.jobs if job_limit is None else manifest.jobs[:job_limit]
    for job in jobs:
        atoms = read(dataset_dir / job.source_path, index=job.frame_index)
        atoms.info["benchmark_source"] = job.system_id
        atoms.info["benchmark_source_path"] = job.source_path
        systems.append(atoms)
    return manifest, systems


def total_predicted_bytes(planner: BatchPlanner, plan) -> int:
    return planner.estimate_profiles_bytes(plan.profiles)


def build_execution_schedule(
    *,
    mode: str,
    plan,
    workload_size: int,
    batch_size: int,
    total_predicted: int,
) -> tuple[list[tuple[tuple[int, ...], int, bool]], str]:
    all_indices = tuple(range(workload_size))
    if mode == "fifo":
        return [(all_indices, min(batch_size, workload_size), False)], "fifo"
    if mode == "refill":
        return [
            (all_indices, min(batch_size, workload_size), True)
        ], "fifo_active_refill"
    if mode == "bucketed":
        return [
            (
                bucket.system_indices,
                min(batch_size, len(bucket.system_indices)),
                False,
            )
            for bucket in plan.buckets
        ], "cost_buckets_fixed_capacity"
    if mode == "planned":
        return [
            (
                bucket.system_indices,
                min(bucket.resident_capacity, len(bucket.system_indices)),
                len(bucket.system_indices) > bucket.resident_capacity,
            )
            for bucket in plan.buckets
        ], "cost_buckets_planned_capacity"
    if mode == "auto":
        if (
            workload_size <= batch_size
            and total_predicted <= plan.memory_budget_bytes
        ):
            return [
                (all_indices, workload_size, False)
            ], "whole_batch_predicted_to_fit"
        return [
            (
                bucket.system_indices,
                min(bucket.resident_capacity, len(bucket.system_indices)),
                len(bucket.system_indices) > bucket.resident_capacity,
            )
            for bucket in plan.buckets
        ], "fallback_to_cost_buckets"
    raise ValueError(f"unsupported scheduling mode {mode!r}")


def execute_atombit(
    systems: list[Any],
    schedule: list[tuple[tuple[int, ...], int, bool]],
    *,
    model: torch.nn.Module,
    device: torch.device,
    skin: float,
    fmax: float,
    max_steps: int,
    linear_algebra_backend: str,
) -> dict[str, Any]:
    outputs = []
    for indices, capacity, refill in schedule:
        selected = [systems[index] for index in indices]
        output = run_atombit_batch(
            model,
            selected,
            batch_size=capacity,
            active_compaction=True,
            device=device,
            cutoff=6.0,
            skin=skin,
            fmax=fmax,
            max_steps=max_steps,
            dt_start=0.1,
            dt_max=1.0,
            max_step=0.2,
            optimizer_name="bfgs",
            alpha=70.0,
            optimizer_dtype="float64",
            model_dtype=torch.float32,
            neighbor_backend="auto",
            refill=refill,
            refill_policy="immediate",
            linear_algebra_backend=linear_algebra_backend,
        )
        outputs.append((indices, output))
    return combine_bucket_outputs(outputs, workload_size=len(systems))


def execute_mace(
    systems: list[Any],
    schedule: list[tuple[tuple[int, ...], int, bool]],
    *,
    calculator: MACEBatchCalculator,
    fmax: float,
    max_steps: int,
    linear_algebra_backend: str,
) -> dict[str, Any]:
    outputs = []
    for indices, capacity, refill in schedule:
        selected = [systems[index] for index in indices]
        output = run_mace_batch(
            calculator,
            selected,
            batch_size=capacity,
            active_compaction=True,
            fmax=fmax,
            max_steps=max_steps,
            dt_start=0.1,
            dt_max=1.0,
            max_step=0.2,
            optimizer_name="bfgs",
            alpha=70.0,
            refill=refill,
            refill_policy="immediate",
            linear_algebra_backend=linear_algebra_backend,
        )
        outputs.append((indices, output))
    return combine_bucket_outputs(outputs, workload_size=len(systems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument(
        "--mode",
        choices=("fifo", "refill", "bucketed", "planned", "auto"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--memory-budget-gib", type=float, default=64.0)
    parser.add_argument("--max-cost-ratio", type=float, default=2.0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--linear-algebra-backend",
        choices=("auto", "cholesky", "grouped", "serial"),
        default="auto",
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        default=Path(
            "runs/robustness/workloads/manifests/"
            "OPT-RB-CROSS-MIX-R192-v1.json"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument(
        "--calibration-dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("benchmarks/t2_fixed_samples.json"),
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("experiments/bfgs-active-refill/results.json"),
    )
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
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        parser.error("batch size and CPU thread count must be positive")
    if args.memory_budget_gib <= 0.0:
        parser.error("memory budget must be positive")

    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    manifest, systems = load_signed_systems(
        args.workload_manifest,
        args.dataset_dir,
    )
    cutoff = 6.0 if args.mlip == "atombit" else 5.0
    skin = 0.5 if args.mlip == "atombit" else 0.0
    calibration_manifest = load_manifest(args.calibration_manifest, 32)
    coefficients, calibration = calibrate(
        model=args.mlip,
        manifest=calibration_manifest,
        dataset_dir=args.calibration_dataset_dir,
        calibration_path=args.calibration_results,
        cutoff=cutoff,
    )
    planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
        max_batch_size=args.batch_size,
        max_cost_ratio=args.max_cost_ratio,
    )
    planning_started = time.perf_counter()
    plan = planner.plan(systems, cutoff=cutoff, skin=skin)
    planning_seconds = time.perf_counter() - planning_started
    predicted = total_predicted_bytes(planner, plan)
    schedule, decision = build_execution_schedule(
        mode=args.mode,
        plan=plan,
        workload_size=len(systems),
        batch_size=args.batch_size,
        total_predicted=predicted,
    )

    device = torch.device(args.device)
    if args.mlip == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        warm = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=skin,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
        warm(warm.create_state([systems[0]]), compute_stress=True)
        calculator = None
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
            skin=skin,
            neighbor_backend="auto",
        )
        calculator(
            calculator.create_state([systems[0]]),
            compute_stress=True,
        )
        model = None
        model_info = {"model": args.mace_model}
    synchronize(device)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with RuntimeProfiler(device=device) as profiler:
        if args.mlip == "atombit":
            if model is None:
                raise RuntimeError("AtomBit model was not initialized")
            output = execute_atombit(
                systems,
                schedule,
                model=model,
                device=device,
                skin=skin,
                fmax=args.fmax,
                max_steps=args.max_steps,
                linear_algebra_backend=args.linear_algebra_backend,
            )
        else:
            if calculator is None:
                raise RuntimeError("MACE calculator was not initialized")
            output = execute_mace(
                systems,
                schedule,
                calculator=calculator,
                fmax=args.fmax,
                max_steps=args.max_steps,
                linear_algebra_backend=args.linear_algebra_backend,
            )
    synchronize(device)
    optimization_seconds = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "status": "complete",
        "mlip": args.mlip,
        "mode": args.mode,
        "decision": decision,
        "optimizer": "bfgs",
        "jobs": len(systems),
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.workload_manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "parameters": {
            "batch_size": args.batch_size,
            "memory_budget_bytes": planner.memory_budget_bytes,
            "max_cost_ratio": args.max_cost_ratio,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "skin_A": skin,
            "cpu_threads": args.cpu_threads,
            "deterministic_algorithms": args.deterministic,
            "linear_algebra_backend": args.linear_algebra_backend,
        },
        "calibration": calibration,
        "plan": {
            **serialize_plan(plan),
            "total_workload_predicted_bytes": predicted,
        },
        "execution_buckets": [
            {
                "systems": len(indices),
                "resident_capacity": capacity,
                "refill": refill,
            }
            for indices, capacity, refill in schedule
        ],
        "planning_seconds": planning_seconds,
        "optimization_seconds": optimization_seconds,
        "end_to_end_seconds": planning_seconds + optimization_seconds,
        "systems_per_second": len(systems) / optimization_seconds,
        "end_to_end_systems_per_second": (
            len(systems) / (planning_seconds + optimization_seconds)
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_profile": profiler.summary(),
        "environment": environment_metadata(device),
        **model_info,
        **output,
        "converged": sum(
            bool(record["converged"]) for record in output["records"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": decision,
                "seconds": optimization_seconds,
                "systems_per_second": result["systems_per_second"],
                "converged": result["converged"],
                "peak_allocated_GiB": result["peak_allocated_bytes"] / 2**30,
                "peak_reserved_GiB": result["peak_reserved_bytes"] / 2**30,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
