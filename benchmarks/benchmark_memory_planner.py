#!/usr/bin/env python3
"""Benchmark memory-aware BFGS bucketing on a fixed mixed workload."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    load_manifest,
    load_production_model,
    synchronize,
    write_result,
)
from benchmark_variable_cell_scaling import run_batch as run_atombit_batch  # noqa: E402

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    BatchPlanner,
    CalibrationObservation,
    MemoryCoefficients,
    RuntimeProfiler,
    fit_memory_coefficients,
)


def load_count_group(
    *,
    atom_count: int,
    count: int,
    manifest: dict[str, Any],
    dataset_dir: Path,
) -> list[Any]:
    names = manifest["samples"][str(atom_count)][: min(count, 32)]
    names = [names[index % len(names)] for index in range(count)]
    systems = []
    for name in names:
        atoms = read(dataset_dir / name)
        if len(atoms) != atom_count:
            raise ValueError(f"{name} has {len(atoms)} atoms")
        atoms.info["benchmark_source"] = name
        systems.append(atoms)
    return systems


def mixed_workload(
    *,
    manifest: dict[str, Any],
    dataset_dir: Path,
    atom_counts: tuple[int, ...] = (46, 276),
) -> list[Any]:
    if 256 % len(atom_counts):
        raise ValueError("mixed workload groups must divide 256 systems")
    count_per_group = 256 // len(atom_counts)
    groups = [
        load_count_group(
            atom_count=atom_count,
            count=count_per_group,
            manifest=manifest,
            dataset_dir=dataset_dir,
        )
        for atom_count in atom_counts
    ]
    return [
        group[index]
        for index in range(count_per_group)
        for group in groups
    ]


def calibration_peaks(path: Path, model: str) -> dict[int, dict[int, int]]:
    data = json.loads(path.read_text())
    groups = data["groups"][model]
    return {
        int(atom_count): {
            int(batch_size): point["active_refill"]["peak_memory_bytes"]
            for batch_size, point in group["points"].items()
        }
        for atom_count, group in groups.items()
    }


def calibrate(
    *,
    model: str,
    manifest: dict[str, Any],
    dataset_dir: Path,
    calibration_path: Path,
    cutoff: float,
) -> tuple[MemoryCoefficients, dict[str, Any]]:
    peaks = calibration_peaks(calibration_path, model)
    profiler = BatchPlanner(
        MemoryCoefficients(0.0, 0.0, 0.0, 8.0),
        memory_budget_bytes=2**63 - 1,
    )
    observations = []
    edge_counts: dict[int, int] = {}
    for atom_count in (46, 92, 184, 276):
        systems = load_count_group(
            atom_count=atom_count,
            count=32,
            manifest=manifest,
            dataset_dir=dataset_dir,
        )
        profiles = profiler.profile_systems(systems, cutoff=cutoff)
        total_edges = sum(profile.edge_count for profile in profiles)
        edge_counts[atom_count] = total_edges
        dof = 3 * atom_count + 9
        observations.append(
            CalibrationObservation(
                atom_count=atom_count * 64,
                edge_count=total_edges * 2,
                dof_squared=dof * dof * 64,
                peak_memory_bytes=peaks[atom_count][64],
            )
        )

    coefficients = fit_memory_coefficients(observations, optimizer_itemsize=8)
    validation = {}
    for atom_count in (46, 92, 184, 276):
        dof = 3 * atom_count + 9
        predicted = coefficients.estimate(
            atom_count=atom_count * 128,
            edge_count=edge_counts[atom_count] * 4,
            dof_squared=dof * dof * 128,
        )
        measured = peaks[atom_count][128]
        validation[str(atom_count)] = {
            "predicted_bytes": predicted,
            "measured_bytes": measured,
            "relative_error": (predicted - measured) / measured,
        }
    return coefficients, {
        "fit_batch_size": 64,
        "validation_batch_size": 128,
        "edge_counts_first_32": {str(key): value for key, value in edge_counts.items()},
        "coefficients": {
            "fixed_bytes": coefficients.fixed_bytes,
            "bytes_per_atom": coefficients.bytes_per_atom,
            "bytes_per_edge": coefficients.bytes_per_edge,
            "bytes_per_dof_squared": coefficients.bytes_per_dof_squared,
        },
        "validation": validation,
        "max_abs_validation_error": max(
            abs(item["relative_error"]) for item in validation.values()
        ),
    }


def serialize_plan(plan) -> dict[str, Any]:
    profiles = {profile.index: profile for profile in plan.profiles}
    return {
        "memory_budget_bytes": plan.memory_budget_bytes,
        "profiling_seconds": plan.profiling_seconds,
        "buckets": [
            {
                "system_count": len(bucket.system_indices),
                "atom_counts": sorted(
                    {profiles[index].atom_count for index in bucket.system_indices}
                ),
                "resident_capacity": bucket.resident_capacity,
                "predicted_peak_bytes": bucket.predicted_peak_bytes,
                "max_system_bytes": bucket.max_system_bytes,
            }
            for bucket in plan.buckets
        ],
    }


def combine_bucket_outputs(
    outputs: list[tuple[tuple[int, ...], dict[str, Any]]],
    *,
    workload_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any] | None] = [None] * workload_size
    result: dict[str, Any] = {
        "model_evaluations": 0,
        "graph_evaluations": 0,
        "uncompacted_graph_evaluations": 0,
        "avoided_graph_evaluations": 0,
        "neighbor_rebuilds": 0,
        "optimizer_steps_total": 0,
        "active_batch_sizes": [],
    }
    for indices, output in outputs:
        for index, record in zip(indices, output["records"], strict=True):
            records[index] = record
        for key in (
            "model_evaluations",
            "graph_evaluations",
            "uncompacted_graph_evaluations",
            "avoided_graph_evaluations",
            "neighbor_rebuilds",
            "optimizer_steps_total",
        ):
            result[key] += output[key]
        result["active_batch_sizes"].extend(output["active_batch_sizes"])
    if any(record is None for record in records):
        raise RuntimeError("planned execution did not return every input structure")
    result["records"] = records
    return result


def execute_atombit(
    systems: list[Any],
    buckets,
    *,
    device: torch.device,
    model: torch.nn.Module,
    linear_algebra_backend: str,
    skin: float,
    fmax: float,
    max_steps: int,
) -> dict[str, Any]:
    outputs = []
    for indices, capacity in buckets:
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
            refill=True,
            refill_policy="immediate",
            linear_algebra_backend=linear_algebra_backend,
        )
        outputs.append((indices, output))
    return combine_bucket_outputs(outputs, workload_size=len(systems))


def execute_mace(
    systems: list[Any],
    buckets,
    *,
    calculator: Any,
    linear_algebra_backend: str,
    fmax: float,
    max_steps: int,
) -> dict[str, Any]:
    from benchmark_mace_variable_cell_scaling import run_batch as run_mace_batch

    outputs = []
    for indices, capacity in buckets:
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
            refill=True,
            refill_policy="immediate",
            linear_algebra_backend=linear_algebra_backend,
        )
        outputs.append((indices, output))
    return combine_bucket_outputs(outputs, workload_size=len(systems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("atombit", "mace"), required=True)
    parser.add_argument(
        "--mode", choices=("fixed64", "fixed128", "planned"), required=True
    )
    parser.add_argument(
        "--workload",
        choices=("mixed46_276", "mixed4"),
        default="mixed46_276",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-budget-gib", type=float, default=32.0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-cost-ratio", type=float, default=2.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skin", type=float)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--linear-algebra-backend",
        choices=("auto", "grouped", "serial"),
        default="auto",
    )
    parser.add_argument(
        "--mace-graph-mode",
        choices=("cached", "rebuild"),
        default="rebuild",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("experiments/bfgs-active-refill/results.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.memory_budget_gib <= 0.0:
        raise ValueError("memory budget must be positive")
    torch.use_deterministic_algorithms(args.deterministic)
    manifest = load_manifest(args.manifest, 32)
    cutoff = 6.0 if args.model == "atombit" else 5.0
    skin = args.skin
    if skin is None:
        skin = 0.5 if args.model == "atombit" else 0.0
    coefficients, calibration = calibrate(
        model=args.model,
        manifest=manifest,
        dataset_dir=args.dataset_dir,
        calibration_path=args.calibration_results,
        cutoff=cutoff,
    )
    atom_counts = (
        (46, 92, 184, 276)
        if args.workload == "mixed4"
        else (46, 276)
    )
    systems = mixed_workload(
        manifest=manifest,
        dataset_dir=args.dataset_dir,
        atom_counts=atom_counts,
    )
    planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
        max_batch_size=args.max_batch_size,
        max_cost_ratio=args.max_cost_ratio,
    )
    plan = planner.plan(
        systems,
        cutoff=cutoff,
        skin=skin,
    )
    plan_data = serialize_plan(plan)
    result = {
        "schema_version": 1,
        "status": "planned" if args.plan_only else "running",
        "model": args.model,
        "mode": args.mode,
        "workload": {
            "name": args.workload,
            "systems": 256,
            "atoms": {
                str(atom_count): 256 // len(atom_counts)
                for atom_count in atom_counts
            },
        },
        "calibration": calibration,
        "plan": plan_data,
        "parameters": {
            "memory_budget_gib": args.memory_budget_gib,
            "max_batch_size": args.max_batch_size,
            "max_cost_ratio": args.max_cost_ratio,
            "deterministic": args.deterministic,
            "skin_A": skin,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "linear_algebra_backend": args.linear_algebra_backend,
            "mace_graph_mode": args.mace_graph_mode,
        },
    }
    write_result(args.output, result)
    if args.plan_only:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    all_indices = tuple(range(len(systems)))
    fixed_capacity = {"fixed64": 64, "fixed128": 128}.get(args.mode)
    buckets = (
        [(all_indices, fixed_capacity)]
        if fixed_capacity is not None
        else [
            (bucket.system_indices, bucket.resident_capacity)
            for bucket in plan.buckets
        ]
    )
    device = torch.device(args.device)
    if args.model == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        warm = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=skin,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
        )
        warm(warm.create_state([systems[0]]), compute_stress=True)
        calculator = None
    else:
        from batch_mlip import MACEBatchCalculator

        calculator = MACEBatchCalculator.from_off(
            model="small",
            device=device,
            dtype=torch.float64,
            graph_mode=args.mace_graph_mode,
            skin=skin,
        )
        calculator(calculator.create_state([systems[0]]), compute_stress=True)
        model = None
        model_metadata = {"name": "MACE-OFF-Small"}
    synchronize(device)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with RuntimeProfiler(device=device) as profiler:
        if args.model == "atombit":
            if model is None:
                raise RuntimeError("AtomBit model was not initialized")
            output = execute_atombit(
                systems,
                buckets,
                device=device,
                model=model,
                linear_algebra_backend=args.linear_algebra_backend,
                skin=skin,
                fmax=args.fmax,
                max_steps=args.max_steps,
            )
        else:
            if calculator is None:
                raise RuntimeError("MACE calculator was not initialized")
            output = execute_mace(
                systems,
                buckets,
                calculator=calculator,
                linear_algebra_backend=args.linear_algebra_backend,
                fmax=args.fmax,
                max_steps=args.max_steps,
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )
    result.update(
        {
            "status": "complete",
            "timing_seconds": elapsed,
            "systems_per_second": len(systems) / elapsed,
            "peak_memory_bytes": peak,
            "runtime_profile": profiler.summary(),
            "model_metadata": model_metadata,
            "execution_buckets": [
                {"systems": len(indices), "resident_capacity": capacity}
                for indices, capacity in buckets
            ],
            **output,
        }
    )
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
