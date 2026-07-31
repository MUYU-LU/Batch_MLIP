#!/usr/bin/env python3
"""Run the frozen OMC-CSP contract with the current automatic scheduler."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    synchronize,
)
from benchmark_variable_cell_scaling import serialize_record  # noqa: E402
from omc_csp_tail_recovery import (  # noqa: E402
    AseBFGSRecoveryTask,
    nonconverged_sources,
    recovery_task_cost,
    replace_nonconverged_records,
    run_ase_bfgs_tail_recovery,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    BatchExecutor,
    FrechetCellFilter,
    ReproducibilityConfig,
    RuntimeProfiler,
    configure_reproducibility,
    relax,
    relax_manifest,
)
from batch_mlip.planning import read_planning_profile  # noqa: E402
from batch_mlip.workloads import read_workload_manifest  # noqa: E402

LINEAR_ALGEBRA_BACKENDS = ("auto", "cholesky", "grouped", "serial")
TAIL_RECOVERY_MODES = ("none", "ase_bfgs")
MATERIALIZATION_MODES = ("eager", "manifest_lazy")


def _loader_process_count(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "loader process count must be a positive integer or 'auto'"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "loader process count must be a positive integer or 'auto'"
        )
    return parsed


def _devices(value: str) -> list[torch.device]:
    devices = [torch.device(f"cuda:{item.strip()}") for item in value.split(",")]
    if not devices or any(device.index is None for device in devices):
        raise ValueError("--devices must contain one or more CUDA indices")
    if len({device.index for device in devices}) != len(devices):
        raise ValueError("--devices must not repeat a CUDA index")
    return devices


def _records(result: Any, source_ids: list[str]) -> list[dict[str, Any]]:
    energies = result.evaluation.energy.detach().cpu().numpy()
    forces = result.evaluation.forces.detach().cpu().numpy()
    stresses = result.evaluation.stress.detach().cpu().numpy()
    positions = result.state.positions.detach().cpu().numpy()
    cells = result.state.cells.detach().cpu().numpy()
    converged = result.converged.detach().cpu().numpy()
    converged_steps = result.converged_step.detach().cpu().numpy()
    records = []
    for index, source in enumerate(source_ids):
        atom_slice = result.state.atom_slice(index)
        step = int(converged_steps[index])
        records.append(
            serialize_record(
                source=source,
                converged=bool(converged[index]),
                steps=step if step >= 0 else int(result.steps),
                energy=float(energies[index]),
                forces=np.asarray(forces[atom_slice]),
                stress=np.asarray(stresses[index]),
                positions=np.asarray(positions[atom_slice]),
                cell=np.asarray(cells[index]),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--planning-profile", type=Path)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--executor-calls", type=int, default=0)
    parser.add_argument("--target-chunks-per-device", type=int, default=2)
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
    parser.add_argument("--manifest-prefetch-depth", type=int, default=1)
    parser.add_argument(
        "--manifest-loader-processes",
        type=_loader_process_count,
        default="auto",
    )
    parser.add_argument(
        "--linear-algebra-backend",
        choices=LINEAR_ALGEBRA_BACKENDS,
        default="auto",
    )
    parser.add_argument(
        "--tail-recovery",
        choices=TAIL_RECOVERY_MODES,
        default="none",
    )
    parser.add_argument(
        "--materialization",
        choices=MATERIALIZATION_MODES,
        default="eager",
    )
    args = parser.parse_args()
    if (
        args.max_steps <= 0
        or args.fmax <= 0.0
        or args.executor_calls < 0
        or args.target_chunks_per_device <= 0
        or args.manifest_prefetch_depth < 0
    ):
        parser.error(
            "--max-steps and --fmax must be positive; "
            "--target-chunks-per-device must be positive; --executor-calls "
            "and --manifest-prefetch-depth must be non-negative"
        )
    if args.materialization == "manifest_lazy" and args.planning_profile is None:
        parser.error("--planning-profile is required with manifest_lazy")
    if args.executor_calls and args.materialization != "manifest_lazy":
        parser.error("--executor-calls requires manifest_lazy")

    devices = _devices(args.devices)
    reproducibility = configure_reproducibility(
        ReproducibilityConfig(seed=args.seed, cpu_threads=1, interop_threads=1),
        require_preconfigured_python_hash=True,
    )
    started = time.perf_counter()
    manifest = read_workload_manifest(args.manifest)
    source_ids = [job.system_id for job in manifest.jobs]
    systems = None
    planning_profile = None
    if args.materialization == "eager":
        systems = []
        for job in manifest.jobs:
            atoms = read(
                args.dataset_dir / job.source_path,
                index=job.frame_index,
            )
            atoms.info["benchmark_source"] = job.system_id
            systems.append(atoms)
    else:
        planning_profile = read_planning_profile(args.planning_profile)

    model, model_metadata = load_production_model(args.checkpoint)
    calculator = AtomBitBatchCalculator(
        model.to(device=devices[0], dtype=torch.float32).eval(),
        cutoff=6.0,
        skin=0.5,
        device=devices[0],
        dtype=torch.float32,
        force_mode="autograd",
        neighbor_backend="auto",
    )
    warmup_system = (
        systems[0]
        if systems is not None
        else read(
            args.dataset_dir / manifest.jobs[0].source_path,
            index=manifest.jobs[0].frame_index,
        )
    )
    calculator(calculator.create_state([warmup_system]), compute_stress=True)
    synchronize(devices[0])
    del warmup_system
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    execution_started = time.perf_counter()
    executor_call_timings = []
    executor_shutdown = None
    with RuntimeProfiler(device=devices[0]) as profiler:
        relaxation_options = {
            "cell_filter": FrechetCellFilter(),
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "max_step": 0.2,
            "alpha": 70.0,
            "optimizer_dtype": "float64",
            "linear_algebra_backend": args.linear_algebra_backend,
        }
        if systems is None:
            if planning_profile is None:  # pragma: no cover - CLI narrows this
                raise RuntimeError("manifest_lazy requires a planning profile")
            auto_config = AutoSchedulerConfig(
                manifest_loader_processes=(
                    args.manifest_loader_processes
                ),
                manifest_prefetch_chunks_per_worker=(
                    args.manifest_prefetch_depth
                ),
                multi_gpu_target_chunks_per_device=(
                    args.target_chunks_per_device
                ),
                multi_gpu_dispatch_policy=args.multi_gpu_dispatch_policy,
                multi_gpu_queue_policy=args.multi_gpu_queue_policy,
            )
            if args.executor_calls:
                with BatchExecutor(
                    calculator,
                    devices=devices,
                    auto_config=auto_config,
                ) as executor:
                    for call_index in range(args.executor_calls):
                        call_started = time.perf_counter()
                        result = executor.relax_manifest(
                            manifest,
                            args.dataset_dir,
                            planning_profile,
                            optimizer="bfgs",
                            **relaxation_options,
                        )
                        for device in devices:
                            synchronize(device)
                        call_schedule = result.metadata["scheduling"]
                        executor_call_timings.append(
                            {
                                "call": call_index + 1,
                                "seconds": (
                                    time.perf_counter() - call_started
                                ),
                                "worker_generation": call_schedule[
                                    "worker_generation"
                                ],
                                "worker_pids": call_schedule["worker_pids"],
                                "worker_startup_seconds": call_schedule[
                                    "worker_startup_seconds_this_call"
                                ],
                                "production_run_seconds": call_schedule[
                                    "production_run_seconds"
                                ],
                                "materialization": call_schedule[
                                    "structure_materialization"
                                ],
                            }
                        )
                executor_shutdown = executor.shutdown_metadata
            else:
                result = relax_manifest(
                    manifest,
                    args.dataset_dir,
                    planning_profile,
                    calculator,
                    optimizer="bfgs",
                    devices=devices,
                    auto_config=auto_config,
                    **relaxation_options,
                )
        else:
            result = relax(
                systems,
                calculator,
                optimizer="bfgs",
                scheduling="auto",
                devices=devices,
                **relaxation_options,
            )
    for device in devices:
        synchronize(device)
    tensor_execution_seconds = time.perf_counter() - execution_started
    tensor_records = _records(result, source_ids)
    records = tensor_records
    scheduling = result.metadata.get("scheduling", {})
    tensor_model_evaluations = result.model_evaluations
    tensor_graph_evaluations = result.graph_evaluations
    tensor_optimizer_steps = int(result.steps)
    tensor_converged_count = int(result.converged.sum().item())
    tail_recovery: dict[str, Any] = {
        "enabled": False,
        "mode": "none",
        "attempted_count": 0,
        "converged_count": 0,
        "total_seconds": 0.0,
        "model_evaluations": 0,
        "graph_evaluations": 0,
        "optimizer_steps": 0,
        "workers": [],
        "tasks": [],
        "peak_allocated_bytes_by_device": {},
        "peak_reserved_bytes_by_device": {},
        "parent_reserved_bytes_during_recovery_by_device": {},
    }
    if args.tail_recovery == "ase_bfgs":
        recovery_sources = set(nonconverged_sources(tensor_records))
        jobs_by_source = {job.system_id: job for job in manifest.jobs}
        recovery_tasks = []
        for source in source_ids:
            if source not in recovery_sources:
                continue
            job = jobs_by_source[source]
            recovery_tasks.append(
                AseBFGSRecoveryTask(
                    source=source,
                    source_path=job.source_path,
                    frame_index=job.frame_index,
                    estimated_cost=recovery_task_cost(
                        atom_count=job.atom_count,
                        candidate_edges=max(
                            job.topology_edge_counts.values(),
                            default=0,
                        ),
                    ),
                )
            )
        recovery_started = time.perf_counter()
        if recovery_tasks:
            model.to(device="cpu")
            del calculator
            for device in devices:
                synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            parent_reserved = {
                str(device): torch.cuda.memory_reserved(device)
                for device in devices
            }
        else:
            parent_reserved = {}
        recovery_records, tail_recovery = run_ase_bfgs_tail_recovery(
            recovery_tasks,
            checkpoint=args.checkpoint,
            dataset_dir=args.dataset_dir,
            devices=devices,
            cutoff=6.0,
            fmax=args.fmax,
            max_steps=args.max_steps,
            max_step=0.2,
            alpha=70.0,
        )
        tail_recovery["mode"] = "ase_bfgs"
        tail_recovery[
            "parent_reserved_bytes_during_recovery_by_device"
        ] = parent_reserved
        tail_recovery["total_seconds"] = (
            time.perf_counter() - recovery_started
        )
        records = replace_nonconverged_records(
            tensor_records,
            recovery_records,
        )
    execution_seconds = (
        tensor_execution_seconds + float(tail_recovery["total_seconds"])
    )
    if {record["source"] for record in records} != set(source_ids):
        raise RuntimeError("automatic scheduler did not return exact job coverage")

    workers = scheduling.get("workers", [])
    if args.executor_calls:
        method = "persistent_source_backed_auto"
        peak_memory = {
            worker["device"]: {
                "allocated_bytes": max(
                    (
                        int(chunk["peak_allocated_bytes"] or 0)
                        for chunk in worker["chunks"]
                    ),
                    default=0,
                ),
                "reserved_bytes": max(
                    (
                        int(chunk["peak_reserved_bytes"] or 0)
                        for chunk in worker["chunks"]
                    ),
                    default=0,
                ),
            }
            for worker in workers
        }
    else:
        method = (
            (
                "source_backed_auto_ase_tail_recovery"
                if args.tail_recovery == "ase_bfgs"
                else "source_backed_auto"
            )
            if args.materialization == "manifest_lazy"
            else (
                "refined_auto_ase_tail_recovery"
                if args.tail_recovery == "ase_bfgs"
                else "current_auto"
            )
        )
        peak_memory = {
            str(device): {
                "allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            for device in devices
        }
    output = {
        "schema_version": 1,
        "status": "complete",
        "method": method,
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "pool_size": len(source_ids),
        "devices": [str(device) for device in devices],
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            **model_metadata,
        },
        "contract": {
            "optimizer": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "cutoff_A": 6.0,
            "skin_A": 0.5,
            "force_mode": "autograd",
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "scheduling": "auto",
            "linear_algebra_backend": args.linear_algebra_backend,
            "tail_recovery": args.tail_recovery,
            "tail_recovery_optimizer": (
                "ASE BFGS"
                if args.tail_recovery == "ase_bfgs"
                else None
            ),
            "structure_materialization": args.materialization,
            "manifest_loader_processes": (
                args.manifest_loader_processes
            ),
            "persistent_executor_calls": args.executor_calls,
            "manifest_prefetch_chunks_per_worker": (
                args.manifest_prefetch_depth
            ),
            "target_chunks_per_device": (
                args.target_chunks_per_device
            ),
            "multi_gpu_dispatch_policy": (
                args.multi_gpu_dispatch_policy
            ),
            "multi_gpu_queue_policy": args.multi_gpu_queue_policy,
            "planning_profile_sha256": (
                None
                if planning_profile is None
                else planning_profile.profile_sha256
            ),
            "benchmark_parent_prewarm_system_count": 1,
        },
        "reproducibility": reproducibility,
        "environment": environment_metadata(devices[0]),
        "timing": {
            "script_seconds": time.perf_counter() - started,
            "execution_seconds": execution_seconds,
            "tensor_execution_seconds": tensor_execution_seconds,
            "tail_recovery_seconds": tail_recovery["total_seconds"],
            "scheduler_total_seconds": scheduling.get("total_seconds"),
            "profiling_seconds": scheduling.get("profiling_seconds"),
            "worker_startup_seconds": scheduling.get("worker_startup_wall_seconds"),
            "worker_execution_seconds": scheduling.get("worker_run_wall_seconds"),
            "executor_calls": executor_call_timings,
        },
        "peak_memory": peak_memory,
        "scheduling": scheduling,
        "executor_shutdown": executor_shutdown,
        "tail_recovery": tail_recovery,
        "runtime_profile": profiler.summary(),
        "model_evaluations": (
            tensor_model_evaluations
            + int(tail_recovery["model_evaluations"])
        ),
        "graph_evaluations": (
            tensor_graph_evaluations
            + int(tail_recovery["graph_evaluations"])
        ),
        "optimizer_steps": tensor_optimizer_steps,
        "tensor_converged_count": tensor_converged_count,
        "converged_count": sum(
            int(record["converged"]) for record in records
        ),
        "records": records,
        "workers": workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pool_size": len(source_ids),
                "execution_seconds": execution_seconds,
                "converged_count": output["converged_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
