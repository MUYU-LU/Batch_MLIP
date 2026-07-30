#!/usr/bin/env python3
"""Validate persistent OMC-CSP scheduling across signed workload manifests."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_omc_csp_scheduler_auto import (  # noqa: E402
    _devices,
    _loader_process_count,
    _records,
)
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
    ReproducibilityConfig,
    configure_reproducibility,
)
from batch_mlip.planning import read_planning_profile  # noqa: E402
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _peak_worker_memory(scheduling: dict[str, Any]) -> dict[str, Any]:
    by_device: dict[str, dict[str, int]] = {}
    for worker in scheduling.get("workers", []):
        device = str(worker["device"])
        allocated = max(
            (
                int(chunk.get("peak_allocated_bytes") or 0)
                for chunk in worker.get("chunks", [])
            ),
            default=0,
        )
        reserved = max(
            (
                int(chunk.get("peak_reserved_bytes") or 0)
                for chunk in worker.get("chunks", [])
            ),
            default=0,
        )
        current = by_device.setdefault(
            device,
            {"allocated_bytes": 0, "reserved_bytes": 0},
        )
        current["allocated_bytes"] = max(current["allocated_bytes"], allocated)
        current["reserved_bytes"] = max(current["reserved_bytes"], reserved)
    maximum_reserved = max(
        (value["reserved_bytes"] for value in by_device.values()),
        default=0,
    )
    return {
        "by_device": by_device,
        "maximum_allocated_bytes": max(
            (value["allocated_bytes"] for value in by_device.values()),
            default=0,
        ),
        "maximum_reserved_bytes": maximum_reserved,
        "maximum_reserved_fraction": (
            maximum_reserved / torch.cuda.get_device_properties(0).total_memory
            if maximum_reserved
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--planning-profile-dir", type=Path, required=True)
    parser.add_argument("--workload-id", action="append", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--manifest-prefetch-depth", type=int, default=1)
    parser.add_argument(
        "--multi-gpu-dispatch-policy",
        choices=("subdivide", "preserve_resident"),
        default="subdivide",
    )
    parser.add_argument(
        "--manifest-loader-processes",
        type=_loader_process_count,
        default="auto",
    )
    args = parser.parse_args()
    if (
        args.max_steps <= 0
        or args.fmax <= 0.0
        or args.manifest_prefetch_depth < 0
    ):
        parser.error(
            "--max-steps and --fmax must be positive; "
            "--manifest-prefetch-depth must be non-negative"
        )
    if len(set(args.workload_id)) != len(args.workload_id):
        parser.error("--workload-id values must be unique")

    devices = _devices(args.devices)
    reproducibility = configure_reproducibility(
        ReproducibilityConfig(seed=args.seed, cpu_threads=1, interop_threads=1),
        require_preconfigured_python_hash=True,
    )
    manifests = []
    profiles = []
    for workload_id in args.workload_id:
        manifest_path = args.manifest_dir / f"{workload_id}.json"
        profile_path = args.planning_profile_dir / f"{workload_id}.json"
        manifests.append(read_workload_manifest(manifest_path))
        profiles.append(read_planning_profile(profile_path))

    started = time.perf_counter()
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
    first_job = manifests[0].jobs[0]
    warmup_system = read(
        args.dataset_dir / first_job.source_path,
        index=first_job.frame_index,
    )
    calculator(calculator.create_state([warmup_system]), compute_stress=True)
    synchronize(devices[0])
    del warmup_system

    config = AutoSchedulerConfig(
        manifest_loader_processes=args.manifest_loader_processes,
        manifest_prefetch_chunks_per_worker=args.manifest_prefetch_depth,
        multi_gpu_dispatch_policy=args.multi_gpu_dispatch_policy,
    )
    relaxation_options = {
        "cell_filter": FrechetCellFilter(),
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "max_step": 0.2,
        "alpha": 70.0,
        "optimizer_dtype": "float64",
        "linear_algebra_backend": "auto",
    }
    calls = []
    sequence_started = time.perf_counter()
    with BatchExecutor(
        calculator,
        devices=devices,
        auto_config=config,
    ) as executor:
        for call_index, (manifest, profile) in enumerate(
            zip(manifests, profiles, strict=True),
        ):
            source_ids = [job.system_id for job in manifest.jobs]
            call_started = time.perf_counter()
            result = executor.relax_manifest(
                manifest,
                args.dataset_dir,
                profile,
                optimizer="bfgs",
                **relaxation_options,
            )
            for device in devices:
                synchronize(device)
            call_seconds = time.perf_counter() - call_started
            records = _records(result, source_ids)
            returned_sources = [record["source"] for record in records]
            if returned_sources != source_ids:
                raise RuntimeError(
                    f"{manifest.workload_id} did not retain manifest order"
                )
            scheduling = result.metadata["scheduling"]
            active_device_count = int(scheduling["active_gpu_count"])
            calls.append(
                {
                    "call": call_index + 1,
                    "workload_id": manifest.workload_id,
                    "workload_manifest_sha256": manifest.manifest_sha256,
                    "planning_profile_sha256": profile.profile_sha256,
                    "pool_size": len(source_ids),
                    "seconds": call_seconds,
                    "active_device_wall_seconds": (
                        call_seconds * active_device_count
                    ),
                    "converged_count": int(result.converged.sum().item()),
                    "model_evaluations": result.model_evaluations,
                    "graph_evaluations": result.graph_evaluations,
                    "optimizer_steps": int(result.steps),
                    "source_order_equal": returned_sources == source_ids,
                    "peak_memory": _peak_worker_memory(scheduling),
                    "scheduling": scheduling,
                    "records": records,
                }
            )
    sequence_seconds = time.perf_counter() - sequence_started
    shutdown = executor.shutdown_metadata

    worker_pid_sequences = [
        tuple(call["scheduling"]["worker_pids"]) for call in calls
    ]
    worker_generations = [
        int(call["scheduling"]["worker_generation"]) for call in calls
    ]
    materializer_generations = [
        int(
            call["scheduling"]["structure_materialization"][
                "materializer_generation"
            ]
        )
        for call in calls
    ]
    loader_processes = [
        int(
            call["scheduling"]["structure_materialization"][
                "total_loader_processes"
            ]
        )
        for call in calls
    ]
    output = {
        "schema_version": 1,
        "status": "complete",
        "method": "persistent_source_backed_auto_sequence",
        "workload_ids": list(args.workload_id),
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
            "manifest_loader_processes": args.manifest_loader_processes,
            "manifest_prefetch_chunks_per_worker": (
                args.manifest_prefetch_depth
            ),
            "multi_gpu_dispatch_policy": args.multi_gpu_dispatch_policy,
        },
        "reproducibility": reproducibility,
        "environment": environment_metadata(devices[0]),
        "timing": {
            "script_seconds": time.perf_counter() - started,
            "sequence_seconds": sequence_seconds,
            "call_seconds": [call["seconds"] for call in calls],
            "active_device_wall_seconds": sum(
                call["active_device_wall_seconds"] for call in calls
            ),
        },
        "persistence": {
            "worker_pid_sequences": worker_pid_sequences,
            "worker_pids_constant": (
                all(
                    sequence == worker_pid_sequences[0]
                    for sequence in worker_pid_sequences
                )
            ),
            "worker_generations": worker_generations,
            "worker_generation_constant": len(set(worker_generations)) == 1,
            "materializer_generations": materializer_generations,
            "loader_processes": loader_processes,
        },
        "calls": calls,
        "executor_shutdown": shutdown,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sequence_seconds": sequence_seconds,
                "workload_count": len(calls),
                "job_count": sum(call["pool_size"] for call in calls),
                "converged_count": sum(
                    call["converged_count"] for call in calls
                ),
                "worker_pids_constant": output["persistence"][
                    "worker_pids_constant"
                ],
                "worker_generation_constant": output["persistence"][
                    "worker_generation_constant"
                ],
                "loader_processes": loader_processes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
