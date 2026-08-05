#!/usr/bin/env python3
"""Gate AtomBit OMC-CSP through the one-call production interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    ReproducibilityConfig,
    configure_reproducibility,
    materialize_workload,
    model_state_sha256,
    optimize_pool,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402

_H100_MEMORY_BYTES = 85_017_886_720
_PLANNING_MEMORY_FRACTION = 0.85
_RUNTIME_MEMORY_FRACTION = 0.91
_MINIMUM_CONVERGENCE_RATE = 0.99
_POLICY_ID = "omc-csp-atombit-h100-capacity-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structures-output", type=Path, required=True)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()
    if args.fmax <= 0.0 or args.max_steps <= 0:
        parser.error("fmax and max-steps must be positive")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _array_bytes(values: Any, dtype: np.dtype[Any]) -> bytes:
    array = np.asarray(values, dtype=dtype)
    return array.astype(dtype.newbyteorder("<"), copy=False).tobytes()


def _endpoint_sha256(
    structures: list[Any],
    system_ids: list[str],
    energies: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for atoms, system_id, energy in zip(
        structures,
        system_ids,
        energies,
        strict=True,
    ):
        digest.update(system_id.encode("utf-8"))
        digest.update(_array_bytes(atoms.numbers, np.dtype(np.int64)))
        digest.update(_array_bytes(atoms.positions, np.dtype(np.float64)))
        digest.update(_array_bytes(atoms.cell.array, np.dtype(np.float64)))
        digest.update(_array_bytes(atoms.pbc, np.dtype(np.uint8)))
        digest.update(_array_bytes([energy], np.dtype(np.float64)))
    return digest.hexdigest()


def _worker_summary(
    scheduling: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, list[int]]:
    workers = []
    peak_allocated = 0
    peak_reserved = 0
    predictions = []
    for worker in scheduling.get("workers", []):
        chunks = []
        for chunk in worker.get("chunks", []):
            allocated = int(chunk.get("peak_allocated_bytes") or 0)
            reserved = int(chunk.get("peak_reserved_bytes") or 0)
            predicted = chunk.get("predicted_peak_bytes")
            if predicted is not None:
                predictions.append(int(predicted))
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
            chunks.append(
                {
                    "task_index": int(chunk["task_index"]),
                    "system_count": int(chunk["system_count"]),
                    "resident_capacity": int(chunk["resident_capacity"]),
                    "wall_seconds": float(chunk["wall_seconds"]),
                    "predicted_peak_bytes": (
                        None if predicted is None else int(predicted)
                    ),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                }
            )
        workers.append(
            {
                "worker_id": int(worker["worker_id"]),
                "device": str(worker["device"]),
                "startup_seconds": float(worker["startup_seconds"]),
                "task_seconds": float(worker["task_seconds"]),
                "chunks": chunks,
            }
        )
    return workers, peak_allocated, peak_reserved, predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "git_head": _git_head(),
        "contract": {
            "application": "closed-pool OMC-CSP variable-cell relaxation",
            "model": "AtomBit-smooth-rms-fp32",
            "model_dtype": "torch.float32",
            "optimizer": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "planning_memory_fraction": _PLANNING_MEMORY_FRACTION,
            "runtime_peak_reserved_fraction": _RUNTIME_MEMORY_FRACTION,
        },
    }
    try:
        reproducibility = configure_reproducibility(
            ReproducibilityConfig(),
            require_preconfigured_python_hash=True,
        )
        devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
        if not devices:
            raise ValueError("at least one device is required")

        manifest = read_workload_manifest(args.manifest)
        systems = materialize_workload(manifest, args.dataset_dir)
        expected_ids = [job.system_id for job in manifest.jobs]
        calculator = AtomBitBatchCalculator.from_checkpoint(
            args.checkpoint,
            e0=args.e0,
            device=devices[0],
            dtype=torch.float32,
            force_mode="autograd",
            cutoff=6.0,
            skin=0.5,
            neighbor_backend="auto",
        )

        optimization_started = time.perf_counter()
        result = optimize_pool(
            systems,
            calculator,
            devices=devices,
            optimizer="bfgs",
            cell_filter="frechet",
            policy="auto",
            auto_config=AutoSchedulerConfig(
                cache_enabled=False,
                max_batch_size=256,
                memory_safety_fraction=_PLANNING_MEMORY_FRACTION,
                memory_growth_margin=1.10,
                multi_gpu_target_chunks_per_device=2,
                multi_gpu_queue_policy="bucket_stratified",
            ),
            fmax=args.fmax,
            max_steps=args.max_steps,
            max_step=0.2,
            alpha=70.0,
            linear_algebra_backend="auto",
        )
        optimization_seconds = time.perf_counter() - optimization_started

        structures = result.structures
        observed_ids = [atoms.info.get("workload_system_id") for atoms in structures]
        energies = result.evaluation.energy.detach().cpu().to(torch.float64).numpy()
        converged = result.converged.detach().cpu().numpy().astype(bool)
        max_force = result.max_force.detach().cpu().to(torch.float64).numpy()
        if result.max_stress is None:
            raise RuntimeError("variable-cell result does not contain maximum stress")
        max_stress = result.max_stress.detach().cpu().to(torch.float64).numpy()
        endpoint_sha256 = _endpoint_sha256(structures, expected_ids, energies)

        args.structures_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_structures = args.structures_output.with_name(
            args.structures_output.name + ".tmp.extxyz.gz"
        )
        write(temporary_structures, structures, format="extxyz")
        temporary_structures.replace(args.structures_output)

        scheduling = result.metadata["scheduling"]
        workers, peak_allocated, peak_reserved, predictions = _worker_summary(scheduling)
        capacity = scheduling["capacity_planning"]
        probe = scheduling["probe"]
        shutdown = result.metadata["optimize_pool"]["executor_shutdown"]
        active_gpu_count = int(scheduling["active_gpu_count"])
        convergence_rate = float(np.mean(converged))
        unique_sources = len({job.source_sha256 for job in manifest.jobs})
        unique_structures = len(
            {job.normalized_structure_sha256 for job in manifest.jobs}
        )
        finite_endpoints = bool(
            np.isfinite(energies).all()
            and np.isfinite(max_force).all()
            and np.isfinite(max_stress).all()
            and all(np.isfinite(atoms.positions).all() for atoms in structures)
            and all(np.isfinite(atoms.cell.array).all() for atoms in structures)
        )
        gates = {
            "signed_manifest_verified": (
                manifest.manifest_sha256 == manifest.calculate_sha256()
            ),
            "unique_unreplicated_pool": (
                unique_sources == len(manifest.jobs)
                and unique_structures == len(manifest.jobs)
            ),
            "exact_job_coverage": len(structures) == len(manifest.jobs),
            "input_order_preserved": observed_ids == expected_ids,
            "finite_endpoints": finite_endpoints,
            "convergence_rate_at_least_0_99": (
                convergence_rate >= _MINIMUM_CONVERGENCE_RATE
            ),
            "exact_packaged_capacity_policy": (
                capacity["mode"] == "offline_hardware_model"
                and capacity["policy_id"] == _POLICY_ID
            ),
            "zero_memory_probe_forwards": (
                int(probe["system_count"]) == 0
                and int(probe["model_forward_count"]) == 0
            ),
            "predicted_chunks_within_85_percent": (
                bool(predictions)
                and max(predictions)
                <= int(_PLANNING_MEMORY_FRACTION * _H100_MEMORY_BYTES)
            ),
            "runtime_peak_reserved_within_91_percent": (
                peak_reserved <= int(_RUNTIME_MEMORY_FRACTION * _H100_MEMORY_BYTES)
            ),
            "all_active_workers_acknowledged_shutdown": (
                sorted(int(value) for value in shutdown["acknowledged_worker_ids"])
                == list(range(active_gpu_count))
            ),
        }
        accepted = all(gates.values())
        payload.update(
            {
                "status": "pass" if accepted else "fail",
                "decision": (
                    "accept_atombit_omc_csp_public_api_v1"
                    if accepted
                    else "reject_atombit_omc_csp_public_api_v1"
                ),
                "workload": {
                    "workload_id": manifest.workload_id,
                    "manifest_sha256": manifest.manifest_sha256,
                    "pool_size": len(manifest.jobs),
                    "unique_sources": unique_sources,
                    "unique_structures": unique_structures,
                    "scheduler_split": manifest.metadata.get("scheduler_split"),
                    "source_families": manifest.metadata.get("source_families"),
                    "atom_count_min": min(job.atom_count for job in manifest.jobs),
                    "atom_count_max": max(job.atom_count for job in manifest.jobs),
                },
                "checkpoint": {
                    "path": str(args.checkpoint),
                    "file_sha256": _sha256_file(args.checkpoint),
                    "model_state_sha256": model_state_sha256(calculator.model),
                    "e0_path": str(args.e0),
                    "e0_sha256": _sha256_file(args.e0),
                },
                "devices": list(devices),
                "reproducibility": reproducibility,
                "timing": {
                    "optimization_and_shutdown_seconds": optimization_seconds,
                    "full_script_seconds": time.perf_counter() - started,
                    "production_run_seconds": scheduling["production_run_seconds"],
                    "profiling_seconds": scheduling["profiling_seconds"],
                    "planning_seconds": scheduling["planning_seconds"],
                    "reassembly_seconds": scheduling["reassembly_seconds"],
                    "worker_startup_seconds": scheduling[
                        "worker_startup_seconds_this_call"
                    ],
                },
                "execution": {
                    "converged_count": int(converged.sum()),
                    "convergence_rate": convergence_rate,
                    "model_evaluations": int(result.model_evaluations),
                    "graph_evaluations": int(result.graph_evaluations),
                    "maximum_final_force_eV_per_A": float(max_force.max()),
                    "maximum_final_stress_eV_per_A3": float(max_stress.max()),
                    "endpoint_sha256": endpoint_sha256,
                    "relaxed_structures_path": str(args.structures_output),
                    "relaxed_structures_sha256": _sha256_file(args.structures_output),
                },
                "scheduling": {
                    "summary": result.schedule,
                    "capacity_planning": capacity,
                    "capacity_policy_resolution": scheduling[
                        "capacity_policy_resolution"
                    ],
                    "allocator": scheduling["allocator"],
                    "probe": probe,
                    "execution_chunk_count": scheduling["execution_chunk_count"],
                    "active_gpu_count": active_gpu_count,
                    "workers": workers,
                },
                "memory": {
                    "device_memory_bytes": _H100_MEMORY_BYTES,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_fraction": peak_reserved / _H100_MEMORY_BYTES,
                    "maximum_predicted_peak_bytes": max(predictions),
                },
                "shutdown": shutdown,
                "gates": gates,
            }
        )
        _write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "converged_count": payload["execution"]["converged_count"],
                    "pool_size": len(manifest.jobs),
                    "peak_reserved_fraction": payload["memory"][
                        "peak_reserved_fraction"
                    ],
                    "optimization_seconds": optimization_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not accepted:
            raise SystemExit("acceptance gates failed")
    except BaseException as error:
        if payload.get("status") == "running":
            payload.update(
                {
                    "status": "error",
                    "decision": "reject_atombit_omc_csp_public_api_v1",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    "full_script_seconds": time.perf_counter() - started,
                }
            )
            _write_json(args.output, payload)
        raise


if __name__ == "__main__":
    main()
