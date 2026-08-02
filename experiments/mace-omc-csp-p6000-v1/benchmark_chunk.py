#!/usr/bin/env python3
"""Replay one exact MACE production chunk under an explicit runtime policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    FrechetCellFilter,
    MACEBatchCalculator,
    ReproducibilityConfig,
    RuntimeProfiler,
    configure_reproducibility,
    relax,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_array_sha256(values: torch.Tensor) -> str:
    array = np.ascontiguousarray(values.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def _select_chunk(
    production: dict[str, Any],
    *,
    worker_id: int,
    bucket_index: int,
    system_count: int,
) -> dict[str, Any]:
    matches = [
        chunk
        for worker in production["workers"]
        if int(worker["worker_id"]) == worker_id
        for chunk in worker["chunks"]
        if int(chunk["bucket_index"]) == bucket_index
        and int(chunk["system_count"]) == system_count
    ]
    if len(matches) != 1:
        raise ValueError(
            "exactly one production chunk must match worker, bucket, and size"
        )
    return matches[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mace-model", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--bucket-index", type=int, required=True)
    parser.add_argument("--system-count", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--graph-mode",
        choices=("cached", "rebuild"),
        default="rebuild",
    )
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--store-state",
        action="store_true",
        help="Include full endpoint tensors for bounded correctness controls.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.system_count <= 0 or args.max_steps <= 0 or args.fmax <= 0.0:
        parser.error("system count, maximum steps, and fmax must be positive")
    if args.skin < 0.0:
        parser.error("skin must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    configure_reproducibility(
        ReproducibilityConfig(
            seed=args.seed,
            cpu_threads=1,
            interop_threads=1,
        ),
        require_preconfigured_python_hash=True,
    )
    production = json.loads(
        args.production_result.read_text(encoding="utf-8")
    )
    chunk = _select_chunk(
        production,
        worker_id=args.worker_id,
        bucket_index=args.bucket_index,
        system_count=args.system_count,
    )
    manifest = read_workload_manifest(args.manifest)
    indices = tuple(int(value) for value in chunk["system_indices"])
    systems = []
    sources = []
    for index in indices:
        job = manifest.jobs[index]
        atoms = read(
            args.dataset_dir / job.source_path,
            index=job.frame_index,
        )
        atoms.info["benchmark_source"] = job.system_id
        systems.append(atoms)
        sources.append(job.system_id)

    device = torch.device(args.device)
    calculator = MACEBatchCalculator.from_off(
        model=args.mace_model,
        device=device,
        dtype=torch.float64,
        graph_mode=args.graph_mode,
        skin=args.skin,
        neighbor_backend="auto",
    )
    warm_state = calculator.create_state([systems[0]])
    calculator(warm_state, compute_stress=True)
    torch.cuda.synchronize(device)
    del warm_state
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    with RuntimeProfiler(device=device) as profiler:
        result = relax(
            systems,
            calculator,
            optimizer="bfgs",
            scheduling="single_batch",
            cell_filter=FrechetCellFilter(),
            active_compaction=True,
            fmax=args.fmax,
            max_steps=args.max_steps,
            max_step=0.2,
            alpha=70.0,
            optimizer_dtype="float64",
            linear_algebra_backend="auto",
        )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "status": "complete",
        "source_production_result": {
            "path": str(args.production_result.resolve()),
            "sha256": _sha256(args.production_result),
        },
        "source_manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest.manifest_sha256,
        },
        "source_chunk": {
            "worker_id": args.worker_id,
            "bucket_index": args.bucket_index,
            "system_count": args.system_count,
            "system_indices": list(indices),
            "source_ids_sha256": hashlib.sha256(
                "\n".join(sources).encode("utf-8")
            ).hexdigest(),
            "production_wall_seconds": chunk["wall_seconds"],
            "production_peak_allocated_bytes": chunk[
                "peak_allocated_bytes"
            ],
            "production_peak_reserved_bytes": chunk["peak_reserved_bytes"],
        },
        "contract": {
            "model": str(args.mace_model.resolve()),
            "model_file_sha256": _sha256(args.mace_model),
            "model_dtype": "torch.float64",
            "graph_mode": args.graph_mode,
            "skin_A": args.skin,
            "optimizer": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "max_step_A": 0.2,
            "alpha_eV_per_A2": 70.0,
        },
        "allocator": {
            "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "pytorch_cuda_alloc_conf": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
            "backend": torch.cuda.memory.get_allocator_backend(),
        },
        "result": {
            "seconds": seconds,
            "systems_per_second": len(systems) / seconds,
            "converged_count": int(result.converged.sum().item()),
            "model_evaluations": result.model_evaluations,
            "graph_evaluations": result.graph_evaluations,
            "steps": result.steps,
            "step_values": result.converged_step.cpu().tolist(),
            "energy_eV": result.evaluation.energy.cpu().tolist(),
            "max_force_eV_per_A": result.max_force.cpu().tolist(),
            "positions_sha256": _canonical_array_sha256(
                result.state.positions
            ),
            "cells_sha256": _canonical_array_sha256(result.state.cells),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "runtime_profile": profiler.summary(),
        },
    }
    if args.store_state:
        payload["result"]["positions_A"] = result.state.positions.cpu().tolist()
        payload["result"]["cells_A"] = result.state.cells.cpu().tolist()
        payload["result"]["forces_eV_per_A"] = (
            result.evaluation.forces.cpu().tolist()
        )
        payload["result"]["stress_eV_per_A3"] = (
            None
            if result.evaluation.stress is None
            else result.evaluation.stress.cpu().tolist()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seconds": seconds,
                "converged_count": payload["result"]["converged_count"],
                "peak_allocated_bytes": payload["result"][
                    "peak_allocated_bytes"
                ],
                "peak_reserved_bytes": payload["result"][
                    "peak_reserved_bytes"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
