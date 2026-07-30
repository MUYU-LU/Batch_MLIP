#!/usr/bin/env python3
"""Measure one true single-resident-batch AtomBit/BFGS calibration point."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

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

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    FrechetCellFilter,
    RuntimeProfiler,
    relax,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--indices-json",
        type=Path,
        help="Optional exact workload indices selected by the offline planner.",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("maximum steps must be positive")
    if args.indices_json is None and (
        args.batch_size is None or args.batch_size <= 0
    ):
        parser.error("provide a positive batch size or --indices-json")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = read_workload_manifest(args.manifest)
    if args.indices_json is None:
        indices = tuple(range(int(args.batch_size)))
    else:
        selection = json.loads(args.indices_json.read_text(encoding="utf-8"))
        if selection["workload_id"] != manifest.workload_id:
            parser.error("index selection workload differs from manifest")
        if selection["workload_manifest_sha256"] != manifest.manifest_sha256:
            parser.error("index selection manifest hash differs from manifest")
        indices = tuple(int(index) for index in selection["indices"])
    if not indices or len(set(indices)) != len(indices):
        parser.error("selected workload indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= len(manifest.jobs):
        parser.error("selected workload index is outside the manifest")
    jobs = tuple(manifest.jobs[index] for index in indices)
    if args.batch_size is not None and len(jobs) != args.batch_size:
        parser.error("manifest contains fewer jobs than the requested batch")
    systems = [
        read(args.dataset_dir / job.source_path, index=job.frame_index)
        for job in jobs
    ]
    device = torch.device(args.device)
    model, model_metadata = load_production_model(args.checkpoint)
    calculator = AtomBitBatchCalculator(
        model.to(device=device, dtype=torch.float32).eval(),
        cutoff=6.0,
        skin=0.5,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
        neighbor_backend="auto",
    )
    calculator(
        calculator.create_state([systems[0]]),
        compute_stress=True,
    )
    synchronize(device)
    options = {
        "optimizer": "bfgs",
        "scheduling": "single_batch",
        "cell_filter": FrechetCellFilter(),
        "fmax": 1e-12,
        "smax": None,
        "max_steps": args.max_steps,
        "max_step": 0.2,
        "callback_interval": args.max_steps + 1,
        "alpha": 70.0,
        "optimizer_dtype": "float64",
        "linear_algebra_backend": "auto",
    }

    warm_started = time.perf_counter()
    warm = relax(systems, calculator, **options)
    synchronize(device)
    warm_seconds = time.perf_counter() - warm_started

    del warm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    with RuntimeProfiler(device=device) as profiler:
        measured = relax(systems, calculator, **options)
    synchronize(device)
    measured_seconds = time.perf_counter() - started
    output = {
        "schema_version": 1,
        "status": "complete",
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "batch_size": len(jobs),
        "workload_indices": list(indices),
        "atom_count": sum(len(system) for system in systems),
        "model_evaluations": measured.model_evaluations,
        "graph_evaluations": measured.graph_evaluations,
        "warm_execution_seconds": warm_seconds,
        "measured_execution_seconds": measured_seconds,
        "seconds_per_evaluation": (
            measured_seconds / measured.model_evaluations
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_profile": profiler.summary(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": model_metadata,
        "environment": environment_metadata(device),
        "contract": {
            "optimizer": "BatchedBFGS",
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "cutoff_A": 6.0,
            "skin_A": 0.5,
            "force_mode": "autograd",
            "neighbor_backend": "auto",
            "max_steps": args.max_steps,
            "scheduling": "single_batch",
            "deterministic": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "batch_size": len(jobs),
                "model_evaluations": measured.model_evaluations,
                "output": str(args.output),
                "peak_reserved_bytes": output["peak_reserved_bytes"],
                "seconds_per_evaluation": output[
                    "seconds_per_evaluation"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
