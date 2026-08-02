#!/usr/bin/env python3
"""Exercise the public optimize_pool interface on a signed MACE workload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    AutoSchedulerConfig,
    MACEBatchCalculator,
    ReproducibilityConfig,
    configure_reproducibility,
    optimize_pool,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mace-model", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pool_size <= 0 or args.max_steps <= 0:
        parser.error("pool size and maximum steps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    reproducibility = configure_reproducibility(
        ReproducibilityConfig(),
        require_preconfigured_python_hash=True,
    )
    manifest = read_workload_manifest(args.manifest)
    jobs = manifest.jobs[: args.pool_size]
    if len(jobs) != args.pool_size:
        raise ValueError("requested pool exceeds the signed manifest")
    systems = [
        read(args.dataset_dir / job.source_path, index=job.frame_index)
        for job in jobs
    ]
    devices = tuple(value.strip() for value in args.devices.split(","))
    calculator = MACEBatchCalculator.from_off(
        model=args.mace_model,
        device=devices[0],
        dtype=torch.float64,
        graph_mode="cached",
        skin=0.5,
        neighbor_backend="auto",
    )
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
            memory_safety_fraction=0.85,
            memory_growth_margin=1.10,
            multi_gpu_target_chunks_per_device=2,
            multi_gpu_queue_policy="bucket_stratified",
        ),
        fmax=1e-30,
        max_steps=args.max_steps,
        max_step=0.2,
        alpha=70.0,
        linear_algebra_backend="auto",
    )
    schedule = result.metadata["scheduling"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "pool_size": len(systems),
        "converged_count": int(result.converged.sum().item()),
        "model_evaluations": result.model_evaluations,
        "capacity_planning": schedule["capacity_planning"],
        "capacity_policy_resolution": schedule[
            "capacity_policy_resolution"
        ],
        "probe": schedule["probe"],
        "allocator": schedule["allocator"],
        "summary": result.schedule,
        "interface": result.metadata["optimize_pool"],
        "reproducibility": reproducibility,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["capacity_planning"], sort_keys=True))


if __name__ == "__main__":
    main()
