#!/usr/bin/env python3
"""Optimize an ASE structure pool with automatic MACE batch scheduling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    MACEBatchCalculator,
    ReproducibilityConfig,
    configure_reproducibility,
    optimize_pool,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input structure file")
    parser.add_argument("output", type=Path, help="relaxed extxyz or trajectory file")
    parser.add_argument(
        "--model",
        required=True,
        help="MACE-OFF model name or local checkpoint path",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="comma-separated devices, for example cuda:0,cuda:1",
    )
    parser.add_argument("--index", default=":", help="ASE input index")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--policy",
        choices=("auto", "probe"),
        default="auto",
    )
    parser.add_argument(
        "--fixed-cell",
        action="store_true",
        help="disable FrechetCellFilter and optimize positions only",
    )
    args = parser.parse_args()
    if args.fmax <= 0.0 or args.max_steps <= 0 or args.skin < 0.0:
        parser.error("fmax/max-steps must be positive and skin non-negative")
    return args


def main() -> None:
    args = _parse_args()
    reproducibility = configure_reproducibility(
        ReproducibilityConfig(seed=args.seed)
    )
    devices = tuple(
        value.strip() for value in args.devices.split(",") if value.strip()
    )
    if not devices:
        raise ValueError("at least one device is required")

    structures = read(args.input, index=args.index)
    if not isinstance(structures, list):
        structures = [structures]
    if not structures:
        raise ValueError("the selected input contains no structures")

    calculator = MACEBatchCalculator.from_off(
        model=args.model,
        device=devices[0],
        dtype=torch.float64,
        graph_mode="cached",
        skin=args.skin,
        neighbor_backend="auto",
    )
    result = optimize_pool(
        structures,
        calculator=calculator,
        devices=devices,
        optimizer="bfgs",
        cell_filter=None if args.fixed_cell else "frechet",
        policy=args.policy,
        fmax=args.fmax,
        max_steps=args.max_steps,
        max_step=0.2,
        alpha=70.0,
        linear_algebra_backend="auto",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result.structures)
    interface = result.metadata["optimize_pool"]
    print(
        json.dumps(
            {
                "pool_size": len(structures),
                "converged": int(result.converged.sum().item()),
                "output": str(args.output),
                "schedule": result.schedule,
                "capacity_planning": interface["capacity_planning"],
                "allocator": interface["allocator"],
                "executor_shutdown": interface["executor_shutdown"],
                "reproducibility": reproducibility,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
