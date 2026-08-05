#!/usr/bin/env python3
"""Optimize an OMC-CSP candidate pool with the frozen AtomBit policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase.io import read, write

from batch_mlip import (
    AtomBitBatchCalculator,
    ReproducibilityConfig,
    configure_reproducibility,
    optimize_pool,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Multi-frame extxyz input")
    parser.add_argument("output", type=Path, help="Relaxed extxyz output")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--e0", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_reproducibility(ReproducibilityConfig())
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices:
        raise ValueError("--devices must contain at least one device")

    structures = read(args.input, index=":")
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
    result = optimize_pool(
        structures,
        calculator,
        devices=devices,
        optimizer="bfgs",
        cell_filter="frechet",
        policy="auto",
        fmax=args.fmax,
        max_steps=args.max_steps,
        max_step=0.2,
        alpha=70.0,
        linear_algebra_backend="auto",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result.structures, format="extxyz")
    print(result.schedule)
    print(result.metadata["optimize_pool"]["capacity_planning"])


if __name__ == "__main__":
    main()
