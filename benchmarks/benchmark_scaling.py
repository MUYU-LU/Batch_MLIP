"""Measure batched versus sequential evaluation throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from atombit_batch.toy_models import PairHarmonicModel

from atombit_batch import AseGraphBatch, BatchedPotential


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_systems(n_systems: int, atoms_per_system: int, seed: int):
    rng = np.random.default_rng(seed)
    systems = []
    for _ in range(n_systems):
        # Randomized grid prevents severe overlaps while producing local edges.
        side = int(np.ceil(atoms_per_system ** (1.0 / 3.0)))
        grid = np.stack(
            np.meshgrid(np.arange(side), np.arange(side), np.arange(side), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)[:atoms_per_system]
        positions = 1.35 * grid + rng.normal(scale=0.03, size=(atoms_per_system, 3))
        systems.append(Atoms("H" * atoms_per_system, positions=positions))
    return systems


def timed(callable_, repeats: int, device: torch.device):
    for _ in range(3):
        callable_()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        callable_()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--output", default="runs/benchmark.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    systems = make_systems(args.systems, args.atoms, seed=123)
    model = PairHarmonicModel(cutoff=3.0)
    potential = BatchedPotential(model, device=device, dtype=dtype)
    batch = AseGraphBatch.from_ase(
        systems, cutoff=3.0, skin=args.skin, device=device, dtype=dtype
    )
    singles = [
        AseGraphBatch.from_ase(
            [atoms], cutoff=3.0, skin=args.skin, device=device, dtype=dtype
        )
        for atoms in systems
    ]

    batch_samples = timed(lambda: potential(batch), args.repeats, device)
    sequential_samples = timed(
        lambda: [potential(single) for single in singles], args.repeats, device
    )
    batch_median = float(np.median(batch_samples))
    sequential_median = float(np.median(sequential_samples))
    result = {
        "systems": args.systems,
        "atoms_per_system": args.atoms,
        "total_atoms": args.systems * args.atoms,
        "device": str(device),
        "dtype": args.dtype,
        "skin": args.skin,
        "batch_seconds": batch_samples,
        "sequential_seconds": sequential_samples,
        "batch_median_seconds": batch_median,
        "sequential_median_seconds": sequential_median,
        "speedup": sequential_median / batch_median,
        "systems_per_second": args.systems / batch_median,
        "atoms_per_second": (args.systems * args.atoms) / batch_median,
        "edges": int(batch.edge_index.shape[1]),
        "neighbor_rebuild_count": batch.neighbor_rebuild_count,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
