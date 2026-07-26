#!/usr/bin/env python3
"""Benchmark integrated Matscipy, dense CUDA, and sparse CUDA cell lists."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batch_mlip import AseGraphBatch  # noqa: E402


@dataclass(frozen=True)
class Point:
    source_atoms: int
    repeat: tuple[int, int, int]
    batch_size: int

    @property
    def name(self) -> str:
        repeat = "x".join(str(value) for value in self.repeat)
        return f"H{self.source_atoms}-S{repeat}-B{self.batch_size}"


DEFAULT_POINTS = (
    Point(46, (1, 1, 1), 1),
    Point(46, (1, 1, 1), 32),
    Point(276, (1, 1, 1), 1),
    Point(276, (1, 1, 1), 16),
    Point(46, (2, 2, 2), 8),
    Point(46, (3, 3, 3), 2),
    Point(276, (2, 2, 2), 2),
)


def _parse_points(value: str) -> list[Point]:
    if not value:
        return list(DEFAULT_POINTS)
    points = []
    for item in value.split(","):
        source, repeat, batch = item.split(":")
        points.append(
            Point(
                source_atoms=int(source.removeprefix("H")),
                repeat=tuple(int(part) for part in repeat.split("x")),
                batch_size=int(batch.removeprefix("B")),
            )
        )
    if any(len(point.repeat) != 3 or point.batch_size <= 0 for point in points):
        raise ValueError("points must use H46:2x2x2:B8 syntax")
    return points


def _load_base(atom_count: int, manifest_dir: Path, dataset_dir: Path) -> Atoms:
    manifest = json.loads(
        (manifest_dir / f"EVAL-H{atom_count}-R32-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return read(dataset_dir / manifest["jobs"][0]["source_path"])


def _cell_statistics(atoms: Atoms, cutoff: float) -> dict[str, Any]:
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    reciprocal_norm = np.linalg.norm(np.linalg.inv(cell), axis=0)
    fractional_radius = cutoff * reciprocal_norm
    bins = np.maximum(1, np.floor(1.0 / fractional_radius)).astype(np.int64)
    extents = np.ceil(fractional_radius * bins).astype(np.int64)
    inverse = np.linalg.inv(cell)
    fractional = atoms.positions @ inverse
    fractional -= np.floor(fractional)
    coordinates = np.floor(fractional * bins).astype(np.int64)
    coordinates = np.minimum(coordinates, bins - 1)
    linear = (coordinates[:, 0] * bins[1] + coordinates[:, 1]) * bins[2]
    linear += coordinates[:, 2]
    occupancy = np.bincount(linear, minlength=int(np.prod(bins)))
    offsets = np.stack(
        np.meshgrid(
            *[
                np.arange(-extent, extent + 1, dtype=np.int64)
                for extent in extents
            ],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    candidates = 0
    for coordinate in coordinates:
        targets = np.remainder(coordinate + offsets, bins)
        target_linear = (targets[:, 0] * bins[1] + targets[:, 1]) * bins[2]
        target_linear += targets[:, 2]
        candidates += int(occupancy[target_linear].sum())
    image_extents = np.ceil(fractional_radius + 1.0).astype(np.int64) - 1
    dense_candidates = len(atoms) ** 2 * math.prod(2 * image_extents + 1)
    return {
        "bins": bins.tolist(),
        "occupied_bins": int(np.count_nonzero(occupancy)),
        "max_bin_occupancy": int(occupancy.max()),
        "offset_extents": extents.tolist(),
        "neighbor_bin_count": int(np.prod(2 * extents + 1)),
        "cell_candidate_pairs": candidates,
        "dense_candidate_pairs": int(dense_candidates),
        "candidate_reduction": 1.0 - candidates / dense_candidates,
    }


def _state(
    systems: list[Atoms],
    *,
    cutoff: float,
    backend: str,
    device: torch.device,
) -> AseGraphBatch:
    return AseGraphBatch.from_ase(
        systems,
        cutoff=cutoff,
        device=device,
        dtype=torch.float32,
        neighbor_backend=backend,
        build_neighbors=False,
    )


def _measure(
    state: AseGraphBatch,
    *,
    warmups: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmups):
        state.rebuild_neighbor_list()
    timings = []
    peaks = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        state.rebuild_neighbor_list()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        peaks.append(torch.cuda.max_memory_allocated(device))
    return {
        "wall_seconds": timings,
        "median_wall_seconds": statistics.median(timings),
        "minimum_wall_seconds": min(timings),
        "maximum_wall_seconds": max(timings),
        "systems_per_second": state.n_systems / statistics.median(timings),
        "atoms_per_second": state.n_atoms / statistics.median(timings),
        "directed_edges": state.edge_index.shape[1],
        "peak_allocated_bytes": max(peaks),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _run_point(
    point: Point,
    *,
    cutoff: float,
    base: Atoms,
    warmups: int,
    repeats: int,
    device: torch.device,
    backends: tuple[str, ...],
) -> dict[str, Any]:
    structure = base.repeat(point.repeat)
    systems = [structure.copy() for _ in range(point.batch_size)]
    methods = {}
    reference = None
    for backend in backends:
        state = _state(
            systems,
            cutoff=cutoff,
            backend=backend,
            device=device,
        )
        result = _measure(
            state,
            warmups=warmups,
            repeats=repeats,
            device=device,
        )
        graph = (
            state.edge_index.detach().cpu(),
            state.shifts_int.detach().cpu(),
        )
        result["exact_ordered_vs_matscipy"] = (
            True
            if reference is None
            else torch.equal(graph[0], reference[0])
            and torch.equal(graph[1], reference[1])
        )
        if reference is None:
            reference = graph
        methods[backend] = result
        del state, graph
        torch.cuda.empty_cache()
    baseline = methods["cuda_dense"]["median_wall_seconds"]
    for result in methods.values():
        result["speedup_vs_cuda_dense"] = (
            baseline / result["median_wall_seconds"]
        )
    return {
        "point": asdict(point),
        "name": point.name,
        "atoms_per_system": len(structure),
        "cutoff_A": cutoff,
        "cell_statistics": _cell_statistics(structure, cutoff),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("benchmarks/workloads/manifests"),
    )
    parser.add_argument("--points", default="")
    parser.add_argument("--cutoffs", default="4.5,6.0")
    parser.add_argument(
        "--backends",
        default="matscipy,cuda_dense,cuda_cell",
        help="Comma-separated neighbor backends; the first is the topology reference",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("cell-list benchmarking requires CUDA")
    points = _parse_points(args.points)
    backends = tuple(value.strip() for value in args.backends.split(",") if value.strip())
    if not backends or "cuda_dense" not in backends:
        parser.error("backends must be nonempty and include cuda_dense")
    bases = {
        atom_count: _load_base(atom_count, args.manifest_dir, args.dataset_dir)
        for atom_count in {point.source_atoms for point in points}
    }
    results = []
    for cutoff in (float(value) for value in args.cutoffs.split(",")):
        for point in points:
            results.append(
                _run_point(
                    point,
                    cutoff=cutoff,
                    base=bases[point.source_atoms],
                    warmups=args.warmups,
                    repeats=args.repeats,
                    device=device,
                    backends=backends,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"status": "running", "points": results}, indent=2)
                + "\n",
                encoding="utf-8",
            )
    status = (
        "passed"
        if all(
            method["exact_ordered_vs_matscipy"]
            for point in results
            for method in point["methods"].values()
        )
        else "validation_failed"
    )
    output = {
        "schema_version": 1,
        "status": status,
        "scope": "resident tensors through integrated graph replacement",
        "warmups": args.warmups,
        "repeats": args.repeats,
        "backends": backends,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "points": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
