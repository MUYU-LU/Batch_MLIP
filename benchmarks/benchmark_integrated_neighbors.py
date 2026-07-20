#!/usr/bin/env python3
"""Benchmark integrated CPU and CUDA neighbor rebuilds on frozen workloads."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import read

from batch_mlip import AseGraphBatch


def _load_workload(distribution: str, manifest_dir: Path, dataset_dir: Path) -> list[Atoms]:
    path = manifest_dir / f"EVAL-{distribution}-R32-v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    cache: dict[str, Atoms] = {}
    systems = []
    for job in document["jobs"]:
        source_path = job["source_path"]
        if source_path not in cache:
            cache[source_path] = read(dataset_dir / source_path)
        systems.append(cache[source_path])
    return systems


def _states(
    systems: list[Atoms],
    *,
    batch_size: int,
    cutoff: float,
    backend: str,
    device: torch.device,
) -> list[AseGraphBatch]:
    return [
        AseGraphBatch.from_ase(
            systems[start : start + batch_size],
            cutoff=cutoff,
            device=device,
            dtype=torch.float32,
            neighbor_backend=backend,
            build_neighbors=False,
        )
        for start in range(0, len(systems), batch_size)
    ]


def _measure(states: list[AseGraphBatch], device: torch.device) -> tuple[float, float, float]:
    for state in states:
        state.rebuild_neighbor_list()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for state in states:
        state.rebuild_neighbor_list()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        elapsed,
        torch.cuda.max_memory_allocated(device) / 1e9,
        torch.cuda.max_memory_reserved(device) / 1e9,
    )


def _graph_values(states: list[AseGraphBatch]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(state.edge_index.detach().cpu(), state.shifts_int.detach().cpu()) for state in states]


def _run_point(
    systems: list[Atoms],
    *,
    distribution: str,
    batch_size: int,
    cutoff: float,
    device: torch.device,
) -> dict[str, Any]:
    methods = {}
    reference = None
    for backend in ("matscipy", "cuda_dense"):
        states = _states(
            systems,
            batch_size=batch_size,
            cutoff=cutoff,
            backend=backend,
            device=device,
        )
        elapsed, peak_allocated, peak_reserved = _measure(states, device)
        values = _graph_values(states)
        exact = reference is None or all(
            torch.equal(edge, reference_edge) and torch.equal(shifts, reference_shifts)
            for (edge, shifts), (reference_edge, reference_shifts) in zip(
                values, reference, strict=True
            )
        )
        if reference is None:
            reference = values
        methods[backend] = {
            "wall_time_s": elapsed,
            "structures_per_s": len(systems) / elapsed,
            "directed_edges": sum(edge.shape[1] for edge, _ in values),
            "peak_allocated_GB": peak_allocated,
            "peak_reserved_GB": peak_reserved,
            "exact_ordered_vs_matscipy": exact,
        }
        del states, values
        torch.cuda.empty_cache()
    methods["matscipy"]["speedup_vs_matscipy"] = 1.0
    methods["cuda_dense"]["speedup_vs_matscipy"] = (
        methods["matscipy"]["wall_time_s"] / methods["cuda_dense"]["wall_time_s"]
    )
    return {
        "distribution": distribution,
        "pool_size": len(systems),
        "resident_batch_size": batch_size,
        "cutoff_A": cutoff,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("benchmarks/workloads/manifests"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distributions", default="H46,H276")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--cutoffs", default="4.5,6.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("integrated neighbor benchmarking requires CUDA")
    distributions = [value.strip() for value in args.distributions.split(",")]
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    cutoffs = [float(value) for value in args.cutoffs.split(",")]
    points = []
    for distribution in distributions:
        systems = _load_workload(distribution, args.manifest_dir, args.dataset_dir)
        for cutoff in cutoffs:
            for batch_size in batch_sizes:
                if batch_size > len(systems):
                    continue
                points.append(
                    _run_point(
                        systems,
                        distribution=distribution,
                        batch_size=batch_size,
                        cutoff=cutoff,
                        device=device,
                    )
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"status": "running", "points": points}, indent=2) + "\n",
                    encoding="utf-8",
                )
    status = (
        "passed"
        if all(point["methods"]["cuda_dense"]["exact_ordered_vs_matscipy"] for point in points)
        else "validation_failed"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "scope": "resident AseGraphBatch geometry through integrated graph replacement",
        "screening_repeats": 1,
        "warmup": "one complete rebuild per resident chunk and backend",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "points": points,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
