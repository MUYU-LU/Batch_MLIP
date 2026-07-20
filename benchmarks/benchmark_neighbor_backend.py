#!/usr/bin/env python3
"""Compare matscipy and dense CUDA neighbor construction end to end."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read

from batch_mlip.core.neighbors import BACKEND, neighbor_list


@dataclass(frozen=True)
class WorkloadSpec:
    distribution: str
    pool_size: int
    resident_batch_size: int

    @property
    def workload_id(self) -> str:
        return f"EVAL-{self.distribution}-R{self.pool_size}-v1"


WORKLOADS = (
    WorkloadSpec("H46", 32, 32),
    WorkloadSpec("H46", 256, 128),
    WorkloadSpec("H276", 32, 16),
    WorkloadSpec("H276", 256, 16),
    WorkloadSpec("MIX", 32, 32),
    WorkloadSpec("MIX", 256, 32),
)


def _load_workload(spec: WorkloadSpec, manifest_dir: Path, dataset_dir: Path) -> list[Atoms]:
    document = json.loads((manifest_dir / f"{spec.workload_id}.json").read_text(encoding="utf-8"))
    cache: dict[str, Atoms] = {}
    systems = []
    for job in document["jobs"]:
        source_path = job["source_path"]
        if source_path not in cache:
            cache[source_path] = read(dataset_dir / source_path)
        systems.append(cache[source_path])
    return systems


def _offsets(systems: list[Atoms]) -> np.ndarray:
    return np.cumsum([0, *(len(atoms) for atoms in systems)], dtype=np.int64)


def _one_matscipy(atoms: Atoms, cutoff: float) -> tuple[np.ndarray, ...]:
    return neighbor_list("ijS", atoms, cutoff)


def _pack_cpu_graphs(
    graphs: list[tuple[np.ndarray, ...]],
    offsets: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = np.concatenate(
        [
            graph[0].astype(np.int64, copy=False) + offsets[index]
            for index, graph in enumerate(graphs)
        ]
    )
    neighbors = np.concatenate(
        [
            graph[1].astype(np.int64, copy=False) + offsets[index]
            for index, graph in enumerate(graphs)
        ]
    )
    shifts = np.concatenate([graph[2].astype(np.int64, copy=False) for graph in graphs])
    edge_index = torch.as_tensor(np.stack((centers, neighbors)), dtype=torch.long, device=device)
    shifts_device = torch.as_tensor(shifts, dtype=torch.long, device=device)
    return edge_index, shifts_device


def _run_matscipy(
    systems: list[Atoms],
    *,
    cutoff: float,
    workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workers == 1:
        graphs = [_one_matscipy(atoms, cutoff) for atoms in systems]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            graphs = list(executor.map(lambda atoms: _one_matscipy(atoms, cutoff), systems))
    return _pack_cpu_graphs(graphs, _offsets(systems), device)


def _shift_vectors(device: torch.device) -> torch.Tensor:
    values = torch.tensor((-1, 0, 1), dtype=torch.long, device=device)
    return torch.cartesian_prod(values, values, values)


def _dense_cuda_bucket(
    systems: list[Atoms],
    system_indices: list[int],
    offsets: np.ndarray,
    *,
    cutoff: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.as_tensor(
        np.stack([systems[index].positions for index in system_indices]),
        dtype=torch.float64,
        device=device,
    )
    cells = torch.as_tensor(
        np.stack([systems[index].cell.array for index in system_indices]),
        dtype=torch.float64,
        device=device,
    )
    atom_offsets = torch.as_tensor(offsets[system_indices], dtype=torch.long, device=device)
    shifts = _shift_vectors(device)
    cartesian_shifts = torch.einsum("sd,bdk->bsk", shifts.to(torch.float64), cells)
    base = positions[:, None, :, :] - positions[:, :, None, :]
    cutoff_squared = cutoff * cutoff
    edge_parts = []
    shift_parts = []
    atom_count = positions.shape[1]
    diagonal = torch.eye(atom_count, dtype=torch.bool, device=device).unsqueeze(0)
    zero_shift_index = shifts.shape[0] // 2
    for shift_index in range(shifts.shape[0]):
        delta = base + cartesian_shifts[:, shift_index, None, None, :]
        mask = torch.sum(delta * delta, dim=-1) < cutoff_squared
        if shift_index == zero_shift_index:
            mask &= ~diagonal
        entries = torch.nonzero(mask, as_tuple=False)
        if entries.numel() == 0:
            continue
        batch_ids, centers, neighbors = entries.unbind(dim=1)
        edge_parts.append(
            torch.stack((centers + atom_offsets[batch_ids], neighbors + atom_offsets[batch_ids]))
        )
        shift_parts.append(shifts[shift_index].expand(entries.shape[0], -1))
    return torch.cat(edge_parts, dim=1), torch.cat(shift_parts, dim=0)


def _run_dense_cuda(
    systems: list[Atoms], *, cutoff: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = _offsets(systems)
    by_size: dict[int, list[int]] = {}
    for index, atoms in enumerate(systems):
        by_size.setdefault(len(atoms), []).append(index)
    graphs = [
        _dense_cuda_bucket(
            systems,
            indices,
            offsets,
            cutoff=cutoff,
            device=device,
        )
        for _, indices in sorted(by_size.items())
    ]
    return (
        torch.cat([graph[0] for graph in graphs], dim=1),
        torch.cat([graph[1] for graph in graphs], dim=0),
    )


def _canonical_graph(edge_index: torch.Tensor, shifts: torch.Tensor) -> np.ndarray:
    values = torch.cat((edge_index.T, shifts), dim=1).detach().cpu().numpy()
    order = np.lexsort(tuple(values[:, column] for column in reversed(range(5))))
    return values[order]


def _measure(function: Any, device: torch.device) -> tuple[float, Any, float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = function()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        elapsed,
        output,
        torch.cuda.max_memory_allocated(device) / 1e9,
        torch.cuda.max_memory_reserved(device) / 1e9,
    )


def _measure_matscipy_phases(
    chunks: list[list[Atoms]], *, cutoff: float, device: torch.device
) -> dict[str, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    graphs_by_chunk = [[_one_matscipy(atoms, cutoff) for atoms in chunk] for chunk in chunks]
    search_time = time.perf_counter() - start

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    packed_chunks = [
        _pack_cpu_graphs(graphs, _offsets(chunk), device)
        for graphs, chunk in zip(graphs_by_chunk, chunks, strict=True)
    ]
    torch.cuda.synchronize(device)
    pack_and_transfer_time = time.perf_counter() - start
    del graphs_by_chunk, packed_chunks
    return {
        "search_time_s": search_time,
        "pack_and_transfer_time_s": pack_and_transfer_time,
        "search_fraction": search_time / (search_time + pack_and_transfer_time),
    }


def _run_point(
    systems: list[Atoms],
    spec: WorkloadSpec,
    *,
    cutoff: float,
    cpu_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    methods = {
        "matscipy_serial": lambda chunk: _run_matscipy(
            chunk, cutoff=cutoff, workers=1, device=device
        ),
        f"matscipy_threads_{cpu_workers}": lambda chunk: _run_matscipy(
            chunk, cutoff=cutoff, workers=cpu_workers, device=device
        ),
        "torch_cuda_dense": lambda chunk: _run_dense_cuda(chunk, cutoff=cutoff, device=device),
    }
    chunks = [
        systems[start : start + spec.resident_batch_size]
        for start in range(0, len(systems), spec.resident_batch_size)
    ]
    for function in methods.values():
        function(chunks[0])
        torch.cuda.synchronize(device)
    results = {}
    reference = None
    for name, function in methods.items():
        elapsed, packed_chunks, peak_allocated, peak_reserved = _measure(
            lambda function=function: [function(chunk) for chunk in chunks], device
        )
        edge_index = torch.cat([packed[0] for packed in packed_chunks], dim=1)
        shifts = torch.cat([packed[1] for packed in packed_chunks], dim=0)
        canonical = _canonical_graph(edge_index, shifts)
        if reference is None:
            reference = canonical
            exact = True
        else:
            exact = bool(np.array_equal(reference, canonical))
        results[name] = {
            "wall_time_s": elapsed,
            "structures_per_s": len(systems) / elapsed,
            "directed_edges": int(edge_index.shape[1]),
            "exact_vs_matscipy_serial": exact,
            "peak_allocated_GB": peak_allocated,
            "peak_reserved_GB": peak_reserved,
        }
        del packed_chunks, edge_index, shifts
    serial_time = results["matscipy_serial"]["wall_time_s"]
    for values in results.values():
        values["speedup_vs_matscipy_serial"] = serial_time / values["wall_time_s"]
    return {
        "workload_id": spec.workload_id,
        "distribution": spec.distribution,
        "pool_size": spec.pool_size,
        "resident_batch_size": spec.resident_batch_size,
        "cutoff_A": cutoff,
        "matscipy_serial_phase_probe": _measure_matscipy_phases(
            chunks, cutoff=cutoff, device=device
        ),
        "methods": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("benchmarks/workloads/manifests"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--cutoffs", default="6.0,6.5")
    parser.add_argument(
        "--resident-batches",
        default="",
        help="optional comma-separated resident batch scan for the selected workloads",
    )
    parser.add_argument(
        "--workloads",
        default=",".join(spec.workload_id for spec in WORKLOADS),
        help="comma-separated workload IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_workers <= 0:
        raise ValueError("cpu-workers must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    cutoffs = [float(value) for value in args.cutoffs.split(",")]
    requested = {value.strip() for value in args.workloads.split(",") if value.strip()}
    selected_workloads = [spec for spec in WORKLOADS if spec.workload_id in requested]
    if not selected_workloads or requested != {spec.workload_id for spec in selected_workloads}:
        raise ValueError("workloads must select known benchmark workload IDs")
    if args.resident_batches:
        resident_batches = [int(value) for value in args.resident_batches.split(",")]
        if any(batch_size <= 0 for batch_size in resident_batches):
            raise ValueError("resident batches must be positive")
        selected_workloads = [
            WorkloadSpec(spec.distribution, spec.pool_size, batch_size)
            for spec in selected_workloads
            for batch_size in resident_batches
            if batch_size <= spec.pool_size
        ]
    points = []
    for spec in selected_workloads:
        systems = _load_workload(spec, args.manifest_dir, args.dataset_dir)
        for cutoff in cutoffs:
            point = _run_point(
                systems,
                spec,
                cutoff=cutoff,
                cpu_workers=args.cpu_workers,
                device=device,
            )
            points.append(point)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"status": "running", "points": points}, indent=2) + "\n",
                encoding="utf-8",
            )
    result = {
        "schema_version": 1,
        "status": "passed"
        if all(
            method["exact_vs_matscipy_serial"]
            for point in points
            for method in point["methods"].values()
        )
        else "validation_failed",
        "comparison_scope": "ASE structures to packed graph tensors resident on CUDA",
        "gpu_algorithm": "dense O(N^2) over 27 periodic images, bucketed by atom count",
        "screening_repeats": 1,
        "warmup": "one resident chunk per method and point",
        "cpu_workers": args.cpu_workers,
        "neighbor_backend": BACKEND,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "workloads": [asdict(spec) for spec in selected_workloads],
        "points": points,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
