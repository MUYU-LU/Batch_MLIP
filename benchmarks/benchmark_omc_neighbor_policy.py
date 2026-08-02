#!/usr/bin/env python3
"""Benchmark complete neighbor-backend decisions on unique OMC structures."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from ase.io import read

from batch_mlip import AseGraphBatch
from batch_mlip.core import neighbors as neighbor_module
from batch_mlip.core.cell_neighbors import estimate_cell_candidate_reduction


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evenly_spaced_paths(directory: Path, count: int) -> list[Path]:
    paths = sorted(directory.glob("*.cif"))
    if len(paths) < count:
        raise ValueError(f"{directory} contains {len(paths)} CIFs, fewer than {count}")
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=np.int64)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("evenly spaced selection produced duplicate CIF indices")
    return [paths[index] for index in indices]


def _load_family(root: Path, family: str, count: int):
    paths = _evenly_spaced_paths(root / family / "structures", count)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        systems = [read(path) for path in paths]
    return paths, systems


def _selected_tensors(state: AseGraphBatch, system_ids: list[int]):
    graph_ids = torch.as_tensor(system_ids, device=state.device, dtype=torch.long)
    atom_ids = torch.cat(
        [
            torch.arange(
                state.ptr[index],
                state.ptr[index + 1],
                device=state.device,
                dtype=torch.long,
            )
            for index in system_ids
        ]
    )
    return graph_ids, atom_ids


def _resolve(state: AseGraphBatch, system_ids: list[int]) -> dict[str, Any]:
    graph_ids, atom_ids = _selected_tensors(state, system_ids)
    kwargs = {
        "device": state.device,
        "counts": state.counts[graph_ids],
        "cutoff": state.cutoff + state.skin,
        "cells": state.cells[graph_ids],
        "pbc": state.pbc[graph_ids],
        "positions": state.positions[atom_ids],
    }
    reference_valid = state._neighbor_reference_valid[graph_ids]
    selected_pbc = state.pbc[graph_ids]
    if bool(reference_valid.all()) and bool(selected_pbc.all()):
        edge_owners = state.system_idx[state.edge_index[0]]
        edge_counts = torch.bincount(edge_owners, minlength=state.n_systems)
        volumes = torch.linalg.det(state.cells[graph_ids]).abs()
        if bool(torch.isfinite(volumes).all()) and bool((volumes > 0.0).all()):
            kwargs["candidate_edges"] = int(edge_counts[graph_ids].sum().item())
            kwargs["mean_volume_per_atom"] = float(
                (volumes / state.counts[graph_ids]).mean().item()
            )
    decision_function = getattr(
        neighbor_module,
        "resolve_neighbor_backend_decision",
        None,
    )
    if decision_function is None:
        legacy_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in {"candidate_edges", "mean_volume_per_atom"}
        }
        backend = neighbor_module.resolve_neighbor_backend(
            state.neighbor_backend,
            **legacy_kwargs,
        )
        return {
            "backend": backend,
            "reason": "legacy resolver",
            "pair_work": int(torch.sum(kwargs["counts"].to(torch.int64) ** 2).item()),
            "candidate_reduction": None,
        }
    decision = decision_function(state.neighbor_backend, **kwargs)
    return {
        "backend": decision.backend,
        "reason": decision.reason,
        "pair_work": decision.pair_work,
        "candidate_reduction": decision.candidate_reduction,
    }


def _policy_features(state: AseGraphBatch, system_ids: list[int]) -> dict[str, Any]:
    graph_ids, atom_ids = _selected_tensors(state, system_ids)
    counts = state.counts[graph_ids]
    edge_owners = state.system_idx[state.edge_index[0]]
    edge_counts = torch.bincount(edge_owners, minlength=state.n_systems)[graph_ids]
    cells = state.cells[graph_ids]
    atom_count = int(counts.sum().item())
    candidate_edges = int(edge_counts.sum().item())
    pair_work = int(torch.sum(counts.to(torch.int64) ** 2).item())
    volumes = torch.linalg.det(cells).abs()
    reduction = estimate_cell_candidate_reduction(
        cells,
        state.pbc[graph_ids],
        cutoff=state.cutoff + state.skin,
        positions=state.positions[atom_ids],
        counts=counts,
    )
    return {
        "system_count": len(system_ids),
        "atom_count": atom_count,
        "minimum_atoms_per_system": int(counts.min().item()),
        "maximum_atoms_per_system": int(counts.max().item()),
        "pair_work": pair_work,
        "candidate_edges": candidate_edges,
        "candidate_edges_per_atom": candidate_edges / atom_count,
        "candidate_edges_per_pair_work": candidate_edges / pair_work,
        "minimum_volume_A3": float(volumes.min().item()),
        "mean_volume_A3": float(volumes.mean().item()),
        "maximum_volume_A3": float(volumes.max().item()),
        "mean_volume_per_atom_A3": float((volumes / counts).mean().item()),
        "estimated_cell_candidate_reduction": reduction,
    }


def _measure_selection(
    state: AseGraphBatch,
    system_ids: list[int],
    *,
    warmups: int,
    repeats: int,
) -> tuple[list[float], dict[str, Any]]:
    for _ in range(warmups):
        decision = _resolve(state, system_ids)
    timings = []
    for _ in range(repeats):
        _synchronize(state.device)
        started = time.perf_counter()
        decision = _resolve(state, system_ids)
        _synchronize(state.device)
        timings.append(time.perf_counter() - started)
    return timings, decision


def _measure_rebuild(
    state: AseGraphBatch,
    system_ids: list[int],
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        state.rebuild_neighbor_list(system_ids)
    timings = []
    if state.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(state.device)
    for _ in range(repeats):
        _synchronize(state.device)
        started = time.perf_counter()
        state.rebuild_neighbor_list(system_ids)
        _synchronize(state.device)
        timings.append(time.perf_counter() - started)
    return timings


def _one_backend(
    systems,
    reference_edges: torch.Tensor,
    reference_shifts: torch.Tensor,
    *,
    backend: str,
    candidate_cutoff: float,
    rebuilt_systems: list[int],
    device: torch.device,
    warmups: int,
    repeats: int,
) -> dict[str, Any]:
    state = AseGraphBatch.from_ase(
        systems,
        cutoff=candidate_cutoff,
        skin=0.0,
        device=device,
        dtype=torch.float32,
        neighbor_backend=backend,
        build_neighbors=False,
    )
    state.rebuild_neighbor_list()
    selection_times, decision = _measure_selection(
        state,
        rebuilt_systems,
        warmups=warmups,
        repeats=repeats,
    )
    rebuild_times = _measure_rebuild(
        state,
        rebuilt_systems,
        warmups=warmups,
        repeats=repeats,
    )
    exact = bool(
        torch.equal(state.edge_index.cpu(), reference_edges)
        and torch.equal(state.shifts_int.cpu(), reference_shifts)
    )
    median_rebuild = statistics.median(rebuild_times)
    result = {
        "backend": backend,
        "resolved_backend": decision["backend"],
        "decision_reason": decision["reason"],
        "pair_work": decision["pair_work"],
        "candidate_reduction": decision["candidate_reduction"],
        "selection_seconds": selection_times,
        "median_selection_seconds": statistics.median(selection_times),
        "rebuild_seconds": rebuild_times,
        "median_rebuild_seconds": median_rebuild,
        "rebuilt_systems_per_second": len(rebuilt_systems) / median_rebuild,
        "total_candidate_edges": state.edge_index.shape[1],
        "exact_ordered_vs_matscipy": exact,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _synchronize(device)
    return result


def _one_family(
    args: argparse.Namespace,
    family: str,
    device: torch.device,
) -> dict[str, Any]:
    paths, systems = _load_family(args.dataset_root, family, args.resident_systems)
    points = []
    for candidate_cutoff in args.candidate_cutoffs:
        reference = AseGraphBatch.from_ase(
            systems,
            cutoff=candidate_cutoff,
            skin=0.0,
            device=device,
            dtype=torch.float32,
            neighbor_backend="matscipy",
        )
        reference_edges = reference.edge_index.cpu()
        reference_shifts = reference.shifts_int.cpu()
        for rebuilt_count in args.rebuilt_system_counts:
            system_ids = list(range(rebuilt_count))
            policy_features = _policy_features(reference, system_ids)
            methods = {}
            for backend in args.backends:
                methods[backend] = _one_backend(
                    systems,
                    reference_edges,
                    reference_shifts,
                    backend=backend,
                    candidate_cutoff=candidate_cutoff,
                    rebuilt_systems=system_ids,
                    device=device,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            fastest = min(
                ("matscipy", "cuda_dense", "cuda_cell"),
                key=lambda name: methods[name]["median_rebuild_seconds"],
            )
            auto_time = methods["auto"]["median_rebuild_seconds"]
            methods["auto"]["regret_vs_fastest"] = (
                auto_time / methods[fastest]["median_rebuild_seconds"] - 1.0
            )
            points.append(
                {
                    "candidate_cutoff_A": candidate_cutoff,
                    "resident_systems": len(systems),
                    "rebuilt_systems": rebuilt_count,
                    "policy_features": policy_features,
                    "fastest_explicit_backend": fastest,
                    "methods": methods,
                }
            )
        del reference
    return {
        "family": family,
        "sample_count": len(systems),
        "source_paths": [str(path) for path in paths],
        "atom_counts": [len(atoms) for atoms in systems],
        "cell_volumes_A3": [float(atoms.get_volume()) for atoms in systems],
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument(
        "--families",
        default="GUFJOG,XATMOV,BOQWIN,XAFPAY,OBEQIX,rof-b",
    )
    parser.add_argument("--resident-systems", type=int, default=64)
    parser.add_argument("--rebuilt-system-counts", default="1,2,4,8,16,32,64")
    parser.add_argument("--candidate-cutoffs", default="6.0,6.5")
    parser.add_argument("--backends", default="matscipy,cuda_dense,cuda_cell,auto")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.families = [value.strip() for value in args.families.split(",")]
    args.rebuilt_system_counts = [
        int(value) for value in args.rebuilt_system_counts.split(",")
    ]
    args.candidate_cutoffs = [float(value) for value in args.candidate_cutoffs.split(",")]
    args.backends = [value.strip() for value in args.backends.split(",")]
    if set(args.backends) != {"matscipy", "cuda_dense", "cuda_cell", "auto"}:
        parser.error("the frozen matrix requires matscipy,cuda_dense,cuda_cell,auto")
    if any(value <= 0 or value > args.resident_systems for value in args.rebuilt_system_counts):
        parser.error("rebuilt-system counts must be within the resident batch")
    if args.warmups < 0 or args.repeats <= 0:
        parser.error("warmups must be nonnegative and repeats must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen OMC neighbor benchmark requires CUDA")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    families = []
    for family in args.families:
        families.append(_one_family(args, family, device))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"status": "running", "families": families}, indent=2) + "\n",
            encoding="utf-8",
        )

    points = [point for result in families for point in result["points"]]
    topology_passed = all(
        method["exact_ordered_vs_matscipy"]
        for point in points
        for method in point["methods"].values()
    )
    result = {
        "schema_version": 1,
        "status": "passed" if topology_passed else "validation_failed",
        "baseline_commit": "9dd272821f8f93d1e19842a2ee751a3b9ddf30c1",
        "resident_systems": args.resident_systems,
        "rebuilt_system_counts": args.rebuilt_system_counts,
        "candidate_cutoffs_A": args.candidate_cutoffs,
        "warmups": args.warmups,
        "timing_repeats": args.repeats,
        "families": families,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not topology_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
