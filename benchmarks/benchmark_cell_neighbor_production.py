#!/usr/bin/env python3
"""Measure sparse cell-list effects in real-model EVAL, BFGS, and NVE."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_controlled_matrix import build_model_bundle  # noqa: E402

from batch_mlip import (  # noqa: E402
    FrechetCellFilter,
    RuntimeProfiler,
    batched_velocity_verlet,
    evaluate,
    initialize_maxwell_boltzmann,
    relax,
)


def _base_structure(atom_count: int, manifest_dir: Path, dataset_dir: Path):
    manifest = json.loads(
        (manifest_dir / f"EVAL-H{atom_count}-R32-v1.json").read_text(
            encoding="utf-8"
        )
    )
    return read(dataset_dir / manifest["jobs"][0]["source_path"])


def _family_structures(root: Path, family: str, count: int):
    paths = sorted((root / family / "structures").glob("*.cif"))
    if len(paths) < count:
        raise ValueError(f"{family} contains {len(paths)} CIFs, fewer than {count}")
    if count == 1:
        selected = [paths[0]]
    else:
        selected = [
            paths[round(index * (len(paths) - 1) / (count - 1))]
            for index in range(count)
        ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        systems = [read(path) for path in selected]
    return selected, systems


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _snapshot(output: Any, task: str) -> dict[str, torch.Tensor]:
    if task == "eval":
        state = output.state
        evaluation = output.evaluation
        values = {}
    else:
        state = output.state
        evaluation = output.evaluation
        values = {
            "velocities": state.velocities.detach().cpu(),
        }
    return {
        **values,
        "positions": state.positions.detach().cpu(),
        "cells": state.cells.detach().cpu(),
        "energy": evaluation.energy.detach().cpu(),
        "forces": evaluation.forces.detach().cpu(),
    }


def _execute(
    systems,
    bundle,
    *,
    task: str,
    nve_steps: int,
) -> tuple[Any, dict[str, Any]]:
    if task == "eval":
        return evaluate(systems, bundle.native, compute_stress=False), {}
    if task == "bfgs":
        return (
            relax(
                systems,
                bundle.native,
                optimizer="bfgs",
                cell_filter=FrechetCellFilter(),
                fmax=1e-30,
                smax=None,
                max_steps=3,
                optimizer_dtype="float64",
                active_compaction=False,
            ),
            {},
        )

    state = bundle.native.create_state(systems)
    initialize_maxwell_boltzmann(
        state,
        300.0,
        seed=20260726,
        remove_com=True,
        force_exact_temperature=True,
    )
    output = batched_velocity_verlet(
        state,
        bundle.native,
        timestep_fs=0.5,
        n_steps=nve_steps,
    )
    drift = output.evaluation.energy.detach() + output.kinetic_energy.detach()
    drift -= output.initial_total_energy.detach()
    return output, {
        "max_abs_total_energy_drift_eV_per_atom": float(
            torch.max(torch.abs(drift) / state.counts).item()
        )
    }


def _measure_backend(
    args: argparse.Namespace,
    systems,
    *,
    backend: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    args.resolved_neighbor_backends = [backend]
    bundle = build_model_bundle(args)
    device = bundle.device
    _execute(
        systems[:1],
        bundle,
        task=args.task,
        nve_steps=min(1, args.nve_steps),
    )
    for _ in range(args.warmups):
        _execute(
            systems,
            bundle,
            task=args.task,
            nve_steps=args.nve_steps,
        )
    timings = []
    diagnostics = {}
    profile = None
    output = None
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.repeats):
        with RuntimeProfiler(device=device) as profiler:
            _synchronize(device)
            started = time.perf_counter()
            output, diagnostics = _execute(
                systems,
                bundle,
                task=args.task,
                nve_steps=args.nve_steps,
            )
            _synchronize(device)
            timings.append(time.perf_counter() - started)
        profile = profiler.summary()
    if output is None or profile is None:
        raise RuntimeError("production measurement did not execute")
    snapshot = _snapshot(output, args.task)
    result = {
        "backend": backend,
        "wall_seconds": timings,
        "median_wall_seconds": statistics.median(timings),
        "minimum_wall_seconds": min(timings),
        "maximum_wall_seconds": max(timings),
        "systems_per_second": len(systems) / statistics.median(timings),
        "atoms_per_second": sum(len(atoms) for atoms in systems)
        / statistics.median(timings),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "runtime_profile": profile,
        **diagnostics,
    }
    del output, bundle
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize(device)
    return result, snapshot


def _differences(
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {
        f"max_abs_{name}_difference": float(
            torch.max(torch.abs(actual[name] - expected)).item()
        )
        for name, expected in reference.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("atombit", "mace"), required=True)
    parser.add_argument("--task", choices=("eval", "bfgs", "nve"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-atoms", type=int, choices=(46, 276))
    source.add_argument("--family")
    parser.add_argument("--repeat", default="1x1x1")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--backends", default="cuda_dense,cuda_cell")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nve-steps", type=int, default=10)
    parser.add_argument("--skin", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument(
        "--omc-root",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("benchmarks/workloads/manifests"),
    )
    parser.add_argument(
        "--atombit-checkpoint",
        type=Path,
        default=Path(
            "/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/"
            "smooth_rms_finetune/"
            "AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
        ),
    )
    parser.add_argument(
        "--atombit-e0",
        type=Path,
        default=Path(
            "/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/"
            "meta_e0_data_OMC_r6_single.pt"
        ),
    )
    parser.add_argument("--atombit-cutoff", type=float, default=6.0)
    parser.add_argument(
        "--mace-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-OFF23_small.model",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.repeats <= 0 or args.warmups < 0:
        parser.error("batch/repeats must be positive and warmups nonnegative")
    if args.nve_steps <= 0:
        parser.error("nve-steps must be positive")
    repeat = tuple(int(value) for value in args.repeat.split("x"))
    if len(repeat) != 3 or any(value <= 0 for value in repeat):
        parser.error("repeat must contain three positive integers")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.family is None:
        base = _base_structure(args.source_atoms, args.manifest_dir, args.dataset_dir)
        structure = base.repeat(repeat)
        systems = [structure.copy() for _ in range(args.batch_size)]
        source_paths = None
    else:
        if repeat != (1, 1, 1):
            parser.error("--repeat is only supported with --source-atoms")
        selected_paths, systems = _family_structures(
            args.omc_root,
            args.family,
            args.batch_size,
        )
        source_paths = [str(path) for path in selected_paths]
    methods = {}
    snapshots = {}
    for backend in (value.strip() for value in args.backends.split(",")):
        methods[backend], snapshots[backend] = _measure_backend(
            args,
            systems,
            backend=backend,
        )
    reference_name = next(iter(methods))
    for backend, result in methods.items():
        result["validation_vs_" + reference_name] = _differences(
            snapshots[reference_name],
            snapshots[backend],
        )
        result["speedup_vs_" + reference_name] = (
            methods[reference_name]["median_wall_seconds"]
            / result["median_wall_seconds"]
        )
    output = {
        "schema_version": 1,
        "status": "passed",
        "model": args.model,
        "task": args.task,
        "source_atoms": args.source_atoms,
        "family": args.family,
        "source_paths": source_paths,
        "repeat": repeat,
        "atom_counts": [len(atoms) for atoms in systems],
        "atoms_per_system": (
            len(systems[0])
            if all(len(atoms) == len(systems[0]) for atoms in systems)
            else None
        ),
        "batch_size": args.batch_size,
        "skin_A": args.skin,
        "nve_steps": args.nve_steps if args.task == "nve" else None,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "methods": methods,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
