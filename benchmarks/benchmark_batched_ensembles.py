#!/usr/bin/env python3
"""Benchmark persistent-state batched NVT and isotropic MTK NPT."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_controlled_matrix import build_model_bundle  # noqa: E402

from batch_mlip import (  # noqa: E402
    batched_isotropic_mtk,
    batched_langevin_baoab,
    initialize_maxwell_boltzmann,
)
from batch_mlip.workloads import (  # noqa: E402
    materialize_workload,
    read_workload_manifest,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run(
    systems,
    calculator,
    *,
    ensemble: str,
    steps: int,
    seed: int,
) -> Any:
    state = calculator.create_state(systems)
    initialize_maxwell_boltzmann(
        state,
        300.0,
        seed=seed,
        remove_com=True,
        force_exact_temperature=True,
    )
    if ensemble == "nvt":
        return batched_langevin_baoab(
            state,
            calculator,
            timestep_fs=0.5,
            n_steps=steps,
            temperature_K=300.0,
            friction_per_fs=0.01,
            seed=seed + 1,
        )
    return batched_isotropic_mtk(
        state,
        calculator,
        timestep_fs=0.5,
        n_steps=steps,
        temperature_K=300.0,
        pressure_eV_per_A3=0.0,
        thermostat_damping_fs=50.0,
        barostat_damping_fs=500.0,
    )


def _tensor_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("atombit", "mace"), required=True)
    parser.add_argument("--ensemble", choices=("nvt", "npt"), required=True)
    parser.add_argument("--atom-count", type=int, choices=(46, 276), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measured-steps", type=int, default=100)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--neighbor-backend", default="auto")
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument(
        "--atombit-checkpoint",
        type=Path,
        default=Path("../AtomBit-OMC-s/model_epoch_15.pt"),
    )
    parser.add_argument(
        "--atombit-e0",
        type=Path,
        default=Path("../AtomBit-OMC-s/meta_e0_data_OMC_r6_single.pt"),
    )
    parser.add_argument("--atombit-cutoff", type=float, default=6.0)
    parser.add_argument(
        "--mace-checkpoint",
        type=Path,
        default=Path.home() / ".cache/mace/MACE-OFF23_small.model",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        parser.error("warmup-steps must be nonnegative and measured-steps positive")
    if args.batch_size <= 0 or args.batch_size > 256:
        parser.error("batch-size must be between 1 and 256")

    torch.set_num_threads(1)
    args.task = "nve"
    args.resolved_neighbor_backends = [args.neighbor_backend]
    bundle = build_model_bundle(args)

    pool_size = 256 if args.batch_size > 32 else 32
    manifest_path = args.manifest_dir / (
        f"MD-NVE-H{args.atom_count}-R{pool_size}-v1.json"
    )
    manifest = read_workload_manifest(manifest_path)
    systems = materialize_workload(manifest, args.dataset_dir)[: args.batch_size]
    if len(systems) != args.batch_size:
        raise RuntimeError("signed workload does not contain the requested batch")

    warmup = _run(
        systems,
        bundle.native,
        ensemble=args.ensemble,
        steps=args.warmup_steps,
        seed=20260727,
    )
    _synchronize(bundle.device)
    del warmup
    gc.collect()
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(bundle.device)

    setup_started = time.perf_counter()
    state = bundle.native.create_state(systems)
    initialize_maxwell_boltzmann(
        state,
        300.0,
        seed=20260729,
        remove_com=True,
        force_exact_temperature=True,
    )
    _synchronize(bundle.device)
    setup_seconds = time.perf_counter() - setup_started
    rebuilds_before = state.neighbor_rebuild_count

    _synchronize(bundle.device)
    started = time.perf_counter()
    if args.ensemble == "nvt":
        result = batched_langevin_baoab(
            state,
            bundle.native,
            timestep_fs=0.5,
            n_steps=args.measured_steps,
            temperature_K=300.0,
            friction_per_fs=0.01,
            seed=20260730,
        )
    else:
        result = batched_isotropic_mtk(
            state,
            bundle.native,
            timestep_fs=0.5,
            n_steps=args.measured_steps,
            temperature_K=300.0,
            pressure_eV_per_A3=0.0,
            thermostat_damping_fs=50.0,
            barostat_damping_fs=500.0,
        )
    _synchronize(bundle.device)
    wall_seconds = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated(bundle.device) / 1e9
        if bundle.device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(bundle.device) / 1e9
        if bundle.device.type == "cuda"
        else None
    )

    replica_steps = args.batch_size * args.measured_steps
    atom_steps = sum(len(atoms) for atoms in systems) * args.measured_steps
    output = {
        "schema_version": 1,
        "status": "passed",
        "model": bundle.name,
        "model_checkpoint_sha256": bundle.checkpoint_sha256,
        "ensemble": args.ensemble,
        "algorithm": (
            "Langevin BAOAB"
            if args.ensemble == "nvt"
            else "isotropic MTK Nose-Hoover chain"
        ),
        "workload_manifest": str(manifest_path),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "atom_count": args.atom_count,
        "batch_size": args.batch_size,
        "total_atoms": sum(len(atoms) for atoms in systems),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "setup_seconds_excluded": setup_seconds,
        "wall_seconds": wall_seconds,
        "replica_steps_per_second": replica_steps / wall_seconds,
        "atom_steps_per_second": atom_steps / wall_seconds,
        "peak_allocated_GB": peak_allocated,
        "peak_reserved_GB": peak_reserved,
        "model_evaluations": result.model_evaluations,
        "graph_evaluations": result.graph_evaluations,
        "neighbor_rebuilds": state.neighbor_rebuild_count - rebuilds_before,
        "final_temperature_K": _tensor_list(result.temperature),
        "integrator_metadata": {
            key: _tensor_list(value) for key, value in result.metadata.items()
        },
        "settings": {
            "timestep_fs": 0.5,
            "temperature_K": 300.0,
            "friction_per_fs": 0.01 if args.ensemble == "nvt" else None,
            "pressure_eV_per_A3": 0.0 if args.ensemble == "npt" else None,
            "thermostat_damping_fs": 50.0 if args.ensemble == "npt" else None,
            "barostat_damping_fs": 500.0 if args.ensemble == "npt" else None,
            "skin_A": args.skin,
            "neighbor_backend": args.neighbor_backend,
            "dtype": str(bundle.dtype),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(bundle.device)
                if bundle.device.type == "cuda"
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
