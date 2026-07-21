#!/usr/bin/env python3
"""Measure one isolated resident-batch capacity point for full MLIP evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_controlled_matrix import (  # noqa: E402
    _observed_neighbor_backends,
    _peak_memory,
    _reset_peak,
    _synchronize,
    build_model_bundle,
)
from benchmark_production import environment_metadata, write_result  # noqa: E402

from batch_mlip import RuntimeProfiler, evaluate  # noqa: E402
from batch_mlip.profiling import runtime_profile_registry_fields  # noqa: E402
from batch_mlip.workloads import materialize_workload, read_workload_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("atombit", "mace"), required=True)
    parser.add_argument("--distribution", choices=("H46", "H276"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--neighbor-backend", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("benchmarks/workloads/manifests")
    )
    parser.add_argument("--code-commit", required=True)
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
    parser.add_argument("--energy-per-atom-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("batch-size and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    args.task = "eval"
    args.skin = 0.0
    args.resolved_neighbor_backends = [args.neighbor_backend]

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "study_id": "eval-resident-batch-capacity",
        "code_commit": args.code_commit,
        "requested_model": args.model,
        "distribution": args.distribution,
        "resident_batch_size": args.batch_size,
        "neighbor_backend": args.neighbor_backend,
        "measurement_repeats": 1,
    }
    write_result(args.output, result)

    try:
        bundle = build_model_bundle(args)
        manifest = read_workload_manifest(
            args.manifest_dir / f"EVAL-{args.distribution}-R32-v1.json"
        )
        base = materialize_workload(manifest, args.dataset_dir)
        systems = [base[index % len(base)].copy() for index in range(args.batch_size)]
        for index, atoms in enumerate(systems):
            atoms.info["capacity_batch_index"] = index

        # Initialize both generic model state and shape-dependent allocator paths.
        evaluate([systems[0]], bundle.native)
        for _ in range(2):
            evaluate(systems, bundle.native)
        _synchronize(bundle.device)
        _reset_peak(bundle.device)

        with RuntimeProfiler(device=bundle.device) as profiler:
            _synchronize(bundle.device)
            started = time.perf_counter()
            output = evaluate(systems, bundle.native)
            _synchronize(bundle.device)
            wall_time = time.perf_counter() - started
        peak_allocated, peak_reserved = _peak_memory(bundle.device)
        profile = profiler.summary()

        # Validate each unique source once; repeated jobs have identical inputs.
        max_energy_error = 0.0
        max_force_error = 0.0
        validation_count = min(len(base), args.batch_size)
        for index in range(validation_count):
            single = evaluate([base[index]], bundle.native).structures[0]
            batched = output.structures[index]
            energy_error = abs(
                float(single.get_potential_energy()) - float(batched.get_potential_energy())
            ) / len(single)
            force_error = float(
                np.max(np.abs(single.get_forces() - batched.get_forces()), initial=0.0)
            )
            max_energy_error = max(max_energy_error, energy_error)
            max_force_error = max(max_force_error, force_error)

        gpu_memory = float(torch.cuda.get_device_properties(bundle.device).total_memory) / 1e9
        result.update(
            {
                "status": "passed",
                "model": bundle.name,
                "model_checkpoint_sha256": bundle.checkpoint_sha256,
                "source_workload_id": manifest.workload_id,
                "source_manifest_sha256": manifest.manifest_sha256,
                "pool_expansion": "cyclic repetition of frozen R32 manifest order",
                "atom_count": len(systems[0]),
                "total_atoms": sum(len(atoms) for atoms in systems),
                "wall_time_s": wall_time,
                "structures_per_s": args.batch_size / wall_time,
                "atoms_per_s": sum(len(atoms) for atoms in systems) / wall_time,
                "peak_allocated_GB": peak_allocated,
                "peak_reserved_GB": peak_reserved,
                "gpu_memory_GB": gpu_memory,
                "allocated_memory_fraction": peak_allocated / gpu_memory,
                "reserved_memory_fraction": peak_reserved / gpu_memory,
                "runtime_profile": profile,
                "runtime_registry_fields": runtime_profile_registry_fields(profile),
                "observed_neighbor_backends": _observed_neighbor_backends(profile),
                "validation": {
                    "single_structure_checks": validation_count,
                    "max_energy_error_eV_per_atom": max_energy_error,
                    "max_force_error_eV_per_A": max_force_error,
                    "energy_per_atom_atol": args.energy_per_atom_atol,
                    "force_atol": args.force_atol,
                    "passed": max_energy_error <= args.energy_per_atom_atol
                    and max_force_error <= args.force_atol,
                },
                "environment": environment_metadata(bundle.device),
            }
        )
        if not result["validation"]["passed"]:  # type: ignore[index]
            result["status"] = "validation_failed"
    except torch.cuda.OutOfMemoryError as error:
        result.update({"status": "oom", "error": str(error)})
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        result.update(
            {
                "status": "oom" if "out of memory" in message.lower() else "error",
                "error": message,
            }
        )
    finally:
        write_result(args.output, result)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
