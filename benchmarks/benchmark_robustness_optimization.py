#!/usr/bin/env python3
"""Benchmark ASE or tensor-batched optimization on a signed workload."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_mace_variable_cell_scaling import (  # noqa: E402
    make_counting_ase_calculator as make_mace_ase_calculator,
)
from benchmark_mace_variable_cell_scaling import (  # noqa: E402
    run_ase as run_mace_ase,
)
from benchmark_mace_variable_cell_scaling import (  # noqa: E402
    run_batch as run_mace_batch,
)
from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    synchronize,
)
from benchmark_variable_cell_scaling import (  # noqa: E402
    run_ase as run_atombit_ase,
)
from benchmark_variable_cell_scaling import (  # noqa: E402
    run_batch as run_atombit_batch,
)

from batch_mlip import AtomBitBatchCalculator, MACEBatchCalculator  # noqa: E402
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _systems(
    manifest_path: Path,
    dataset_dir: Path,
    job_limit: int | None,
):
    manifest = read_workload_manifest(manifest_path)
    jobs = manifest.jobs if job_limit is None else manifest.jobs[:job_limit]
    systems = []
    for job in jobs:
        atoms = read(dataset_dir / job.source_path, index=job.frame_index)
        atoms.info["benchmark_source"] = job.system_id
        atoms.info["benchmark_source_path"] = job.source_path
        atoms.info["workload_system_id"] = job.system_id
        systems.append(atoms)
    return manifest, systems


def _timed(fn, *, device: torch.device) -> tuple[Any, float]:
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = fn()
    synchronize(device)
    return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument(
        "--method", choices=("ase", "active", "refill"), required=True
    )
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--refill-storage",
        choices=("repack", "slots"),
        default="repack",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--job-limit",
        type=int,
        help="Run the first N signed jobs; omit to run the complete workload.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument(
        "--linear-algebra-backend",
        choices=("auto", "cholesky", "grouped", "serial"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/"
            "smooth_rms_finetune/"
            "AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
        ),
    )
    parser.add_argument("--mace-model", default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.cpu_threads <= 0:
        parser.error("CPU thread count must be positive")
    if args.job_limit is not None and args.job_limit <= 0:
        parser.error("job limit must be positive")

    manifest, systems = _systems(
        args.workload_manifest,
        args.dataset_dir,
        args.job_limit,
    )
    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    common = {
        "systems": systems,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "dt_start": 0.1,
        "dt_max": 1.0,
        "max_step": 0.2,
        "optimizer_name": "bfgs",
        "alpha": 70.0,
    }
    if args.mlip == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        batch_calculator = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=args.skin,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )

        def execute():
            if args.method == "ase":
                return run_atombit_ase(
                    model,
                    device=device,
                    cutoff=6.0,
                    optimizer_dtype="float64",
                    model_dtype=torch.float32,
                    **common,
                )
            return run_atombit_batch(
                model,
                batch_size=args.batch_size,
                active_compaction=True,
                device=device,
                cutoff=6.0,
                skin=args.skin,
                optimizer_dtype="float64",
                model_dtype=torch.float32,
                neighbor_backend="auto",
                refill=args.method == "refill",
                refill_policy="immediate",
                refill_storage=args.refill_storage,
                refill_min_chunk=1 if args.method == "refill" else None,
                linear_algebra_backend=args.linear_algebra_backend,
                **common,
            )

        model_info = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_metadata": model_metadata,
        }
    else:
        batch_calculator = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=device,
            dtype=torch.float64,
            graph_mode="rebuild",
            skin=args.skin,
            neighbor_backend="auto",
        )
        ase_calculator = make_mace_ase_calculator(
            batch_calculator.model, device=device
        )

        def execute():
            if args.method == "ase":
                return run_mace_ase(ase_calculator, **common)
            return run_mace_batch(
                batch_calculator,
                batch_size=args.batch_size,
                active_compaction=True,
                refill=args.method == "refill",
                refill_policy="immediate",
                refill_storage=args.refill_storage,
                refill_min_chunk=1 if args.method == "refill" else None,
                linear_algebra_backend=args.linear_algebra_backend,
                **common,
            )

        model_info = {"model": args.mace_model}

    batch_calculator(
        batch_calculator.create_state([systems[0]]),
        compute_stress=True,
    )
    try:
        output, elapsed = _timed(execute, device=device)
        status = "passed"
        error = None
    except torch.OutOfMemoryError as exc:
        output = {}
        elapsed = None
        status = "oom"
        error = str(exc)
    result = {
        "schema_version": 1,
        "status": status,
        "error": error,
        "mlip": args.mlip,
        "method": args.method,
        "optimizer": "bfgs",
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.workload_manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "jobs": len(systems),
        "workload_jobs": len(manifest.jobs),
        "job_limit": args.job_limit,
        "batch_size": None if args.method == "ase" else args.batch_size,
        "refill_storage": (
            args.refill_storage if args.method == "refill" else None
        ),
        "fmax_eV_per_A": args.fmax,
        "max_steps": args.max_steps,
        "linear_algebra_backend": args.linear_algebra_backend,
        "deterministic_algorithms": args.deterministic,
        "cpu_threads": args.cpu_threads,
        "timing_seconds": elapsed,
        "systems_per_second": (
            None if elapsed is None else len(systems) / elapsed
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "environment": environment_metadata(device),
        **model_info,
        **output,
        "converged": sum(
            bool(record["converged"])
            for record in output.get("records", ())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "seconds": elapsed,
                "systems_per_second": result["systems_per_second"],
                "peak_allocated_GiB": result["peak_allocated_bytes"] / 2**30,
                "peak_reserved_GiB": result["peak_reserved_bytes"] / 2**30,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
