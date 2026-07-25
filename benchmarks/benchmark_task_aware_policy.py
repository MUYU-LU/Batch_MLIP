#!/usr/bin/env python3
"""Benchmark the public task-aware relaxation policy on a signed workload."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_memory_planner import calibrate  # noqa: E402
from benchmark_mixed_scheduling import load_signed_systems  # noqa: E402
from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_manifest,
    load_production_model,
    sha256_file,
    synchronize,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    BatchPlanner,
    FrechetCellFilter,
    MACEBatchCalculator,
    OptimizationPilot,
    SystemProfile,
    TaskAwarePolicy,
    relax,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument("--optimizer", choices=("bfgs", "fire"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-budget-gib", type=float, default=68.0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-cost-ratio", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=Path("benchmarks/t2_fixed_samples.json"),
    )
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=Path("experiments/bfgs-active-refill/results.json"),
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
    max_steps = args.max_steps or (2_000 if args.optimizer == "fire" else 500)
    if args.memory_budget_gib <= 0.0 or args.max_batch_size <= 0:
        parser.error("memory budget and maximum batch size must be positive")

    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest, systems = load_signed_systems(
        args.workload_manifest,
        args.dataset_dir,
    )
    pilot = OptimizationPilot.from_dict(
        json.loads(args.pilot.read_text(encoding="utf-8"))
    )
    calibration_manifest = load_manifest(args.calibration_manifest, 32)
    cutoff = 6.0 if args.mlip == "atombit" else 5.0
    coefficients, calibration = calibrate(
        model=args.mlip,
        manifest=calibration_manifest,
        dataset_dir=args.dataset_dir,
        calibration_path=args.calibration_results,
        cutoff=cutoff,
    )
    planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=int(args.memory_budget_gib * 2**30),
        max_batch_size=args.max_batch_size,
        max_cost_ratio=args.max_cost_ratio,
    )
    edge_key = f"cutoff={cutoff:.3f}_skin=0.000"
    system_profiles = tuple(
        SystemProfile(
            index=index,
            atom_count=job.atom_count,
            edge_count=job.topology_edge_counts[edge_key],
            dof_squared=(3 * job.atom_count + 9) ** 2,
        )
        for index, job in enumerate(manifest.jobs)
    )

    device = torch.device(args.device)
    if args.mlip == "atombit":
        model, model_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=cutoff,
            skin=0.0,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
        model_info = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_metadata": model_metadata,
        }
    else:
        calculator = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=device,
            dtype=torch.float64,
            graph_mode="rebuild",
            skin=0.0,
            neighbor_backend="auto",
        )
        model_info = {"model": args.mace_model}
    calculator(
        calculator.create_state([systems[0]]),
        compute_stress=True,
    )
    synchronize(device)

    optimizer_options = {
        "cell_filter": FrechetCellFilter(),
        "fmax": args.fmax,
        "smax": None,
        "max_steps": max_steps,
        "max_step": 0.2,
        "callback_interval": max_steps + 1,
    }
    if args.optimizer == "bfgs":
        optimizer_options.update(
            {
                "alpha": 70.0,
                "optimizer_dtype": "float64",
                "linear_algebra_backend": "auto",
            }
        )
    else:
        optimizer_options.update({"dt_start": 0.1, "dt_max": 1.0})

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    result = relax(
        systems,
        calculator,
        optimizer=args.optimizer,
        scheduling="auto",
        planner=planner,
        pilot=pilot,
        policy=TaskAwarePolicy(),
        system_profiles=system_profiles,
        **optimizer_options,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    energies = result.evaluation.energy.detach().cpu().tolist()
    steps = result.converged_step.detach().cpu().tolist()
    converged = result.converged.detach().cpu().tolist()
    records = [
        {
            "system_id": job.system_id,
            "source_path": job.source_path,
            "atom_count": job.atom_count,
            "converged": bool(converged[index]),
            "steps": int(steps[index]),
            "energy_eV": float(energies[index]),
            "energy_eV_per_atom": float(energies[index]) / job.atom_count,
        }
        for index, job in enumerate(manifest.jobs)
    ]
    output = {
        "schema_version": 1,
        "status": "complete",
        "mlip": args.mlip,
        "optimizer": args.optimizer,
        "workload_id": manifest.workload_id,
        "workload_manifest": str(args.workload_manifest),
        "workload_manifest_sha256": manifest.manifest_sha256,
        "pilot": str(args.pilot),
        "pilot_sha256": sha256_file(args.pilot),
        "jobs": len(systems),
        "parameters": {
            "memory_budget_bytes": planner.memory_budget_bytes,
            "max_batch_size": args.max_batch_size,
            "max_cost_ratio": args.max_cost_ratio,
            "fmax_eV_per_A": args.fmax,
            "max_steps": max_steps,
            "deterministic_algorithms": args.deterministic,
        },
        "schedule": result.metadata["scheduling"],
        "wall_time_s": elapsed,
        "systems_per_second": len(systems) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "model_evaluations": result.model_evaluations,
        "graph_evaluations": result.graph_evaluations,
        "neighbor_rebuilds": result.state.neighbor_rebuild_count,
        "converged": sum(record["converged"] for record in records),
        "records": records,
        "calibration": calibration,
        "environment": environment_metadata(device),
        **model_info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seconds": elapsed,
                "converged": output["converged"],
                "schedule": output["schedule"]["batches"],
                "recommended_worker_mode": output["schedule"][
                    "recommended_worker_mode"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
