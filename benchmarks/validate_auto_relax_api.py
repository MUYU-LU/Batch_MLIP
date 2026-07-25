#!/usr/bin/env python3
"""Validate public automatic relaxation scheduling with production MLIPs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_memory_planner import calibrate  # noqa: E402
from benchmark_mixed_scheduling import load_signed_systems  # noqa: E402
from benchmark_production import (  # noqa: E402
    load_manifest,
    load_production_model,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    BatchPlanner,
    FrechetCellFilter,
    MACEBatchCalculator,
    relax,
)


def differences(
    reference,
    candidate,
) -> dict[str, float]:
    return {
        "energy_eV": float(
            torch.max(
                torch.abs(
                    reference.evaluation.energy
                    - candidate.evaluation.energy
                )
            ).item()
        ),
        "force_eV_per_A": float(
            torch.max(
                torch.abs(
                    reference.evaluation.forces
                    - candidate.evaluation.forces
                )
            ).item()
        ),
        "position_A": float(
            torch.max(
                torch.abs(reference.state.positions - candidate.state.positions)
            ).item()
        ),
        "cell_A": float(
            torch.max(
                torch.abs(reference.state.cells - candidate.state.cells)
            ).item()
        ),
    }


def run_relaxation(
    systems: list[Any],
    calculator,
    *,
    planner: BatchPlanner | None = None,
):
    return relax(
        systems,
        calculator,
        optimizer="bfgs",
        scheduling="auto" if planner is not None else "single_batch",
        planner=planner,
        cell_filter=FrechetCellFilter(),
        active_compaction=True,
        fmax=1e-30,
        max_steps=2,
        max_step=0.2,
        alpha=70.0,
        optimizer_dtype="float64",
        linear_algebra_backend="auto",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--job-limit", type=int, default=4)
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        default=Path(
            "runs/robustness/workloads/manifests/"
            "OPT-RB-CROSS-MIX-R192-v1.json"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument(
        "--calibration-dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
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

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, systems = load_signed_systems(
        args.workload_manifest,
        args.dataset_dir,
    )
    systems = systems[: args.job_limit]
    device = torch.device(args.device)
    cutoff = 6.0 if args.mlip == "atombit" else 5.0
    skin = 0.5 if args.mlip == "atombit" else 0.0
    if args.mlip == "atombit":
        model, _ = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=cutoff,
            skin=skin,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
    else:
        calculator = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=device,
            dtype=torch.float64,
            graph_mode="rebuild",
            skin=skin,
            neighbor_backend="auto",
        )

    calibration_manifest = load_manifest(args.calibration_manifest, 32)
    coefficients, _ = calibrate(
        model=args.mlip,
        manifest=calibration_manifest,
        dataset_dir=args.calibration_dataset_dir,
        calibration_path=args.calibration_results,
        cutoff=cutoff,
    )
    whole_planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=64 * 1024**3,
        max_batch_size=len(systems),
        max_cost_ratio=100.0,
    )
    fallback_planner = BatchPlanner(
        coefficients,
        memory_budget_bytes=64 * 1024**3,
        max_batch_size=2,
        max_cost_ratio=100.0,
    )
    direct = run_relaxation(systems, calculator)
    automatic = run_relaxation(
        systems,
        calculator,
        planner=whole_planner,
    )
    fallback = run_relaxation(
        systems,
        calculator,
        planner=fallback_planner,
    )
    result = {
        "schema_version": 1,
        "mlip": args.mlip,
        "jobs": len(systems),
        "steps": 2,
        "whole_schedule": automatic.metadata["scheduling"],
        "fallback_schedule": fallback.metadata["scheduling"],
        "whole_vs_direct_maximum_absolute_difference": differences(
            direct,
            automatic,
        ),
        "fallback_vs_direct_maximum_absolute_difference": differences(
            direct,
            fallback,
        ),
        "direct_converged": int(direct.converged.sum().item()),
        "whole_converged": int(automatic.converged.sum().item()),
        "fallback_converged": int(fallback.converged.sum().item()),
        "fallback_systems_returned": fallback.state.n_systems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
