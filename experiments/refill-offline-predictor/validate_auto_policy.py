#!/usr/bin/env python3
"""Run one public-API auto-scheduler validation against matrix evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_production import load_production_model, synchronize  # noqa: E402
from benchmark_robustness_optimization import _systems  # noqa: E402

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    AutoSchedulerConfig,
    FrechetCellFilter,
    relax,
)


def _reference_energy_error(
    result: Any,
    reference_path: Path | None,
) -> float | None:
    if reference_path is None:
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = [record["energy_eV"] for record in reference["records"]]
    actual = result.evaluation.energy.detach().cpu().tolist()
    counts = result.state.counts.detach().cpu().tolist()
    return max(
        abs(current - target) * 1000.0 / count
        for current, target, count in zip(actual, expected, counts, strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--resident-capacity", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    _, systems = _systems(args.manifest, args.dataset_dir, None)
    model, _ = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()
    calculator = AtomBitBatchCalculator(
        model,
        cutoff=6.0,
        skin=0.5,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
        neighbor_backend="auto",
    )

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = relax(
        systems,
        calculator,
        optimizer="bfgs",
        scheduling="auto",
        auto_config=AutoSchedulerConfig(
            max_batch_size=args.resident_capacity,
        ),
        cell_filter=FrechetCellFilter(),
        fmax=0.05,
        smax=None,
        max_steps=500,
        max_step=0.2,
        alpha=70.0,
        optimizer_dtype="float64",
        linear_algebra_backend="auto",
    )
    synchronize(device)
    elapsed = time.perf_counter() - started
    scheduling = result.metadata["scheduling"]
    predictions = [
        batch.get("refill_prediction")
        for batch in scheduling["batches"]
        if batch.get("refill_prediction") is not None
    ]
    if not scheduling["active_refill"]:
        raise RuntimeError(f"offline policy did not select refill: {predictions}")

    payload = {
        "schema_version": 1,
        "status": "passed",
        "pool_size": len(systems),
        "resident_capacity": args.resident_capacity,
        "all_converged": bool(result.converged.all()),
        "converged": int(result.converged.sum().item()),
        "elapsed_seconds_including_profile_and_probe": elapsed,
        "systems_per_second": len(systems) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "maximum_reference_energy_difference_meV_per_atom": (
            _reference_energy_error(result, args.reference_json)
        ),
        "predictions": predictions,
        "scheduling": scheduling,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(
                device
            ).total_memory,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
