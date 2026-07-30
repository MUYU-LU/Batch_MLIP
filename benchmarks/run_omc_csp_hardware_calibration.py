#!/usr/bin/env python3
"""Run the signed OMC-CSP AtomBit/BFGS hardware calibration matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

FIT_FAMILIES = {
    "GUFJOG": "INTRA-NARROW",
    "XATMOV": "INTRA-NARROW",
    "OBEQIX": "INTRA-NARROW",
    "ROF-B": "INTRA-NARROW",
}
VALIDATION_FAMILIES = {
    "AXOSOW": "INTRA-NARROW",
    "BOQQUT": "INTRA-WIDE",
    "ROF-A": "INTRA-WIDE",
}
CAPACITY_POINTS = (
    ("fit", "GUFJOG", "INTRA-NARROW", 256),
    ("fit", "XATMOV", "INTRA-NARROW", 192),
    ("fit", "OBEQIX", "INTRA-NARROW", 128),
    ("fit", "ROF-B", "INTRA-NARROW", 128),
    ("validation", "AXOSOW", "INTRA-NARROW", 256),
    ("validation", "BOQQUT", "INTRA-WIDE", 128),
    ("validation", "ROF-A", "INTRA-WIDE", 256),
    ("fit", "GUFJOG", "INTRA-NARROW", 512),
    ("fit", "XATMOV", "INTRA-NARROW", 320),
    ("validation", "AXOSOW", "INTRA-NARROW", 384),
    ("validation", "BOQQUT", "INTRA-WIDE", 256),
)


def _points() -> list[dict[str, object]]:
    points = []
    for split, families, sizes in (
        ("fit", FIT_FAMILIES, (8, 32, 64)),
        ("validation", VALIDATION_FAMILIES, (16, 64)),
    ):
        for family, mixing in families.items():
            workload_id = f"OPT-OMC-{family}-P64-{mixing}-v1"
            for batch_size in sizes:
                points.append(
                    {
                        "split": split,
                        "family": family,
                        "mixing": mixing,
                        "batch_size": batch_size,
                        "pool_size": 64,
                        "workload_id": workload_id,
                        "observation_id": (
                            f"{split}-{family}-B{batch_size}"
                        ),
                    }
                )
    for split, family, mixing, batch_size in CAPACITY_POINTS:
        workload_id = f"OPT-OMC-{family}-P512-{mixing}-v1"
        points.append(
            {
                "split": split,
                "family": family,
                "mixing": mixing,
                "batch_size": batch_size,
                "pool_size": 512,
                "workload_id": workload_id,
                "observation_id": f"{split}-{family}-B{batch_size}",
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--matrix",
        type=Path,
        help=(
            "Optional explicit point matrix. Relative index-selection paths "
            "are resolved from the matrix directory."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain existing complete point outputs and run missing points.",
    )
    args = parser.parse_args()

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        parser.error("--devices must not be empty")
    max_parallel = args.max_parallel or len(devices)
    if max_parallel <= 0 or args.max_steps <= 0:
        parser.error("parallelism and maximum steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_matrix = None
    if args.matrix is None:
        points = _points()
    else:
        source_matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        points = source_matrix["points"]
    environment = {
        **os.environ,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "20260729",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    active: list[tuple[dict[str, object], subprocess.Popen, object]] = []
    failures = []
    completed = []

    def wait_one() -> None:
        point, process, log = active.pop(0)
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append(
                {
                    "observation_id": point["observation_id"],
                    "return_code": return_code,
                }
            )
        else:
            completed.append(point["observation_id"])

    for point_index, point in enumerate(points):
        observation_id = str(point["observation_id"])
        output = args.output_dir / f"{observation_id}.json"
        if args.resume and output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                completed.append(observation_id)
                continue
        while len(active) >= max_parallel:
            wait_one()
        device = devices[point_index % len(devices)]
        log_path = args.output_dir / f"{observation_id}.log"
        manifest = (
            args.workloads
            / "manifests"
            / f"{point['workload_id']}.json"
        )
        command = [
            str(args.python),
            "benchmarks/benchmark_omc_csp_hardware_point.py",
            "--device",
            f"cuda:{device}",
            "--max-steps",
            str(args.max_steps),
            "--manifest",
            str(manifest),
            "--dataset-dir",
            str(args.dataset_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(output),
        ]
        indices_json = point.get("indices_json")
        if indices_json is None:
            command.extend(["--batch-size", str(point["batch_size"])])
        else:
            indices_path = Path(str(indices_json))
            if not indices_path.is_absolute():
                indices_path = args.matrix.parent / indices_path
            command.extend(["--indices-json", str(indices_path)])
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.runner_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        active.append((point, process, log))
    while active:
        wait_one()

    execution_contract = {
        "mlip": "AtomBit-smooth-rms-fp32",
        "optimizer": "BatchedBFGS",
        "optimizer_dtype": "torch.float64",
        "cell_filter": "FrechetCellFilter",
        "cutoff_A": 6.0,
        "skin_A": 0.5,
        "force_mode": "autograd",
        "neighbor_backend": "auto",
        "max_steps": args.max_steps,
        "warm_executions": 1,
        "measured_executions": 1,
        "scheduling": "single_batch",
        "cuda_allocator": "expandable_segments",
        "deterministic": True,
    }
    if source_matrix is not None:
        expected = source_matrix.get("execution_contract")
        if expected is not None and expected != execution_contract:
            raise ValueError(
                "explicit matrix execution contract differs from runner"
            )
    matrix = {
        "schema_version": 1,
        "status": "failed" if failures else "complete",
        "execution_contract": execution_contract,
        "devices": devices,
        "points": points,
        "completed": completed,
        "failures": failures,
    }
    (args.output_dir / "matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(matrix, sort_keys=True))
    if failures:
        raise RuntimeError(f"calibration point failures: {failures}")


if __name__ == "__main__":
    main()
