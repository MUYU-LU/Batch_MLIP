#!/usr/bin/env python3
"""Execute one frozen OMC-CSP scheduler epoch split exactly once."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


EXPECTED_WORKLOADS = {
    "development": 23,
    "validation": 10,
    "test": 16,
}


def _selected_workloads(
    index: dict[str, Any],
    split: str,
) -> list[tuple[str, dict[str, Any]]]:
    matrix = index["scheduler_matrix"][split]
    if split == "test":
        all_pool_families = set(matrix["all_family_p64_p512_p2048"])
        extreme_pool_families = all_pool_families
    else:
        all_pool_families = set(matrix["all_family_p512"])
        extreme_pool_families = set(
            matrix["selected_intra_family_extreme_pool_families"]
        )
    selected = []
    for workload_id, item in index["workloads"].items():
        if item["scheduler_split"] != split:
            continue
        jobs = int(item["jobs"])
        scope = item["family_scope"]
        families = set(item["source_families"])
        include = (
            jobs == 512
            and (
                scope == "inter"
                or (scope == "intra" and families <= all_pool_families)
            )
        ) or (
            scope == "intra"
            and jobs in (64, 2048)
            and families <= extreme_pool_families
        )
        if include:
            selected.append((workload_id, item))
    selected.sort(key=lambda item: (int(item[1]["jobs"]), item[0]))
    expected = EXPECTED_WORKLOADS[split]
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} {split} workloads, found {len(selected)}"
        )
    return selected


def _gpus(pool_size: int) -> str:
    return {64: "0", 512: "0,1,2,3", 2048: "0,1,2,3,4,5,6"}[pool_size]


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": command,
        "external_process_wall_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--mps-session-config", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_WORKLOADS),
        default="development",
    )
    parser.add_argument(
        "--tail-recovery",
        choices=("none", "ase_bfgs"),
        default="none",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit-workloads",
        type=int,
        help="Run an initial split prefix for lifecycle verification.",
    )
    args = parser.parse_args()

    index = _load(args.index)
    selected = _selected_workloads(index, args.split)
    if args.limit_workloads is not None:
        if not 0 < args.limit_workloads <= len(selected):
            parser.error("--limit-workloads is outside the selected matrix")
        selected = selected[: args.limit_workloads]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "20260729",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(args.runner_root),
        }
    )
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
    ):
        environment.pop(name, None)
    summary: list[dict[str, Any]] = []
    for run_index, (workload_id, item) in enumerate(selected, start=1):
        pool_size = int(item["jobs"])
        gpus = _gpus(pool_size)
        manifest = args.workload_dir / "manifests" / f"{workload_id}.json"
        directory = args.output_dir / workload_id
        directory.mkdir(exist_ok=True)
        outputs = {
            "current_auto": directory / "current_auto.json",
            "ase_cuda_mps": directory / "ase_cuda_mps.json",
        }
        print(
            json.dumps(
                {
                    "run": run_index,
                    "total": len(selected),
                    "workload_id": workload_id,
                    "pool_size": pool_size,
                    "gpus": gpus,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        commands = {
            "current_auto": [
                str(args.python),
                str(args.runner_root / "benchmarks" / "benchmark_omc_csp_scheduler_auto.py"),
                "--manifest",
                str(manifest),
                "--dataset-dir",
                str(args.dataset_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--devices",
                gpus,
                "--output",
                str(outputs["current_auto"]),
                "--tail-recovery",
                args.tail_recovery,
            ],
            "ase_cuda_mps": [
                str(args.python),
                str(args.runner_root / "benchmarks" / "run_omc_csp_scheduler_mps.py"),
                "--manifest",
                str(manifest),
                "--dataset-dir",
                str(args.dataset_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--gpus",
                gpus,
                "--runtime-dir",
                str(directory / "mps_runtime"),
                "--output",
                str(outputs["ase_cuda_mps"]),
                "--mps-session-config",
                str(args.mps_session_config),
            ],
        }
        run_summary: dict[str, Any] = {
            "workload_id": workload_id,
            "workload_manifest_sha256": item["manifest_sha256"],
            "pool_size": pool_size,
            "gpus": gpus,
            "methods": {},
        }
        for method, command in commands.items():
            external = outputs[method].with_suffix(".external.json")
            if args.resume and outputs[method].exists() and external.exists():
                outcome = _load(external)
                outcome["resumed"] = True
            else:
                outcome = _run(
                    command,
                    environment=environment,
                    log_path=outputs[method].with_suffix(".log"),
                )
                external.write_text(json.dumps(outcome, indent=2) + "\n")
            if outcome["returncode"] != 0:
                raise RuntimeError(f"{method} failed for {workload_id}: {outcome}")
            run_summary["methods"][method] = outcome
        summary.append(run_summary)
        (args.output_dir / f"{args.split}_progress.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "status": "complete",
        "split": args.split,
        "workload_count": len(summary),
        "methods_per_workload": ["current_auto", "ase_cuda_mps_static_lpt"],
        "runs": summary,
    }
    (args.output_dir / f"{args.split}_complete.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
