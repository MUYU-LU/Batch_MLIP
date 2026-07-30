#!/usr/bin/env python3
"""Run the frozen P2048 source-backed OMC-CSP causal ablation once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

CAUSAL_WORKLOAD_IDS = (
    "OPT-OMC-SCHED-E1-TEST-JAYDUI-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-TEST-OBEQIX-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2",
    "OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2",
    "OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2",
)
VALIDATION_WORKLOAD_IDS = (
    "OPT-OMC-SCHED-E1-DEVELOPMENT-BOQWIN-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-DEVELOPMENT-GUFJOG-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-DEVELOPMENT-KONTIQ-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-DEVELOPMENT-ROF-B-P2048-INTRA-NARROW-v2",
    "OPT-OMC-SCHED-E1-DEVELOPMENT-XAFPAY-P2048-INTRA-WIDE-v2",
)
PLAN_KEYS = (
    "probe",
    "parallel_chunk_policy",
    "resident_plan_chunk_count",
    "execution_chunk_count",
    "resident_plan_chunks",
    "planned_chunks",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _assert_frozen_equivalence(
    source_backed: dict[str, Any],
    eager: dict[str, Any],
) -> None:
    if source_backed["workload_manifest_sha256"] != eager[
        "workload_manifest_sha256"
    ]:
        raise RuntimeError("source-backed and eager manifest hashes differ")
    if source_backed["pool_size"] != eager["pool_size"]:
        raise RuntimeError("source-backed and eager pool sizes differ")
    source_records = source_backed["records"]
    eager_records = eager["records"]
    if [record["source"] for record in source_records] != [
        record["source"] for record in eager_records
    ]:
        raise RuntimeError("source-backed and eager source order differs")
    source_schedule = source_backed["scheduling"]
    eager_schedule = eager["scheduling"]
    for key in PLAN_KEYS:
        if source_schedule[key] != eager_schedule[key]:
            raise RuntimeError(f"source-backed plan field {key!r} differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--planning-profile-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--epoch1-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument(
        "--matrix",
        choices=("causal", "validation"),
        default="causal",
    )
    parser.add_argument(
        "--tail-recovery",
        choices=("none", "ase_bfgs"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-workloads", type=int)
    args = parser.parse_args()

    workload_ids = (
        CAUSAL_WORKLOAD_IDS
        if args.matrix == "causal"
        else VALIDATION_WORKLOAD_IDS
    )
    tail_recovery = args.tail_recovery or (
        "ase_bfgs" if args.matrix == "causal" else "none"
    )
    if args.limit_workloads is not None:
        if not 0 < args.limit_workloads <= len(workload_ids):
            parser.error("--limit-workloads is outside the frozen matrix")
        workload_ids = workload_ids[: args.limit_workloads]
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

    summary = []
    for run_index, workload_id in enumerate(workload_ids, start=1):
        print(
            json.dumps(
                {
                    "run": run_index,
                    "total": len(workload_ids),
                    "workload_id": workload_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        manifest = (
            args.workload_dir / "manifests" / f"{workload_id}.json"
        )
        planning_profile = (
            args.planning_profile_dir / f"{workload_id}.json"
        )
        eager_path = (
            args.epoch1_results / workload_id / "current_auto.json"
        )
        mps_path = (
            args.epoch1_results / workload_id / "ase_cuda_mps.json"
        )
        for path in (manifest, planning_profile, eager_path, mps_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        directory = args.output_dir / workload_id
        directory.mkdir(exist_ok=True)
        output = directory / "source_backed_auto.json"
        external = directory / "source_backed_auto.external.json"
        command = [
            str(args.python),
            str(
                args.runner_root
                / "benchmarks"
                / "benchmark_omc_csp_scheduler_auto.py"
            ),
            "--manifest",
            str(manifest),
            "--planning-profile",
            str(planning_profile),
            "--dataset-dir",
            str(args.dataset_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--devices",
            args.devices,
            "--materialization",
            "manifest_lazy",
            "--tail-recovery",
            tail_recovery,
            "--output",
            str(output),
        ]
        if args.resume and output.exists() and external.exists():
            outcome = _load(external)
            outcome["resumed"] = True
        else:
            outcome = _run(
                command,
                environment=environment,
                log_path=directory / "source_backed_auto.log",
            )
            external.write_text(
                json.dumps(outcome, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if outcome["returncode"] != 0:
            raise RuntimeError(
                f"source-backed run failed for {workload_id}: {outcome}"
            )
        source_backed = _load(output)
        eager = _load(eager_path)
        _assert_frozen_equivalence(source_backed, eager)
        summary.append(
            {
                "workload_id": workload_id,
                "workload_manifest_sha256": (
                    source_backed["workload_manifest_sha256"]
                ),
                "source_backed": {
                    **outcome,
                    "output": str(output),
                    "output_sha256": _sha256(output),
                },
                "frozen_references": {
                    "eager": {
                        "path": str(eager_path),
                        "sha256": _sha256(eager_path),
                    },
                    "ase_cuda_mps": {
                        "path": str(mps_path),
                        "sha256": _sha256(mps_path),
                    },
                },
                "plan_signature_equal": True,
            }
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    completion = {
        "schema_version": 1,
        "status": "complete",
        "experiment": (
            f"omc_csp_scheduler_epoch2_source_backed_{args.matrix}"
        ),
        "matrix": args.matrix,
        "tail_recovery": tail_recovery,
        "workload_count": len(summary),
        "timing_repeats": 1,
        "devices": args.devices,
        "workloads": summary,
    }
    completion_path = args.output_dir / "completion.json"
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "completion": str(completion_path),
                "status": "complete",
                "workloads": len(summary),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
