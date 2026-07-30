#!/usr/bin/env python3
"""Run the frozen no-probe OMC-CSP capacity-integration matrix once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

WORKLOAD_IDS = (
    "OPT-OMC-SCHED-E1-TEST-ROF-A-P2048-INTRA-WIDE-v2",
    "OPT-OMC-SCHED-E1-TEST-ROF-C-P2048-INTRA-WIDE-v2",
    "OPT-OMC-SCHED-E1-TEST-XULDUD-P2048-INTRA-WIDE-v2",
)


def _loader_process_count(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "loader process count must be a positive integer or 'auto'"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "loader process count must be a positive integer or 'auto'"
        )
    return parsed


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


def _assert_contract_and_identity(
    offline: dict[str, Any],
    probe_backed: dict[str, Any],
) -> None:
    if offline["workload_manifest_sha256"] != probe_backed[
        "workload_manifest_sha256"
    ]:
        raise RuntimeError("offline and probe-backed manifest hashes differ")
    if offline["pool_size"] != probe_backed["pool_size"]:
        raise RuntimeError("offline and probe-backed pool sizes differ")
    if [record["source"] for record in offline["records"]] != [
        record["source"] for record in probe_backed["records"]
    ]:
        raise RuntimeError("offline and probe-backed source order differs")
    ignored = {"structure_materialization", "planning_profile_sha256"}
    for key, value in probe_backed["contract"].items():
        if key in ignored:
            continue
        if offline["contract"].get(key) != value:
            raise RuntimeError(f"offline contract field {key!r} differs")
    scheduling = offline["scheduling"]
    if scheduling["capacity_planning"]["mode"] != "offline_hardware_model":
        raise RuntimeError("offline hardware-capacity policy did not match")
    if (
        scheduling["probe"]["system_count"] != 0
        or scheduling["probe"]["model_forward_count"] != 0
    ):
        raise RuntimeError("offline hardware-capacity run performed a probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--planning-profile-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--epoch2-source-results", type=Path, required=True)
    parser.add_argument("--epoch1-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6")
    parser.add_argument(
        "--manifest-loader-processes",
        type=_loader_process_count,
        default="auto",
    )
    parser.add_argument("--workload-id", action="append")
    parser.add_argument(
        "--tail-recovery",
        choices=("none", "ase_bfgs"),
        default="ase_bfgs",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-workloads", type=int)
    args = parser.parse_args()
    workload_ids = (
        tuple(args.workload_id)
        if args.workload_id is not None
        else WORKLOAD_IDS
    )
    if not workload_ids or len(set(workload_ids)) != len(workload_ids):
        parser.error("--workload-id values must be non-empty and unique")
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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
        probe_path = (
            args.epoch2_source_results
            / workload_id
            / "source_backed_auto.json"
        )
        mps_path = (
            args.epoch1_results / workload_id / "ase_cuda_mps.json"
        )
        for path in (manifest, planning_profile, probe_path, mps_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        directory = args.output_dir / workload_id
        directory.mkdir(exist_ok=True)
        output = directory / "offline_capacity_auto.json"
        external = directory / "offline_capacity_auto.external.json"
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
            args.tail_recovery,
            "--manifest-loader-processes",
            str(args.manifest_loader_processes),
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
                log_path=directory / "offline_capacity_auto.log",
            )
            external.write_text(
                json.dumps(outcome, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if outcome["returncode"] != 0:
            raise RuntimeError(
                f"offline capacity run failed for {workload_id}: {outcome}"
            )
        offline = _load(output)
        probe_backed = _load(probe_path)
        _assert_contract_and_identity(offline, probe_backed)
        summary.append(
            {
                "workload_id": workload_id,
                "workload_manifest_sha256": (
                    offline["workload_manifest_sha256"]
                ),
                "offline_capacity": {
                    **outcome,
                    "output": str(output),
                    "output_sha256": _sha256(output),
                },
                "frozen_references": {
                    "probe_backed_source": {
                        "path": str(probe_path),
                        "sha256": _sha256(probe_path),
                    },
                    "ase_cuda_mps": {
                        "path": str(mps_path),
                        "sha256": _sha256(mps_path),
                    },
                },
                "contract_and_identity_equal": True,
            }
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    completion = {
        "schema_version": 1,
        "status": "complete",
        "matrix": "capacity_integration_nonfit_p2048",
        "timing_repeats": 1,
        "manifest_loader_processes": args.manifest_loader_processes,
        "workloads": summary,
    }
    completion["completion_sha256"] = _sha256(
        args.output_dir / "progress.json"
    )
    (args.output_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, sort_keys=True))


if __name__ == "__main__":
    main()
