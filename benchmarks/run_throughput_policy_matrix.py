#!/usr/bin/env python3
"""Run the controlled offline throughput-policy matrix across visible GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATOMBIT_PYTHON = Path("/public/home/lmy/.conda/envs/lmy/bin/python")
MACE_PYTHON = Path("/public/home/lmy/.conda/envs/MACE_clean/bin/python")


@dataclass(frozen=True)
class MatrixTask:
    split: str
    workload_id: str
    manifest: Path
    dataset_dir: Path
    mlip: str
    optimizer: str
    max_steps: int
    capacity: int
    output: Path

    @property
    def task_id(self) -> str:
        return f"{self.split}-{self.mlip}-{self.optimizer}-{self.workload_id}-B{self.capacity}"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = _load(path)
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("status") in ("passed", "oom")


def _tasks(
    matrix_path: Path,
    raw_dir: Path,
    *,
    split: str,
    capacities: tuple[int, ...] | None,
    resume: bool,
) -> list[MatrixTask]:
    matrix = _load(matrix_path)
    selected_capacities = capacities or tuple(matrix["capacities"])
    tasks = []
    for workload in matrix["workloads"]:
        if split != "all" and workload["split"] != split:
            continue
        for mlip in matrix["mlips"]:
            for optimizer, optimizer_config in matrix["optimizers"].items():
                for capacity in selected_capacities:
                    evidence_prefix = workload.get(
                        "evidence_prefix",
                        workload["split"],
                    )
                    output = raw_dir / (
                        f"{evidence_prefix}_{mlip}_{optimizer}_"
                        f"{workload['workload_id']}_B{capacity}.json"
                    )
                    task = MatrixTask(
                        split=workload["split"],
                        workload_id=workload["workload_id"],
                        manifest=ROOT / workload["manifest"],
                        dataset_dir=Path(workload["dataset_dir"]),
                        mlip=mlip,
                        optimizer=optimizer,
                        max_steps=int(optimizer_config["max_steps"]),
                        capacity=capacity,
                        output=output,
                    )
                    if not resume or not _completed(output):
                        tasks.append(task)
    return tasks


def _command(task: MatrixTask) -> list[str]:
    python = ATOMBIT_PYTHON if task.mlip == "atombit" else MACE_PYTHON
    return [
        str(python),
        str(ROOT / "benchmarks/benchmark_robustness_optimization.py"),
        "--mlip",
        task.mlip,
        "--method",
        "active",
        "--optimizer",
        task.optimizer,
        "--workload-manifest",
        str(task.manifest),
        "--dataset-dir",
        str(task.dataset_dir),
        "--batch-size",
        str(task.capacity),
        "--cpu-threads",
        "1",
        "--deterministic",
        "--device",
        "cuda:0",
        "--fmax",
        "0.05",
        "--max-steps",
        str(task.max_steps),
        "--skin",
        "0.5",
        "--linear-algebra-backend",
        "auto",
        "--output",
        str(task.output),
    ]


def _worker(
    gpu: int,
    pending: queue.Queue[MatrixTask],
    records: list[dict[str, Any]],
    lock: threading.Lock,
    log_dir: Path,
) -> None:
    while True:
        try:
            task = pending.get_nowait()
        except queue.Empty:
            return
        command = _command(task)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        environment["PYTHONPATH"] = str(ROOT)
        if task.mlip == "atombit" and task.optimizer == "bfgs":
            environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        else:
            environment.pop("PYTORCH_ALLOC_CONF", None)
            environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        task.output.parent.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task.task_id}.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = {
            "task_id": task.task_id,
            "gpu": gpu,
            "command": command,
            "returncode": completed.returncode,
            "wall_seconds": time.perf_counter() - started,
            "output": str(task.output),
            "log": str(log_path),
        }
        with lock:
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        pending.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/matrix.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/raw",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=ROOT / "experiments/offline-throughput-policy/run-manifest.json",
    )
    parser.add_argument("--split", choices=("fit", "heldout", "all"), default="all")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--capacities", help="comma-separated override")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(value) for value in args.gpus.split(",") if value)
    if not gpus:
        parser.error("at least one GPU is required")
    capacities = (
        None
        if args.capacities is None
        else tuple(int(value) for value in args.capacities.split(",") if value)
    )
    tasks = _tasks(
        args.matrix,
        args.raw_dir,
        split=args.split,
        capacities=capacities,
        resume=not args.no_resume,
    )
    pending: queue.Queue[MatrixTask] = queue.Queue()
    for task in tasks:
        pending.put(task)
    records: list[dict[str, Any]] = []
    lock = threading.Lock()
    log_dir = args.raw_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    threads = [
        threading.Thread(
            target=_worker,
            args=(gpu, pending, records, lock, log_dir),
            name=f"gpu-{gpu}",
        )
        for gpu in gpus
    ]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    manifest = {
        "schema_version": 1,
        "matrix": str(args.matrix),
        "gpus": list(gpus),
        "task_count": len(tasks),
        "wall_seconds": time.perf_counter() - started,
        "records": sorted(records, key=lambda record: record["task_id"]),
    }
    args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.run_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = [record for record in records if record["returncode"] != 0]
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "failure_count": len(failures),
                "wall_seconds": manifest["wall_seconds"],
                "run_manifest": str(args.run_manifest),
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
