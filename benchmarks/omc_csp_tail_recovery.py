"""Persistent ordinary-ASE recovery for an OMC-CSP nonconverged tail."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.io import read
from ase.optimize import BFGS

from batch_mlip.execution import PersistentTaskPool, TaskWorker
from benchmarks.benchmark_production import load_production_model
from benchmarks.benchmark_variable_cell_scaling import (
    CountingAtomBitCalculator,
    serialize_record,
)


@dataclass(frozen=True)
class AseBFGSRecoveryTask:
    """Lightweight reference to one frozen input structure."""

    source: str
    source_path: str
    frame_index: int
    estimated_cost: float


def recovery_task_cost(*, atom_count: int, candidate_edges: int) -> float:
    """Match the static OMC-CSP BFGS cost proxy used by the MPS baseline."""

    dimension = 3 * int(atom_count) + 9
    return (
        16.0 * dimension * dimension
        + 256.0 * int(atom_count)
        + 64.0 * int(candidate_edges)
    )


def nonconverged_sources(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return source IDs for failed endpoints after validating uniqueness."""

    sources = [str(record["source"]) for record in records]
    if len(set(sources)) != len(sources):
        raise ValueError("base endpoint records contain duplicate source IDs")
    return tuple(
        source
        for source, record in zip(sources, records, strict=True)
        if not bool(record["converged"])
    )


def replace_nonconverged_records(
    base_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replace every failed base endpoint exactly once, preserving input order."""

    expected = nonconverged_sources(base_records)
    recovered = [str(record["source"]) for record in recovery_records]
    if len(set(recovered)) != len(recovered):
        raise ValueError("recovery endpoint records contain duplicate source IDs")
    if set(recovered) != set(expected) or len(recovered) != len(expected):
        missing = sorted(set(expected) - set(recovered))
        unexpected = sorted(set(recovered) - set(expected))
        raise ValueError(
            "recovery endpoint coverage does not match the nonconverged tail: "
            f"missing={missing}, unexpected={unexpected}"
        )
    by_source = {
        str(record["source"]): dict(record) for record in recovery_records
    }
    return [
        by_source.get(str(record["source"]), dict(record))
        for record in base_records
    ]


@dataclass(frozen=True)
class _RecoveryWorkerPreparer:
    checkpoint: str
    dataset_dir: str
    warmup_task: AseBFGSRecoveryTask
    cutoff: float
    fmax: float
    max_steps: int
    max_step: float
    alpha: float

    def __call__(self, worker: TaskWorker) -> _RecoveryWorkerRunner:
        device = torch.device(worker.device)
        if device.type != "cuda":
            raise ValueError("OMC-CSP ASE tail recovery requires CUDA workers")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized before recovery worker setup")
        torch.cuda.set_device(device)
        model, _ = load_production_model(Path(self.checkpoint))
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = CountingAtomBitCalculator(
            model,
            cutoff=self.cutoff,
            device=device,
            dtype=torch.float32,
            enable_stress=True,
            add_e0=False,
        )
        warmup = read(
            Path(self.dataset_dir) / self.warmup_task.source_path,
            index=self.warmup_task.frame_index,
        )
        warmup.calc = calculator
        warmup.get_potential_energy()
        warmup.get_forces()
        warmup.get_stress(voigt=False)
        torch.cuda.synchronize(device)
        calculator.calculate_calls = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return _RecoveryWorkerRunner(
            calculator=calculator,
            dataset_dir=self.dataset_dir,
            device=str(device),
            fmax=self.fmax,
            max_steps=self.max_steps,
            max_step=self.max_step,
            alpha=self.alpha,
        )


@dataclass
class _RecoveryWorkerRunner:
    calculator: CountingAtomBitCalculator
    dataset_dir: str
    device: str
    fmax: float
    max_steps: int
    max_step: float
    alpha: float

    def __call__(self, task: AseBFGSRecoveryTask) -> dict[str, Any]:
        device = torch.device(self.device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started_calls = self.calculator.calculate_calls
        materialization_started = time.perf_counter()
        atoms = read(
            Path(self.dataset_dir) / task.source_path,
            index=task.frame_index,
        )
        atoms.info["benchmark_source"] = task.source
        materialization_seconds = time.perf_counter() - materialization_started
        atoms.calc = self.calculator
        target = ASEFrechetCellFilter(atoms)
        optimizer = BFGS(
            target,
            logfile=None,
            trajectory=None,
            maxstep=self.max_step,
            alpha=self.alpha,
        )
        optimization_started = time.perf_counter()
        converged = bool(
            optimizer.run(fmax=self.fmax, steps=self.max_steps)
        )
        optimization_seconds = time.perf_counter() - optimization_started
        forces = np.asarray(atoms.get_forces())
        stress = np.asarray(atoms.get_stress(voigt=False))
        record = serialize_record(
            source=task.source,
            converged=converged,
            steps=int(optimizer.nsteps),
            energy=float(atoms.get_potential_energy()),
            forces=forces,
            stress=stress,
            positions=np.asarray(atoms.positions),
            cell=np.asarray(atoms.cell.array),
        )
        torch.cuda.synchronize(device)
        return {
            "record": record,
            "source": task.source,
            "pid": os.getpid(),
            "device": str(device),
            "materialization_seconds": materialization_seconds,
            "optimization_seconds": optimization_seconds,
            "model_evaluations": (
                self.calculator.calculate_calls - started_calls
            ),
            "graph_evaluations": (
                self.calculator.calculate_calls - started_calls
            ),
            "optimizer_steps": int(optimizer.nsteps),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }


def run_ase_bfgs_tail_recovery(
    tasks: Sequence[AseBFGSRecoveryTask],
    *,
    checkpoint: Path,
    dataset_dir: Path,
    devices: Sequence[torch.device],
    cutoff: float,
    fmax: float,
    max_steps: int,
    max_step: float,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover a failed tail with dynamically assigned persistent GPU workers."""

    normalized = tuple(tasks)
    if not normalized:
        return [], {
            "enabled": True,
            "attempted_count": 0,
            "converged_count": 0,
            "total_seconds": 0.0,
            "startup_seconds": 0.0,
            "run_seconds": 0.0,
            "shutdown_seconds": 0.0,
            "model_evaluations": 0,
            "graph_evaluations": 0,
            "optimizer_steps": 0,
            "workers": [],
            "tasks": [],
            "peak_allocated_bytes_by_device": {},
            "peak_reserved_bytes_by_device": {},
        }
    if len({task.source for task in normalized}) != len(normalized):
        raise ValueError("recovery tasks contain duplicate source IDs")
    selected_devices = tuple(devices[: min(len(devices), len(normalized))])
    if not selected_devices:
        raise ValueError("tail recovery requires at least one device")
    preparer = _RecoveryWorkerPreparer(
        checkpoint=str(checkpoint.resolve()),
        dataset_dir=str(dataset_dir.resolve()),
        warmup_task=normalized[0],
        cutoff=cutoff,
        fmax=fmax,
        max_steps=max_steps,
        max_step=max_step,
        alpha=alpha,
    )
    total_started = time.perf_counter()
    pool = PersistentTaskPool(
        [str(device) for device in selected_devices],
        preparer,
        startup_timeout_seconds=1800.0,
        run_timeout_seconds=max(7200.0, 7200.0 * len(normalized)),
        shutdown_timeout_seconds=5.0,
    )
    try:
        execution = pool.execute(
            normalized,
            [task.estimated_cost for task in normalized],
        )
        task_payloads = [
            dict(task_result.payload)
            for task_result in execution.task_results
        ]
        workers = [
            {
                "worker_id": worker_result.worker.worker_id,
                "device": worker_result.worker.device,
                "pid": pool.worker_pids[worker_result.worker.worker_id],
                "startup_seconds": worker_result.startup_seconds,
                "run_seconds": worker_result.run_seconds,
                "task_indices": list(worker_result.task_indices),
                "sources": [
                    normalized[index].source
                    for index in worker_result.task_indices
                ],
            }
            for worker_result in execution.worker_results
        ]
        startup_seconds = pool.startup_wall_seconds
        run_seconds = execution.run_wall_seconds
    finally:
        pool.close()
    peak_allocated: dict[str, int] = {}
    peak_reserved: dict[str, int] = {}
    for payload in task_payloads:
        device = str(payload["device"])
        peak_allocated[device] = max(
            peak_allocated.get(device, 0),
            int(payload["peak_allocated_bytes"]),
        )
        peak_reserved[device] = max(
            peak_reserved.get(device, 0),
            int(payload["peak_reserved_bytes"]),
        )
    records = [dict(payload["record"]) for payload in task_payloads]
    telemetry = {
        "enabled": True,
        "attempted_count": len(normalized),
        "converged_count": sum(
            int(record["converged"]) for record in records
        ),
        "total_seconds": time.perf_counter() - total_started,
        "startup_seconds": startup_seconds,
        "run_seconds": run_seconds,
        "shutdown_seconds": pool.shutdown_wall_seconds,
        "model_evaluations": sum(
            int(payload["model_evaluations"]) for payload in task_payloads
        ),
        "graph_evaluations": sum(
            int(payload["graph_evaluations"]) for payload in task_payloads
        ),
        "optimizer_steps": sum(
            int(payload["optimizer_steps"]) for payload in task_payloads
        ),
        "workers": workers,
        "tasks": task_payloads,
        "peak_allocated_bytes_by_device": peak_allocated,
        "peak_reserved_bytes_by_device": peak_reserved,
    }
    return records, telemetry
