#!/usr/bin/env python3
"""Benchmark ASE optimization, evaluation, or MD with CUDA MPS workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), default="atombit")
    parser.add_argument(
        "--task",
        choices=("optimization", "evaluation", "nve", "nvt", "npt"),
        default="optimization",
    )
    parser.add_argument("--atom-count", type=int)
    parser.add_argument(
        "--optimizer",
        choices=("fire", "bfgs", "bfgslinesearch"),
    )
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--measured-steps", type=int)
    parser.add_argument("--timestep-fs", type=float)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--optimizer-dtype",
        choices=("state", "float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/T2_test/structures"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/t2_fixed_samples.json"),
    )
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help="Signed workload manifest; supersedes --atom-count and --manifest.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--atombit-e0", type=Path)
    parser.add_argument("--mace-model", default="small")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu-threads-per-worker", type=int, default=1)
    parser.add_argument("--worker-start-interval", type=float, default=0.0)
    parser.add_argument("--memory-sample-interval", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pool_size <= 0 or args.workers <= 0 or args.cpu_threads_per_worker <= 0:
        parser.error("pool size, worker count, and CPU thread count must be positive")
    if args.worker_start_interval < 0:
        parser.error("worker start interval must be non-negative")
    if args.pool_size % args.workers:
        parser.error("pool size must be divisible by worker count")
    if args.task == "optimization" and args.optimizer is None:
        parser.error("--optimizer is required for optimization")
    if args.task != "optimization" and args.workload_manifest is None:
        parser.error("evaluation and MD require --workload-manifest")
    if args.task in ("nve", "nvt", "npt") and (
        (args.warmup_steps is not None and args.warmup_steps < 0)
        or (args.measured_steps is not None and args.measured_steps <= 0)
        or (args.timestep_fs is not None and args.timestep_fs <= 0.0)
    ):
        parser.error("MD steps must be non-negative/positive and timestep positive")
    if args.workload_manifest is None and args.atom_count is None:
        parser.error("--atom-count is required without --workload-manifest")
    if args.mlip == "atombit" and args.checkpoint is None:
        parser.error("--checkpoint is required for AtomBit")
    for variable in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        if not os.environ.get(variable):
            parser.error(f"{variable} must identify the active MPS daemon")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_throughput(
    *,
    task: str,
    pool_size: int,
    elapsed_seconds: float,
    measured_steps: int | None,
) -> tuple[float, str]:
    """Return the task-appropriate MPS pool throughput."""

    if task in ("nve", "nvt", "npt"):
        if measured_steps is None or measured_steps <= 0:
            raise ValueError("MD throughput requires positive measured steps")
        return (
            pool_size * measured_steps / elapsed_seconds,
            "replica_steps_per_second",
        )
    return pool_size / elapsed_seconds, "systems_per_second"


def consistent_worker_parameters(
    worker_results: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve parameters that must be identical across MPS workers."""

    if not worker_results:
        raise ValueError("worker_results must not be empty")
    resolved = {key: worker_results[0].get(key) for key in keys}
    if any(
        worker_result.get(key) != value
        for worker_result in worker_results
        for key, value in resolved.items()
    ):
        raise RuntimeError("MPS workers used inconsistent parameters")
    return resolved


def _worker_impl(
    worker_id: int,
    args: argparse.Namespace,
    barrier: Any,
    result_path: Path,
) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"logm result may be inaccurate.*",
        category=RuntimeWarning,
    )
    import numpy as np
    import torch
    from ase import units
    from ase.calculators.calculator import all_changes
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.nose_hoover_chain import IsotropicMTKNPT
    from ase.md.verlet import VelocityVerlet

    benchmark_dir = Path(__file__).resolve().parent
    repository_root = benchmark_dir.parent
    sys.path[:0] = [str(repository_root), str(benchmark_dir)]
    from benchmark_production import load_manifest, load_production_model, synchronize

    from batch_mlip import AseGraphBatch, initialize_maxwell_boltzmann
    from batch_mlip.workloads import read_workload_manifest

    torch.use_deterministic_algorithms(args.deterministic)
    torch.set_num_threads(args.cpu_threads_per_worker)
    torch.set_num_interop_threads(1)
    shard_size = args.pool_size // args.workers
    start = worker_id * shard_size
    systems = []
    if args.workload_manifest is not None:
        workload = read_workload_manifest(args.workload_manifest)
        if args.pool_size > len(workload.jobs):
            raise ValueError(
                f"pool size {args.pool_size} exceeds signed workload "
                f"size {len(workload.jobs)}"
            )
        jobs = workload.jobs[start : start + shard_size]
        names = [job.system_id for job in jobs]
        for job in jobs:
            atoms = read(
                args.dataset_dir / job.source_path,
                index=job.frame_index,
            )
            atoms.info["benchmark_source"] = job.system_id
            atoms.info["benchmark_source_path"] = job.source_path
            systems.append(atoms)
    else:
        manifest = load_manifest(args.manifest, min(args.pool_size, 32))
        available = manifest["samples"][str(args.atom_count)]
        base_names = available[: min(args.pool_size, len(available))]
        global_names = [
            base_names[index % len(base_names)]
            for index in range(args.pool_size)
        ]
        names = global_names[start : start + shard_size]
        for name in names:
            atoms = read(args.dataset_dir / name)
            if len(atoms) != args.atom_count:
                raise ValueError(f"{name} has {len(atoms)} atoms")
            atoms.info["benchmark_source"] = name
            systems.append(atoms)

    device = torch.device(args.device)
    if args.mlip == "atombit":
        from benchmark_variable_cell_scaling import (
            AtomBitBatchCalculator,
            CountingAtomBitCalculator,
            run_ase,
        )

        model_dtype = getattr(torch, args.model_dtype)
        optimizer_dtype = (
            None if args.optimizer_dtype == "state" else args.optimizer_dtype
        )
        model, _ = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=model_dtype).eval()
        warm_batch = AtomBitBatchCalculator(
            model,
            cutoff=args.cutoff,
            device=device,
            dtype=model_dtype,
            force_mode="autograd",
        )
        warm_batch(warm_batch.create_state([systems[0]]), compute_stress=True)
        warm_ase = CountingAtomBitCalculator(
            model,
            cutoff=args.cutoff,
            device=device,
            dtype=model_dtype,
            enable_stress=True,
            add_e0=args.atombit_e0 is not None,
            e0_path=args.atombit_e0,
        )
        warm_ase.calculate(
            systems[0],
            properties=("energy", "forces", "stress"),
            system_changes=all_changes,
        )

        ase_calculator = warm_ase
        state_dtype = model_dtype
        model_cutoff = args.cutoff

        def execute_optimization():
            return run_ase(
                model,
                systems,
                device=device,
                cutoff=args.cutoff,
                fmax=args.fmax,
                max_steps=args.max_steps,
                dt_start=args.dt_start,
                dt_max=args.dt_max,
                max_step=args.max_step,
                optimizer_name=str(args.optimizer),
                alpha=args.alpha,
                optimizer_dtype=optimizer_dtype,
                model_dtype=model_dtype,
            )

    else:
        from benchmark_mace_variable_cell_scaling import (
            make_counting_ase_calculator,
            run_ase,
        )

        from batch_mlip import MACEBatchCalculator

        warm_batch = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=device,
            dtype=torch.float64,
            graph_mode="rebuild",
        )
        calculator = make_counting_ase_calculator(
            warm_batch.model,
            device=device,
        )
        warm_batch(warm_batch.create_state([systems[0]]), compute_stress=True)
        calculator.calculate(
            systems[0],
            properties=("energy", "forces", "stress"),
            system_changes=all_changes,
        )

        ase_calculator = calculator
        state_dtype = torch.float64
        model_cutoff = warm_batch.cutoff

        def execute_optimization():
            return run_ase(
                calculator,
                systems,
                fmax=args.fmax,
                max_steps=args.max_steps,
                dt_start=args.dt_start,
                dt_max=args.dt_max,
                max_step=args.max_step,
                optimizer_name=str(args.optimizer),
                alpha=args.alpha,
            )

    def execute_evaluation():
        calls_before = ase_calculator.calculate_calls
        records = []
        for source in systems:
            atoms = source.copy()
            atoms.calc = ase_calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            records.append(
                {
                    "source": source.info["benchmark_source"],
                    "energy_eV": energy,
                    "max_force_eV_per_A": float(
                        np.linalg.vector_norm(forces, axis=1).max()
                    ),
                    "finite": bool(
                        np.isfinite(energy) and np.isfinite(forces).all()
                    ),
                }
            )
        evaluations = ase_calculator.calculate_calls - calls_before
        return {
            "records": records,
            "model_evaluations": evaluations,
            "graph_evaluations": evaluations,
            "neighbor_rebuilds": evaluations,
            "optimizer_steps_total": 0,
        }

    warmup_steps = None
    measured_steps = None
    timestep_fs = None
    if args.task in ("nve", "nvt", "npt"):
        if args.workload_manifest is None:
            raise RuntimeError("MD requires a signed workload")
        warmup_steps = (
            int(args.warmup_steps)
            if args.warmup_steps is not None
            else int(workload.metadata["warmup_steps"])
        )
        measured_steps = (
            int(args.measured_steps)
            if args.measured_steps is not None
            else int(workload.metadata["measured_steps"])
        )
        timestep_fs = (
            float(args.timestep_fs)
            if args.timestep_fs is not None
            else float(workload.metadata["timestep_fs"])
        )
        initial_state = AseGraphBatch.from_ase(
            systems,
            cutoff=model_cutoff,
            device=device,
            dtype=state_dtype,
            build_neighbors=False,
        )
        initialize_maxwell_boltzmann(
            initial_state,
            float(workload.metadata["initial_temperature_K"]),
            seed=[int(job.random_seed) for job in jobs],
            remove_com=bool(workload.metadata["remove_initial_com"]),
            force_exact_temperature=bool(
                workload.metadata["force_exact_initial_temperature"]
            ),
        )
        systems = initial_state.to_ase(evaluation=None, wrap=False)
        dynamics = []
        for atoms, name, job in zip(systems, names, jobs, strict=True):
            atoms.info["benchmark_source"] = name
            atoms.calc = ase_calculator
            if args.task == "nve":
                dynamics_item = VelocityVerlet(
                    atoms,
                    timestep=timestep_fs * units.fs,
                    logfile=None,
                    trajectory=None,
                )
            elif args.task == "nvt":
                dynamics_item = Langevin(
                    atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=300.0,
                    friction=0.01 / units.fs,
                    fixcm=False,
                    rng=np.random.RandomState(int(job.random_seed) + 1),
                    logfile=None,
                    trajectory=None,
                )
            else:
                dynamics_item = IsotropicMTKNPT(
                    atoms,
                    timestep=timestep_fs * units.fs,
                    temperature_K=300.0,
                    pressure_au=0.0,
                    tdamp=50.0 * units.fs,
                    pdamp=500.0 * units.fs,
                    logfile=None,
                    trajectory=None,
                )
            dynamics.append(dynamics_item)
        for dynamics_item in dynamics:
            dynamics_item.run(warmup_steps)
        synchronize(device)
        initial_total_energy = [
            float(atoms.get_total_energy()) for atoms in systems
        ]

        def execute_md():
            calls_before = ase_calculator.calculate_calls
            for dynamics_item in dynamics:
                dynamics_item.run(measured_steps)
            records = []
            for atoms, name, initial_energy in zip(
                systems,
                names,
                initial_total_energy,
                strict=True,
            ):
                final_energy = float(atoms.get_total_energy())
                positions = np.asarray(atoms.positions, dtype=np.float64)
                velocities = np.asarray(
                    atoms.get_velocities(),
                    dtype=np.float64,
                )
                records.append(
                    {
                        "source": name,
                        "initial_total_energy_eV": initial_energy,
                        "final_total_energy_eV": final_energy,
                        "energy_drift_eV_per_atom": (
                            (final_energy - initial_energy) / len(atoms)
                            if args.task == "nve"
                            else None
                        ),
                        "temperature_K": float(atoms.get_temperature()),
                        "volume_A3": float(atoms.get_volume()),
                        "position_rms_A": float(
                            np.sqrt(np.mean(np.square(positions)))
                        ),
                        "velocity_rms_A_per_fs": float(
                            np.sqrt(np.mean(np.square(velocities))) * units.fs
                        ),
                        "finite": bool(
                            np.isfinite(final_energy)
                            and np.isfinite(positions).all()
                            and np.isfinite(velocities).all()
                        ),
                    }
                )
            evaluations = ase_calculator.calculate_calls - calls_before
            return {
                "records": records,
                "model_evaluations": evaluations,
                "graph_evaluations": evaluations,
                "neighbor_rebuilds": evaluations,
                "optimizer_steps_total": shard_size * measured_steps,
            }

    if args.task == "optimization":
        execute = execute_optimization
    elif args.task == "evaluation":
        execute = execute_evaluation
    else:
        execute = execute_md

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    barrier.wait(timeout=600)
    started = time.perf_counter()
    output = execute()
    synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "worker_id": worker_id,
        "status": "passed",
        "sample_files": names,
        "elapsed_seconds": elapsed,
        "systems_per_second": shard_size / elapsed,
        "throughput_per_second": (
            shard_size * measured_steps / elapsed
            if measured_steps is not None
            else shard_size / elapsed
        ),
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "timestep_fs": timestep_fs,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
        **output,
    }
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")


def _worker(
    worker_id: int,
    args: argparse.Namespace,
    barrier: Any,
    result_path: Path,
) -> None:
    try:
        _worker_impl(worker_id, args, barrier, result_path)
    except BaseException as exc:
        result_path.write_text(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        barrier.abort()
        raise


def _gpu_memory_monitor(
    gpu_index: int,
    interval: float,
    stop: threading.Event,
    samples: list[int],
) -> None:
    while not stop.is_set():
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            try:
                samples.append(int(completed.stdout.strip()) * 1024 * 1024)
            except ValueError:
                pass
        stop.wait(interval)


def main() -> None:
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    thread_count = str(args.cpu_threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    barrier = context.Barrier(args.workers + 1)
    run_id = os.getpid()
    worker_paths = [
        args.output.with_suffix(
            f".run-{run_id}.worker-{worker_id}.json"
        )
        for worker_id in range(args.workers)
    ]
    for worker_path in worker_paths:
        worker_path.unlink(missing_ok=True)
    processes = [
        context.Process(
            target=_worker,
            args=(worker_id, args, barrier, worker_paths[worker_id]),
            name=f"mps-ase-worker-{worker_id}",
        )
        for worker_id in range(args.workers)
    ]
    for process in processes:
        process.start()
        if args.worker_start_interval:
            time.sleep(args.worker_start_interval)

    try:
        barrier.wait(timeout=600)
    except threading.BrokenBarrierError:
        for process in processes:
            process.join(timeout=1)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        failures = {
            process.name: process.exitcode
            for process in processes
            if process.exitcode != 0
        }
        raise RuntimeError(f"MPS workers failed during warmup: {failures}") from None

    memory_samples: list[int] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_gpu_memory_monitor,
        args=(
            args.gpu_index,
            args.memory_sample_interval,
            stop_monitor,
            memory_samples,
        ),
        daemon=True,
    )
    monitor.start()
    started = time.perf_counter()
    while not all(worker_path.exists() for worker_path in worker_paths):
        if all(process.exitcode is not None for process in processes):
            break
        time.sleep(0.01)
    elapsed = time.perf_counter() - started
    stop_monitor.set()
    monitor.join()
    for process in processes:
        process.join()

    worker_results = [
        json.loads(worker_path.read_text(encoding="utf-8"))
        for worker_path in worker_paths
        if worker_path.exists()
    ]
    worker_results.sort(key=lambda item: item["worker_id"])
    failures = {
        process.name: process.exitcode
        for process in processes
        if process.exitcode != 0
    }
    if failures or len(worker_results) != args.workers:
        raise RuntimeError(
            f"MPS worker failure: exit_codes={failures}, "
            f"results={len(worker_results)}/{args.workers}"
        )
    failed_results = [
        worker_result
        for worker_result in worker_results
        if worker_result["status"] != "passed"
    ]
    if failed_results:
        raise RuntimeError(f"MPS workers reported failures: {failed_results}")

    resolved_md_parameters = consistent_worker_parameters(
        worker_results,
        ("warmup_steps", "measured_steps", "timestep_fs"),
    )
    measured_steps = resolved_md_parameters["measured_steps"]
    throughput, throughput_unit = aggregate_throughput(
        task=args.task,
        pool_size=args.pool_size,
        elapsed_seconds=elapsed,
        measured_steps=measured_steps,
    )
    records = [
        record
        for worker_result in worker_results
        for record in worker_result["records"]
    ]
    if args.workload_manifest is not None:
        manifest_metadata = {
            "path": str(args.workload_manifest.resolve()),
            "sha256": sha256_file(args.workload_manifest),
            "kind": "signed_workload",
        }
    else:
        manifest_metadata = {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
            "kind": "legacy_fixed_samples",
        }
    model_metadata = (
        {
            "kind": "checkpoint",
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        }
        if args.mlip == "atombit"
        else {"kind": "mace_off", "model": args.mace_model}
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "method": "ase_cuda_mps",
        "task": args.task,
        "mlip": args.mlip,
        "optimizer": args.optimizer,
        "atom_count": args.atom_count,
        "pool_size": args.pool_size,
        "workers": args.workers,
        "systems_per_worker": args.pool_size // args.workers,
        "mps": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pipe_directory": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
            "log_directory": os.environ["CUDA_MPS_LOG_DIRECTORY"],
        },
        "model": model_metadata,
        "checkpoint": model_metadata if args.mlip == "atombit" else None,
        "manifest": manifest_metadata,
        "parameters": {
            "cutoff_A": args.cutoff,
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "dt_start": args.dt_start,
            "dt_max": args.dt_max,
            "max_step_A": args.max_step,
            "bfgs_alpha_eV_per_A2": args.alpha,
            "model_dtype": args.model_dtype,
            "optimizer_dtype": args.optimizer_dtype,
            "cpu_threads_per_worker": args.cpu_threads_per_worker,
            "worker_start_interval_seconds": args.worker_start_interval,
            "deterministic_algorithms": args.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cell_filter": "ASE FrechetCellFilter",
            "warmup_steps": resolved_md_parameters["warmup_steps"],
            "measured_steps": measured_steps,
            "timestep_fs": resolved_md_parameters["timestep_fs"],
            "temperature_K": (
                300.0 if args.task in ("nvt", "npt") else None
            ),
            "friction_per_fs": 0.01 if args.task == "nvt" else None,
            "pressure_eV_per_A3": 0.0 if args.task == "npt" else None,
            "thermostat_damping_fs": 50.0 if args.task == "npt" else None,
            "barostat_damping_fs": 500.0 if args.task == "npt" else None,
        },
        "timing": {
            "wall_seconds": elapsed,
            "systems_per_second": args.pool_size / elapsed,
            "throughput_per_second": throughput,
            "throughput_unit": throughput_unit,
            "worker_seconds": [
                worker_result["elapsed_seconds"] for worker_result in worker_results
            ],
        },
        "peak_gpu_memory_bytes_nvidia_smi": max(memory_samples, default=None),
        "gpu_memory_samples": len(memory_samples),
        "converged": (
            sum(bool(record["converged"]) for record in records)
            if args.task == "optimization"
            else None
        ),
        "finite": sum(bool(record["finite"]) for record in records)
        if args.task != "optimization"
        else None,
        "optimizer_steps_total": sum(
            worker_result["optimizer_steps_total"]
            for worker_result in worker_results
        ),
        "model_evaluations_total": sum(
            worker_result["model_evaluations"]
            for worker_result in worker_results
        ),
        "worker_results": worker_results,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for worker_path in worker_paths:
        worker_path.unlink()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "wall_seconds": elapsed,
                "systems_per_second": args.pool_size / elapsed,
                "converged": result["converged"],
                "peak_gpu_memory_bytes_nvidia_smi": result[
                    "peak_gpu_memory_bytes_nvidia_smi"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
