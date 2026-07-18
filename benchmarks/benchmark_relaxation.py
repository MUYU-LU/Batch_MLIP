"""Compare production batched FIRE relaxation with sequential ASE FIRE."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.io import read, write
from ase.optimize import FIRE
from benchmark_production import (
    environment_metadata,
    load_manifest,
    load_production_model,
    parse_int_list,
    sha256_file,
    synchronize,
    timing_summary,
    write_result,
)

from atombit_batch import AseGraphBatch, BatchedPotential, batched_fire_relax
from src.Calculator import AtomBitCalculator


class CountingAtomBitCalculator(AtomBitCalculator):
    """Production ASE calculator with explicit model-evaluation accounting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calculate_calls = 0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        self.calculate_calls += 1
        super().calculate(atoms, properties, system_changes)


def max_force(forces: np.ndarray) -> float:
    return float(np.linalg.norm(forces, axis=1).max())


def run_ase_pool(
    model: torch.nn.Module,
    systems: list[Any],
    *,
    device: torch.device,
    cutoff: float,
    fmax: float,
    max_steps: int,
    dt_start: float,
    dt_max: float,
    max_step: float,
    alpha_start: float,
    n_min: int,
    f_inc: float,
    f_dec: float,
    f_alpha: float,
) -> dict[str, Any]:
    calculator = CountingAtomBitCalculator(
        model,
        cutoff=cutoff,
        device=device,
        enable_stress=False,
        add_e0=False,
    )
    records = []
    final_systems = []
    for source in systems:
        atoms = source.copy()
        atoms.calc = calculator
        calls_before = calculator.calculate_calls
        optimizer = FIRE(
            atoms,
            logfile=None,
            trajectory=None,
            dt=dt_start,
            dtmax=dt_max,
            maxstep=max_step,
            astart=alpha_start,
            a=alpha_start,
            Nmin=n_min,
            finc=f_inc,
            fdec=f_dec,
            fa=f_alpha,
        )
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        records.append(
            {
                "source": source.info["benchmark_source"],
                "converged": converged,
                "steps": int(optimizer.nsteps),
                "model_evaluations": calculator.calculate_calls - calls_before,
                "energy_ev": energy,
                "max_force_ev_per_a": max_force(forces),
                "positions": atoms.positions.copy(),
            }
        )
        atoms.calc = None
        atoms.info.update(
            {
                "benchmark_source": source.info["benchmark_source"],
                "method": "ase_fire",
                "converged": converged,
                "optimizer_steps": int(optimizer.nsteps),
                "energy_ev": energy,
                "max_force_ev_per_a": max_force(forces),
            }
        )
        final_systems.append(atoms)
    return {
        "records": records,
        "final_systems": final_systems,
        "model_evaluations": calculator.calculate_calls,
        "neighbor_rebuilds": calculator.calculate_calls,
    }


def run_batched_pool(
    model: torch.nn.Module,
    systems: list[Any],
    *,
    batch_size: int,
    device: torch.device,
    cutoff: float,
    skin: float,
    fmax: float,
    max_steps: int,
    dt_start: float,
    dt_max: float,
    max_step: float,
    alpha_start: float,
    n_min: int,
    f_inc: float,
    f_dec: float,
    f_alpha: float,
    active_compaction: bool = False,
) -> dict[str, Any]:
    potential = BatchedPotential(
        model,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )
    records = []
    final_systems = []
    model_evaluations = 0
    graph_evaluations = 0
    uncompacted_graph_evaluations = 0
    useful_graph_evaluations = 0
    neighbor_rebuilds = 0
    chunk_steps = []
    active_batch_sizes = []

    for start in range(0, len(systems), batch_size):
        chunk = systems[start : start + batch_size]
        state = AseGraphBatch.from_ase(
            chunk,
            cutoff=cutoff,
            skin=skin,
            device=device,
            dtype=torch.float32,
        )
        relaxation = batched_fire_relax(
            state,
            potential,
            fmax=fmax,
            max_steps=max_steps,
            dt_start=dt_start,
            dt_max=dt_max,
            max_step=max_step,
            alpha_start=alpha_start,
            n_min=n_min,
            f_inc=f_inc,
            f_dec=f_dec,
            f_alpha=f_alpha,
            callback_interval=max_steps + 1,
            active_compaction=active_compaction,
        )
        model_evaluations += relaxation.model_evaluations
        graph_evaluations += relaxation.graph_evaluations
        uncompacted_graph_evaluations += relaxation.model_evaluations * len(chunk)
        neighbor_rebuilds += relaxation.state.neighbor_rebuild_count
        chunk_steps.append(relaxation.steps)
        active_batch_sizes.append(list(relaxation.active_batch_sizes))

        energies = relaxation.evaluation.energy.cpu().numpy()
        forces = relaxation.evaluation.forces.cpu().numpy()
        converged = relaxation.converged.cpu().numpy()
        converged_steps = relaxation.converged_step.cpu().numpy()
        positions = relaxation.state.positions.cpu().numpy()
        for local_index, source in enumerate(chunk):
            atom_slice = relaxation.state.atom_slice(local_index)
            sample_positions = positions[atom_slice].copy()
            sample_forces = forces[atom_slice]
            sample_step = int(converged_steps[local_index])
            effective_step = sample_step if sample_step >= 0 else relaxation.steps
            useful_graph_evaluations += effective_step + 1
            record = {
                "source": source.info["benchmark_source"],
                "converged": bool(converged[local_index]),
                "steps": effective_step,
                "energy_ev": float(energies[local_index]),
                "max_force_ev_per_a": max_force(sample_forces),
                "positions": sample_positions,
            }
            records.append(record)

            atoms = source.copy()
            atoms.positions[:] = sample_positions
            atoms.info.update(
                {
                    "benchmark_source": source.info["benchmark_source"],
                    "method": (
                        "batched_fire_active"
                        if active_compaction
                        else "batched_fire_masked"
                    ),
                    "batch_size": batch_size,
                    "converged": record["converged"],
                    "optimizer_steps": effective_step,
                    "energy_ev": record["energy_ev"],
                    "max_force_ev_per_a": record["max_force_ev_per_a"],
                }
            )
            final_systems.append(atoms)

    return {
        "records": records,
        "final_systems": final_systems,
        "model_evaluations": model_evaluations,
        "graph_evaluations": graph_evaluations,
        "uncompacted_graph_evaluations": uncompacted_graph_evaluations,
        "useful_graph_evaluations": useful_graph_evaluations,
        "wasted_graph_evaluations": graph_evaluations - useful_graph_evaluations,
        "neighbor_rebuilds": neighbor_rebuilds,
        "chunk_steps": chunk_steps,
        "active_batch_sizes": active_batch_sizes,
    }


def timed_runs(
    fn,
    *,
    repeats: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[float], int | None]:
    output = None
    samples = []
    peak_memory = 0 if device.type == "cuda" else None
    for _repeat in range(repeats):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        start = time.perf_counter()
        current = fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
        if output is None:
            output = current
        if device.type == "cuda":
            peak_memory = max(peak_memory or 0, torch.cuda.max_memory_allocated(device))
    if output is None:
        raise RuntimeError("at least one repeat is required")
    return output, samples, peak_memory


def validate_relaxations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atom_count: int,
    target_fmax: float,
    energy_per_atom_atol: float,
    force_atol: float,
    position_rmsd_atol: float,
    step_atol: int,
) -> dict[str, Any]:
    reference_records = reference["records"]
    candidate_records = candidate["records"]
    if [item["source"] for item in reference_records] != [
        item["source"] for item in candidate_records
    ]:
        raise RuntimeError("reference and batch sample orders differ")

    energy_errors = []
    force_errors = []
    rmsd_errors = []
    max_position_errors = []
    step_errors = []
    convergence_matches = []
    for expected, actual in zip(reference_records, candidate_records, strict=True):
        energy_errors.append(abs(actual["energy_ev"] - expected["energy_ev"]))
        force_errors.append(
            abs(actual["max_force_ev_per_a"] - expected["max_force_ev_per_a"])
        )
        displacement = actual["positions"] - expected["positions"]
        atom_displacement = np.linalg.norm(displacement, axis=1)
        rmsd_errors.append(float(np.sqrt(np.mean(atom_displacement**2))))
        max_position_errors.append(float(atom_displacement.max()))
        step_errors.append(abs(actual["steps"] - expected["steps"]))
        convergence_matches.append(actual["converged"] == expected["converged"])

    max_energy_error = max(energy_errors)
    max_energy_error_per_atom = max_energy_error / atom_count
    all_reference_converged = all(record["converged"] for record in reference_records)
    all_candidate_converged = all(record["converged"] for record in candidate_records)
    validation = {
        "convergence_flags_match": all(convergence_matches),
        "all_reference_converged": all_reference_converged,
        "all_candidate_converged": all_candidate_converged,
        "target_fmax_ev_per_a": target_fmax,
        "max_reference_fmax_ev_per_a": max(
            record["max_force_ev_per_a"] for record in reference_records
        ),
        "max_candidate_fmax_ev_per_a": max(
            record["max_force_ev_per_a"] for record in candidate_records
        ),
        "max_abs_energy_error_ev": max_energy_error,
        "max_abs_energy_error_ev_per_atom": max_energy_error_per_atom,
        "max_abs_final_fmax_error_ev_per_a": max(force_errors),
        "max_position_rmsd_a": max(rmsd_errors),
        "max_atom_position_error_a": max(max_position_errors),
        "max_optimizer_step_difference": max(step_errors),
        "energy_per_atom_atol_ev": energy_per_atom_atol,
        "final_fmax_atol_ev_per_a": force_atol,
        "position_rmsd_atol_a": position_rmsd_atol,
        "optimizer_step_atol": step_atol,
    }
    validation["passed"] = bool(
        validation["convergence_flags_match"]
        and all_reference_converged
        and all_candidate_converged
        and max_energy_error_per_atom <= energy_per_atom_atol
        and validation["max_abs_final_fmax_error_ev_per_a"] <= force_atol
        and validation["max_position_rmsd_a"] <= position_rmsd_atol
        and validation["max_optimizer_step_difference"] <= step_atol
    )
    return validation


def serializable_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key != "positions"}
        for record in records
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atom-counts", type=parse_int_list, default=[46, 92, 184, 276])
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="cyclically repeat the fixed samples to this total system count",
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--alpha-start", type=float, default=0.1)
    parser.add_argument("--n-min", type=int, default=5)
    parser.add_argument("--f-inc", type=float, default=1.1)
    parser.add_argument("--f-dec", type=float, default=0.5)
    parser.add_argument("--f-alpha", type=float, default=0.99)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-repeats", type=int, default=1)
    parser.add_argument("--energy-per-atom-atol", type=float, default=5e-5)
    parser.add_argument("--final-fmax-atol", type=float, default=2e-2)
    parser.add_argument("--position-rmsd-atol", type=float, default=1e-2)
    parser.add_argument("--optimizer-step-atol", type=int, default=15)
    parser.add_argument(
        "--active-compaction",
        action="store_true",
        help="remove converged graphs from subsequent batched model calls",
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=Path("runs/production_relaxation_structures")
    )
    parser.add_argument(
        "--no-write-structures",
        action="store_true",
        help="skip final extxyz files for large performance-only pools",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/production_relaxation_scaling.json")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_count <= 0 or args.repeats <= 0 or args.reference_repeats <= 0:
        raise ValueError("sample count and repeat counts must be positive")
    pool_size = args.sample_count if args.pool_size is None else args.pool_size
    if pool_size <= 0:
        raise ValueError("pool size must be positive")
    if pool_size < args.sample_count:
        raise ValueError("pool size must not be smaller than sample count")
    if any(pool_size % batch_size for batch_size in args.batch_sizes):
        raise ValueError("every batch size must divide the fixed pool size")

    manifest = load_manifest(args.manifest, args.sample_count)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "hypothesis": (
            "Active-batch compaction eliminates converged-graph model work while "
            "preserving ASE FIRE convergence and final structures."
            if args.active_compaction
            else "Batched FIRE reduces fixed-pool relaxation wall time versus "
            "sequential ASE FIRE while preserving convergence and final structures."
        ),
        "environment": environment_metadata(device),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "random_sampling": False,
            "note": (
                "CUDA scatter reductions may perturb FIRE branching near the force "
                "threshold even though the workload has no random sampling."
            ),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "sample_manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
            "sample_count_per_atom_group": args.sample_count,
            "pool_size_per_atom_group": pool_size,
            "expansion": "cyclic repetition in manifest order",
        },
        "parameters": {
            "atom_counts": args.atom_counts,
            "batch_sizes": args.batch_sizes,
            "dtype": "float32",
            "force_mode": "autograd",
            "cutoff_a": args.cutoff,
            "skin_a": args.skin,
            "fmax_ev_per_a": args.fmax,
            "max_steps": args.max_steps,
            "dt_start": args.dt_start,
            "dt_max": args.dt_max,
            "max_step_a": args.max_step,
            "alpha_start": args.alpha_start,
            "n_min": args.n_min,
            "f_inc": args.f_inc,
            "f_dec": args.f_dec,
            "f_alpha": args.f_alpha,
            "repeats": args.repeats,
            "reference_repeats": args.reference_repeats,
            "active_compaction": args.active_compaction,
            "pool_size": pool_size,
        },
        "groups": {},
    }
    write_result(args.output, result)

    for atom_count in args.atom_counts:
        base_names = manifest["samples"][str(atom_count)][: args.sample_count]
        base_systems = []
        for name in base_names:
            atoms = read(args.dataset_dir / name)
            if len(atoms) != atom_count:
                raise ValueError(f"{name} has {len(atoms)} atoms, expected {atom_count}")
            base_systems.append(atoms)
        systems = []
        for pool_index in range(pool_size):
            base_index = pool_index % len(base_systems)
            atoms = base_systems[base_index].copy()
            atoms.info["benchmark_source"] = (
                f"{base_names[base_index]}#pool-{pool_index:04d}"
            )
            systems.append(atoms)

        # Warm both paths without changing any benchmark structure.
        warmup_calculator = CountingAtomBitCalculator(
            model,
            cutoff=args.cutoff,
            device=device,
            enable_stress=False,
            add_e0=False,
        )
        warmup_calculator.calculate(
            systems[0], properties=("energy", "forces"), system_changes=all_changes
        )
        warmup_state = AseGraphBatch.from_ase(
            [systems[0]], cutoff=args.cutoff, skin=args.skin, device=device, dtype=torch.float32
        )
        warmup_potential = BatchedPotential(
            model, device=device, dtype=torch.float32, force_mode="autograd"
        )
        warmup_potential(warmup_state, neighbor_policy="never")
        synchronize(device)

        common_kwargs = {
            "model": model,
            "systems": systems,
            "device": device,
            "cutoff": args.cutoff,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "dt_start": args.dt_start,
            "dt_max": args.dt_max,
            "max_step": args.max_step,
            "alpha_start": args.alpha_start,
            "n_min": args.n_min,
            "f_inc": args.f_inc,
            "f_dec": args.f_dec,
            "f_alpha": args.f_alpha,
        }
        reference, reference_times, reference_peak = timed_runs(
            lambda common_kwargs=common_kwargs: run_ase_pool(**common_kwargs),
            repeats=args.reference_repeats,
            device=device,
        )
        reference_timing = timing_summary(reference_times)
        group = {
            "atom_count": atom_count,
            "sample_count": pool_size,
            "base_sample_count": args.sample_count,
            "base_sample_files": base_names,
            "pool_expansion": "cyclic repetition in manifest order",
            "ase_reference": {
                "timing": reference_timing,
                "systems_per_second": pool_size / reference_timing["median_seconds"],
                "peak_memory_bytes": reference_peak,
                "converged_count": sum(
                    record["converged"] for record in reference["records"]
                ),
                "optimizer_steps_total": sum(
                    record["steps"] for record in reference["records"]
                ),
                "model_evaluations": reference["model_evaluations"],
                "neighbor_rebuilds": reference["neighbor_rebuilds"],
                "records": serializable_records(reference["records"]),
            },
            "points": [],
        }
        result["groups"][str(atom_count)] = group
        if not args.no_write_structures:
            args.structures_dir.mkdir(parents=True, exist_ok=True)
            write(
                args.structures_dir / f"atoms_{atom_count}_ase.extxyz",
                reference["final_systems"],
            )
        write_result(args.output, result)

        for batch_size in args.batch_sizes:
            point = {"batch_size": batch_size, "status": "running"}
            group["points"].append(point)
            write_result(args.output, result)
            try:
                batch, batch_times, batch_peak = timed_runs(
                    lambda batch_size=batch_size, common_kwargs=common_kwargs: run_batched_pool(
                        **common_kwargs,
                        batch_size=batch_size,
                        skin=args.skin,
                        active_compaction=args.active_compaction,
                    ),
                    repeats=args.repeats,
                    device=device,
                )
                batch_timing = timing_summary(batch_times)
                validation = validate_relaxations(
                    reference,
                    batch,
                    atom_count=atom_count,
                    target_fmax=args.fmax,
                    energy_per_atom_atol=args.energy_per_atom_atol,
                    force_atol=args.final_fmax_atol,
                    position_rmsd_atol=args.position_rmsd_atol,
                    step_atol=args.optimizer_step_atol,
                )
                point.update(
                    {
                        "status": "passed" if validation["passed"] else "validation_failed",
                        "timing": batch_timing,
                        "speedup_vs_ase": reference_timing["median_seconds"]
                        / batch_timing["median_seconds"],
                        "systems_per_second": pool_size
                        / batch_timing["median_seconds"],
                        "peak_memory_bytes": batch_peak,
                        "converged_count": sum(
                            record["converged"] for record in batch["records"]
                        ),
                        "optimizer_steps_total": sum(
                            record["steps"] for record in batch["records"]
                        ),
                        "model_evaluations": batch["model_evaluations"],
                        "graph_evaluations": batch["graph_evaluations"],
                        "uncompacted_graph_evaluations": batch[
                            "uncompacted_graph_evaluations"
                        ],
                        "useful_graph_evaluations": batch[
                            "useful_graph_evaluations"
                        ],
                        "wasted_graph_evaluations": batch[
                            "wasted_graph_evaluations"
                        ],
                        "neighbor_rebuilds": batch["neighbor_rebuilds"],
                        "chunk_steps": batch["chunk_steps"],
                        "active_batch_sizes": batch["active_batch_sizes"],
                        "validation": validation,
                        "records": serializable_records(batch["records"]),
                    }
                )
                if not args.no_write_structures:
                    write(
                        args.structures_dir
                        / f"atoms_{atom_count}_batch_{batch_size}.extxyz",
                        batch["final_systems"],
                    )
            except torch.cuda.OutOfMemoryError as error:
                point.update({"status": "oom", "error": str(error)})
            except Exception as error:
                point.update(
                    {"status": "error", "error": f"{type(error).__name__}: {error}"}
                )
            finally:
                write_result(args.output, result)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    statuses = [
        point["status"]
        for group in result["groups"].values()
        for point in group["points"]
    ]
    result["status"] = (
        "passed"
        if statuses and all(status == "passed" for status in statuses)
        else "completed_with_failures"
    )
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
