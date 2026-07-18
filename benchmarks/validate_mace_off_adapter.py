#!/usr/bin/env python3
"""Validate and benchmark the native MACE-OFF batch adapter against ASE."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import BFGS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atombit_batch import (  # noqa: E402
    BatchedFrechetCellFilter,
    batched_bfgs_relax,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
    load_mace_off_batch,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def timed(fn, repeats: int) -> tuple[Any, list[float]]:
    output = None
    samples = []
    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        synchronize()
        started = time.perf_counter()
        current = fn()
        synchronize()
        samples.append(time.perf_counter() - started)
        if output is None:
            output = current
    return output, samples


def summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def evaluate_ase(systems, calculator) -> dict[str, np.ndarray]:
    energies = []
    forces = []
    stresses = []
    for source in systems:
        atoms = source.copy()
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
        stresses.append(atoms.get_stress(voigt=False))
    return {
        "energy": np.asarray(energies),
        "forces": np.concatenate(forces),
        "stress": np.stack(stresses),
    }


def evaluate_batch(systems, calculator, batch_size: int) -> dict[str, np.ndarray]:
    energies = []
    forces = []
    stresses = []
    for start in range(0, len(systems), batch_size):
        state = calculator.create_state(systems[start : start + batch_size])
        output = calculator(state, compute_stress=True)
        energies.append(output.energy.detach().cpu().numpy())
        forces.append(output.forces.detach().cpu().numpy())
        stresses.append(output.stress.detach().cpu().numpy())
    return {
        "energy": np.concatenate(energies),
        "forces": np.concatenate(forces),
        "stress": np.concatenate(stresses),
    }


def errors(reference, candidate) -> dict[str, float]:
    return {
        "max_energy_error_eV": float(
            np.max(np.abs(reference["energy"] - candidate["energy"]))
        ),
        "max_force_error_eV_per_A": float(
            np.max(np.abs(reference["forces"] - candidate["forces"]))
        ),
        "max_stress_error_eV_per_A3": float(
            np.max(np.abs(reference["stress"] - candidate["stress"]))
        ),
    }


def optimize_ase(systems, calculator, steps: int) -> dict[str, Any]:
    records = []
    for source in systems:
        atoms = source.copy()
        atoms.calc = calculator
        optimizer = BFGS(
            FrechetCellFilter(atoms),
            logfile=None,
            trajectory=None,
            alpha=70.0,
            maxstep=0.2,
        )
        optimizer.run(fmax=1e-30, steps=steps)
        records.append(
            {
                "energy": atoms.get_potential_energy(),
                "forces": atoms.get_forces(),
                "stress": atoms.get_stress(voigt=False),
                "positions": atoms.positions.copy(),
                "cell": atoms.cell.array.copy(),
                "steps": optimizer.nsteps,
            }
        )
    return {"records": records}


def optimize_batch(systems, calculator, steps: int) -> dict[str, Any]:
    state = calculator.create_state(systems)
    result = batched_bfgs_relax(
        state,
        calculator,
        cell_filter=BatchedFrechetCellFilter(),
        active_compaction=True,
        fmax=1e-30,
        max_steps=steps,
        alpha=70.0,
        max_step=0.2,
        optimizer_dtype="float64",
        callback_interval=steps + 1,
    )
    energies = result.evaluation.energy.detach().cpu().numpy()
    forces = result.evaluation.forces.detach().cpu().numpy()
    stresses = result.evaluation.stress.detach().cpu().numpy()
    positions = result.state.positions.detach().cpu().numpy()
    cells = result.state.cells.detach().cpu().numpy()
    records = []
    for system_id in range(result.state.n_systems):
        atom_slice = result.state.atom_slice(system_id)
        records.append(
            {
                "energy": energies[system_id],
                "forces": forces[atom_slice],
                "stress": stresses[system_id],
                "positions": positions[atom_slice],
                "cell": cells[system_id],
                "steps": result.steps,
            }
        )
    return {
        "records": records,
        "model_evaluations": result.model_evaluations,
        "graph_evaluations": result.graph_evaluations,
    }


def optimization_errors(reference, candidate) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "max_energy_error_eV": 0.0,
        "max_force_error_eV_per_A": 0.0,
        "max_stress_error_eV_per_A3": 0.0,
        "max_position_rmsd_A": 0.0,
        "max_cell_rmsd_A": 0.0,
        "max_step_difference": 0,
    }
    for expected, actual in zip(
        reference["records"], candidate["records"], strict=True
    ):
        metrics["max_energy_error_eV"] = max(
            metrics["max_energy_error_eV"],
            abs(float(expected["energy"]) - float(actual["energy"])),
        )
        for key, metric in (
            ("forces", "max_force_error_eV_per_A"),
            ("stress", "max_stress_error_eV_per_A3"),
        ):
            metrics[metric] = max(
                metrics[metric],
                float(np.max(np.abs(expected[key] - actual[key]))),
            )
        for key, metric in (
            ("positions", "max_position_rmsd_A"),
            ("cell", "max_cell_rmsd_A"),
        ):
            metrics[metric] = max(
                metrics[metric],
                float(np.sqrt(np.mean((expected[key] - actual[key]) ** 2))),
            )
        metrics["max_step_difference"] = max(
            metrics["max_step_difference"],
            abs(int(expected["steps"]) - int(actual["steps"])),
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom-count", type=int, default=46)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--optimization-pool-size", type=int, default=4)
    parser.add_argument("--optimization-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if any(args.pool_size % value for value in batch_sizes):
        raise ValueError("each batch size must divide the pool size")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    names = manifest["samples"][str(args.atom_count)][: args.pool_size]
    systems = [read(args.dataset_dir / name) for name in names]

    calculator = load_mace_off_batch(
        "small", device=args.device, dtype=torch.float64
    )
    from mace.calculators import MACECalculator

    ase_calculator = MACECalculator(
        models=calculator.model,
        device=args.device,
        default_dtype="float64",
    )

    # Warm both graph/model paths outside timing.
    evaluate_batch(systems[:1], calculator, 1)
    evaluate_ase(systems[:1], ase_calculator)

    ase_output, ase_samples = timed(
        lambda: evaluate_ase(systems, ase_calculator), args.repeats
    )
    points = []
    for batch_size in batch_sizes:
        output, samples = timed(
            lambda size=batch_size: evaluate_batch(systems, calculator, size),
            args.repeats,
        )
        point_errors = errors(ase_output, output)
        points.append(
            {
                "batch_size": batch_size,
                "timing": summary(samples),
                "speedup_vs_ase": float(np.median(ase_samples) / np.median(samples)),
                "errors_vs_ase": point_errors,
                "passed": all(value < 1e-8 for value in point_errors.values()),
            }
        )

    optimization_systems = systems[: args.optimization_pool_size]
    ase_optimization, ase_opt_samples = timed(
        lambda: optimize_ase(
            optimization_systems, ase_calculator, args.optimization_steps
        ),
        args.repeats,
    )
    batch_optimization, batch_opt_samples = timed(
        lambda: optimize_batch(
            optimization_systems, calculator, args.optimization_steps
        ),
        args.repeats,
    )
    opt_errors = optimization_errors(ase_optimization, batch_optimization)

    md_state = calculator.create_state(systems[:2])
    initialize_maxwell_boltzmann(
        md_state, 300.0, seed=17, force_exact_temperature=True
    )
    md_result = batched_velocity_verlet(
        md_state, calculator, timestep_fs=0.25, n_steps=1
    )

    result = {
        "status": "passed",
        "model": "MACE-OFF23-Small",
        "mace_version": importlib.metadata.version("mace-torch"),
        "dtype": "float64",
        "device": args.device,
        "atom_count": args.atom_count,
        "pool_size": args.pool_size,
        "sample_files": names,
        "single_point": {
            "ase_timing": summary(ase_samples),
            "points": points,
        },
        "variable_cell_bfgs": {
            "pool_size": args.optimization_pool_size,
            "steps": args.optimization_steps,
            "ase_timing": summary(ase_opt_samples),
            "batch_timing": summary(batch_opt_samples),
            "speedup_vs_ase": float(
                np.median(ase_opt_samples) / np.median(batch_opt_samples)
            ),
            "errors_vs_ase": opt_errors,
            "model_evaluations": batch_optimization["model_evaluations"],
            "graph_evaluations": batch_optimization["graph_evaluations"],
        },
        "nve_smoke": {
            "systems": 2,
            "steps": md_result.steps,
            "finite_energy": bool(torch.isfinite(md_result.evaluation.energy).all()),
            "finite_temperature": bool(torch.isfinite(md_result.temperature).all()),
        },
    }
    if not all(point["passed"] for point in points):
        result["status"] = "failed"
    if not result["nve_smoke"]["finite_energy"]:
        result["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
