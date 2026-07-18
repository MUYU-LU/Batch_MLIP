#!/usr/bin/env python3
"""Benchmark MACE ASE, masked, or active variable-cell optimization."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import torch
from ase.calculators.calculator import all_changes
from ase.filters import FrechetCellFilter as ASEFrechetCellFilter
from ase.io import read
from ase.optimize import BFGS, FIRE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_manifest,
    parse_int_list,
    sha256_file,
    synchronize,
    write_result,
)
from benchmark_variable_cell_scaling import (  # noqa: E402
    serialize_record,
    timed_repeats,
)

from batch_mlip import (  # noqa: E402
    FrechetCellFilter,
    MACEBatchCalculator,
    batched_bfgs_relax,
    batched_fire_relax,
)


def make_counting_ase_calculator(model, *, device: torch.device):
    # MACE must be imported only after from_off has initialized CUDA for the
    # legacy foundation checkpoint used in the supplied environment.
    from mace.calculators import MACECalculator

    class CountingMACECalculator(MACECalculator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calculate_calls = 0

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces", "stress"),
            system_changes=all_changes,
        ) -> None:
            self.calculate_calls += 1
            super().calculate(atoms, properties, system_changes)

    return CountingMACECalculator(
        models=model,
        device=str(device),
        default_dtype="float64",
    )


def run_ase(
    calculator,
    systems: list[Any],
    *,
    fmax: float,
    max_steps: int,
    dt_start: float,
    dt_max: float,
    max_step: float,
    optimizer_name: str,
    alpha: float,
) -> dict[str, Any]:
    calls_before = calculator.calculate_calls
    records = []
    total_steps = 0
    for source in systems:
        atoms = source.copy()
        atoms.calc = calculator
        cell_filter = ASEFrechetCellFilter(atoms)
        if optimizer_name == "fire":
            optimizer = FIRE(
                cell_filter,
                logfile=None,
                trajectory=None,
                dt=dt_start,
                dtmax=dt_max,
                maxstep=max_step,
            )
        else:
            optimizer = BFGS(
                cell_filter,
                logfile=None,
                trajectory=None,
                maxstep=max_step,
                alpha=alpha,
            )
        converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        records.append(
            serialize_record(
                source=source.info["benchmark_source"],
                converged=converged,
                steps=int(optimizer.nsteps),
                energy=float(atoms.get_potential_energy()),
                forces=forces,
                stress=stress,
                positions=atoms.positions,
                cell=atoms.cell.array,
            )
        )
        total_steps += int(optimizer.nsteps)
    evaluations = calculator.calculate_calls - calls_before
    return {
        "records": records,
        "model_evaluations": evaluations,
        "graph_evaluations": evaluations,
        "neighbor_rebuilds": evaluations,
        "optimizer_steps_total": total_steps,
    }


def run_batch(
    calculator: MACEBatchCalculator,
    systems: list[Any],
    *,
    batch_size: int,
    active_compaction: bool,
    fmax: float,
    max_steps: int,
    dt_start: float,
    dt_max: float,
    max_step: float,
    optimizer_name: str,
    alpha: float,
    refill: bool = False,
) -> dict[str, Any]:
    records = []
    model_evaluations = 0
    graph_evaluations = 0
    uncompacted_graph_evaluations = 0
    neighbor_rebuilds = 0
    active_batch_sizes = []

    chunks = [systems] if refill else [
        systems[start : start + batch_size]
        for start in range(0, len(systems), batch_size)
    ]
    for chunk in chunks:
        state = calculator.create_state(chunk, build_neighbors=not refill)
        common = {
            "cell_filter": FrechetCellFilter(),
            "active_compaction": active_compaction,
            "fmax": fmax,
            "smax": None,
            "max_steps": max_steps,
            "max_step": max_step,
            "callback_interval": max_steps + 1,
        }
        if optimizer_name == "fire":
            result = batched_fire_relax(
                state,
                calculator,
                dt_start=dt_start,
                dt_max=dt_max,
                **common,
            )
        else:
            result = batched_bfgs_relax(
                state,
                calculator,
                alpha=alpha,
                optimizer_dtype="float64",
                refill_batch_size=batch_size if refill else None,
                **common,
            )
        model_evaluations += result.model_evaluations
        graph_evaluations += result.graph_evaluations
        uncompacted_graph_evaluations += result.model_evaluations * len(chunk)
        neighbor_rebuilds += result.state.neighbor_rebuild_count
        active_batch_sizes.append(list(result.active_batch_sizes))

        energies = result.evaluation.energy.detach().cpu().numpy()
        forces = result.evaluation.forces.detach().cpu().numpy()
        stresses = result.evaluation.stress.detach().cpu().numpy()
        positions = result.state.positions.detach().cpu().numpy()
        cells = result.state.cells.detach().cpu().numpy()
        converged = result.converged.detach().cpu().numpy()
        converged_steps = result.converged_step.detach().cpu().numpy()
        for local_id, source in enumerate(chunk):
            atom_slice = result.state.atom_slice(local_id)
            converged_step = int(converged_steps[local_id])
            records.append(
                serialize_record(
                    source=source.info["benchmark_source"],
                    converged=bool(converged[local_id]),
                    steps=(
                        converged_step if converged_step >= 0 else result.steps
                    ),
                    energy=float(energies[local_id]),
                    forces=forces[atom_slice],
                    stress=stresses[local_id],
                    positions=positions[atom_slice],
                    cell=cells[local_id],
                )
            )

    return {
        "records": records,
        "model_evaluations": model_evaluations,
        "graph_evaluations": graph_evaluations,
        "uncompacted_graph_evaluations": uncompacted_graph_evaluations,
        "avoided_graph_evaluations": (
            uncompacted_graph_evaluations - graph_evaluations
        ),
        "neighbor_rebuilds": neighbor_rebuilds,
        "optimizer_steps_total": sum(record["steps"] for record in records),
        "active_batch_sizes": active_batch_sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("ase", "masked", "active", "refill"),
        required=True,
    )
    parser.add_argument("--optimizer", choices=("fire", "bfgs"), default="fire")
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument(
        "--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="small")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dt-start", type=float, default=0.1)
    parser.add_argument("--dt-max", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=70.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(args.deterministic)
    if args.method == "refill" and args.optimizer != "bfgs":
        raise ValueError("active refill is currently implemented only for BFGS")
    if args.pool_size <= 0 or args.repeats <= 0:
        raise ValueError("pool size and repeats must be positive")
    if args.method != "ase" and any(
        args.pool_size % size for size in args.batch_sizes
    ):
        raise ValueError("every batch size must divide the fixed pool")

    manifest = load_manifest(args.manifest, min(args.pool_size, 32))
    available_names = manifest["samples"][str(args.atom_count)]
    base_names = available_names[: min(args.pool_size, len(available_names))]
    names = [base_names[i % len(base_names)] for i in range(args.pool_size)]
    systems = []
    for name in names:
        atoms = read(args.dataset_dir / name)
        if len(atoms) != args.atom_count:
            raise ValueError(f"{name} has {len(atoms)} atoms")
        atoms.info["benchmark_source"] = name
        systems.append(atoms)

    device = torch.device(args.device)
    calculator = MACEBatchCalculator.from_off(
        model=args.model,
        device=device,
        dtype=torch.float64,
    )
    ase_calculator = make_counting_ase_calculator(
        calculator.model,
        device=device,
    )
    common = {
        "systems": systems,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
        "dt_start": args.dt_start,
        "dt_max": args.dt_max,
        "max_step": args.max_step,
        "optimizer_name": args.optimizer,
        "alpha": args.alpha,
    }

    result = {
        "schema_version": 1,
        "status": "running",
        "mlip": "mace-off",
        "model": args.model,
        "mace_version": importlib.metadata.version("mace-torch"),
        "method": args.method,
        "optimizer": args.optimizer,
        "atom_count": args.atom_count,
        "pool_size": args.pool_size,
        "sample_files": names,
        "environment": environment_metadata(device),
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
        },
        "parameters": {
            "batch_sizes": args.batch_sizes,
            "cutoff_A": calculator.cutoff,
            "fmax_eV_per_A": args.fmax,
            "smax": None,
            "convergence": "ASE FrechetCellFilter generalized-force fmax",
            "max_steps": args.max_steps,
            "dt_start": args.dt_start,
            "dt_max": args.dt_max,
            "max_step_A": args.max_step,
            "bfgs_alpha_eV_per_A2": args.alpha,
            "cell_filter": "full Frechet log deformation",
            "pressure_GPa": 0.0,
            "dtype": "float64",
            "repeats": args.repeats,
            "deterministic_algorithms": args.deterministic,
        },
        "points": [],
    }
    write_result(args.output, result)

    calculator(calculator.create_state([systems[0]]), compute_stress=True)
    ase_calculator.calculate(
        systems[0],
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    )
    synchronize(device)

    sizes = [None] if args.method == "ase" else args.batch_sizes
    for batch_size in sizes:
        point = {"batch_size": batch_size, "status": "running"}
        result["points"].append(point)
        write_result(args.output, result)
        try:
            if args.method == "ase":

                def fn():
                    return run_ase(ase_calculator, **common)

            else:

                def fn(batch_size=batch_size):
                    return run_batch(
                        calculator,
                        **common,
                        batch_size=batch_size,
                        active_compaction=args.method in ("active", "refill"),
                        refill=args.method == "refill",
                    )

            output, timing, peak_memory = timed_repeats(
                fn,
                repeats=args.repeats,
                device=device,
            )
            point.update(
                {
                    "status": "passed",
                    "timing": timing,
                    "systems_per_second": args.pool_size
                    / timing["median_seconds"],
                    "atoms_per_second": (
                        args.pool_size
                        * args.atom_count
                        / timing["median_seconds"]
                    ),
                    "peak_memory_bytes": peak_memory,
                    **output,
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            point.update({"status": "oom", "error": str(exc)})
            torch.cuda.empty_cache()
        except Exception as exc:
            point.update(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
        write_result(args.output, result)

    result["status"] = "complete"
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "method": args.method,
                "optimizer": args.optimizer,
                "atom_count": args.atom_count,
                "output": str(args.output),
                "point_statuses": [point["status"] for point in result["points"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
