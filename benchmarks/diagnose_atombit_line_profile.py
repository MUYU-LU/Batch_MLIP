#!/usr/bin/env python3
"""Trace real AtomBit Wolfe searches and plot their one-dimensional profiles."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    write_result,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    FrechetCellFilter,
    batched_bfgs_line_search_relax,
)
from batch_mlip.optimization.cell_filters import (  # noqa: E402
    BoundFrechetCellFilter,
)


def _select_steps(
    starts: dict[int, dict[str, object]],
    trials: dict[int, list[dict[str, object]]],
    *,
    high_trial_threshold: int,
) -> dict[str, int]:
    ordered = sorted(starts)
    if not ordered:
        raise RuntimeError("line search produced no directions")
    initial = ordered[0]
    first_high = next(
        (
            step
            for step in ordered[1:]
            if len(trials.get(step, ())) >= high_trial_threshold
        ),
        max(ordered[1:] or ordered, key=lambda step: len(trials.get(step, ()))),
    )
    late_start = max(ordered[-1] - 99, 0)
    late_candidates = [step for step in ordered if step >= late_start]
    stalled = max(late_candidates, key=lambda step: len(trials.get(step, ())))
    if stalled in (initial, first_high):
        alternatives = [step for step in late_candidates if step != first_high]
        if alternatives:
            stalled = max(alternatives, key=lambda step: len(trials.get(step, ())))
    return {"initial": initial, "first_high_trial": first_high, "stalled": stalled}


def _trace_summary(
    starts: dict[int, dict[str, object]],
    trials: dict[int, list[dict[str, object]]],
    final_fmax: float,
) -> list[dict[str, object]]:
    ordered = sorted(starts)
    rows = []
    for index, step in enumerate(ordered):
        step_trials = trials.get(step, [])
        final_trial = step_trials[-1] if step_trials else None
        next_fmax = (
            float(starts[ordered[index + 1]]["physical_fmax"])
            if index + 1 < len(ordered)
            else final_fmax
        )
        start = starts[step]
        rows.append(
            {
                "optimizer_step": step,
                "trials": len(step_trials),
                "initial_directional_derivative_scaled": start["derivative0"],
                "descent_direction": float(start["derivative0"]) < 0.0,
                "fmax_before_eV_per_A": start["physical_fmax"],
                "fmax_after_eV_per_A": next_fmax,
                "accepted_step_size": (
                    None if final_trial is None else final_trial["step_size"]
                ),
                "final_task": None if final_trial is None else final_trial["task"],
            }
        )
    return rows


def _profile_point(
    calculator: AtomBitBatchCalculator,
    atoms: Any,
    snapshot: dict[str, object],
    parameter: float,
) -> dict[str, object]:
    dtype = calculator.dtype
    device = calculator.device
    base = snapshot["base_coordinates"].to(device=device, dtype=dtype).reshape(-1, 3)
    direction = snapshot["direction"].to(device=device, dtype=dtype).reshape(-1, 3)
    coordinate = base + parameter * direction
    atom_count = len(atoms)
    generalized_positions = coordinate[:atom_count]
    cell_factor = torch.tensor(
        [snapshot["cell_factor"]], device=device, dtype=dtype
    )
    log_deformation = (coordinate[atom_count:] / cell_factor[0]).unsqueeze(0)
    reference_cells = snapshot["reference_cell"].to(
        device=device, dtype=dtype
    ).unsqueeze(0)
    cell_filter = BoundFrechetCellFilter(
        reference_cells=reference_cells,
        generalized_positions=generalized_positions,
        log_deformation=log_deformation,
        cell_factor=cell_factor,
        pressure=torch.zeros(1, device=device, dtype=dtype),
        mask=snapshot["cell_mask"].to(device=device),
        hydrostatic_strain=bool(snapshot["hydrostatic_strain"]),
    )
    state = calculator.create_state([atoms], build_neighbors=False)
    deformation = cell_filter.deformation.detach()
    state.cells = cell_filter.current_cells().to(dtype=dtype).detach()
    state.positions = torch.bmm(
        generalized_positions.unsqueeze(1),
        deformation[0].transpose(0, 1).expand(atom_count, -1, -1),
    ).squeeze(1)
    evaluation = calculator(
        state, neighbor_policy="always", compute_stress=True
    )
    atomic_forces, cell_forces = cell_filter.generalized_forces(state, evaluation)
    directional_derivative = -torch.sum(atomic_forces * direction[:atom_count])
    directional_derivative -= torch.sum(cell_forces[0] * direction[atom_count:])
    return {
        "parameter": parameter,
        "energy_eV": float(evaluation.energy[0].item()),
        "directional_derivative_eV": float(directional_derivative.item()),
        "max_force_eV_per_A": float(
            torch.linalg.vector_norm(evaluation.forces, dim=1).max().item()
        ),
        "candidate_edges": int(state.edge_index.shape[1]),
        "physical_edges": int(state.as_model_data().edge_index.shape[1]),
    }


def _sampling_parameters(step_trials: list[dict[str, object]]) -> list[float]:
    actual = [float(trial["step_size"]) for trial in step_trials]
    maximum = max(actual or [1.0])
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"invalid line-search range {maximum}")
    linear = np.linspace(0.0, maximum * 1.05, 161)
    logarithmic = np.geomspace(max(maximum * 1e-8, 1e-12), maximum, 81)
    return sorted({0.0, *actual, *linear.tolist(), *logarithmic.tolist()})


def _scan_profiles(
    checkpoint: Path,
    atoms: Any,
    snapshots: dict[str, dict[str, object]],
    trials: dict[int, list[dict[str, object]]],
    selected: dict[str, int],
    *,
    device: torch.device,
    cutoff: float,
    skin: float,
) -> dict[str, object]:
    profiles: dict[str, object] = {}
    for dtype_name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        model, _ = load_production_model(checkpoint)
        model = model.to(device=device, dtype=dtype).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=cutoff,
            skin=skin,
            device=device,
            dtype=dtype,
            force_mode="autograd",
        )
        dtype_profiles = {}
        for label, step in selected.items():
            parameters = _sampling_parameters(trials.get(step, []))
            points = [
                _profile_point(calculator, atoms, snapshots[label], value)
                for value in parameters
            ]
            energy0 = points[0]["energy_eV"]
            derivative0 = points[0]["directional_derivative_eV"]
            for point in points:
                parameter = point["parameter"]
                point["armijo_boundary_eV"] = energy0 + 0.23 * parameter * derivative0
                point["armijo_pass"] = (
                    point["energy_eV"] <= point["armijo_boundary_eV"]
                )
                point["curvature_ratio"] = abs(
                    point["directional_derivative_eV"]
                ) / max(abs(derivative0), 1e-30)
                point["curvature_pass"] = point["curvature_ratio"] <= 0.46
                point["strong_wolfe_pass"] = (
                    point["armijo_pass"] and point["curvature_pass"]
                )
            dtype_profiles[label] = {
                "optimizer_step": step,
                "trial_step_sizes": [
                    float(trial["step_size"]) for trial in trials.get(step, [])
                ],
                "trial_tasks": [trial["task"] for trial in trials.get(step, [])],
                "points": points,
            }
        profiles[dtype_name] = dtype_profiles
        del calculator, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return profiles


def _write_plots(profiles: dict[str, object], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    labels = next(iter(profiles.values())).keys()
    for label in labels:
        figure, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
        for dtype_name, color in (("float64", "#155e75"), ("float32", "#b45309")):
            profile = profiles[dtype_name][label]
            points = profile["points"]
            parameter = np.asarray([point["parameter"] for point in points])
            energy = np.asarray([point["energy_eV"] for point in points])
            armijo = np.asarray([point["armijo_boundary_eV"] for point in points])
            curvature = np.asarray([point["curvature_ratio"] for point in points])
            edges = np.asarray([point["physical_edges"] for point in points])
            axes[0].plot(parameter, energy - energy[0], color=color, label=dtype_name)
            axes[0].plot(
                parameter,
                armijo - energy[0],
                color=color,
                linestyle="--",
                alpha=0.65,
            )
            axes[1].plot(parameter, curvature, color=color, label=dtype_name)
            axes[2].step(parameter, edges, where="mid", color=color, label=dtype_name)
            for trial in profile["trial_step_sizes"]:
                for axis in axes:
                    axis.axvline(trial, color=color, alpha=0.08, linewidth=0.8)
        axes[0].set_ylabel("E(t) - E(0) [eV]")
        axes[0].legend()
        axes[1].axhline(0.46, color="black", linestyle=":", label="Wolfe c2")
        axes[1].set_ylabel("|E'(t)| / |E'(0)|")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[2].set_ylabel("Physical edges")
        axes[2].set_xlabel("Line parameter t")
        axes[2].legend()
        figure.suptitle(
            f"AtomBit BFGSLineSearch: {label.replace('_', ' ')} direction"
        )
        figure.tight_layout()
        path = output_dir / f"{label}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        files.append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--high-trial-threshold", type=int, default=20)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--plot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-trace", action="store_true")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    atoms = read(args.structure)
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    trace_path = args.output.with_suffix(".trace.pt")
    if args.resume_trace:
        if not trace_path.is_file():
            raise FileNotFoundError(f"trace checkpoint does not exist: {trace_path}")
        checkpoint = torch.load(trace_path, map_location="cpu", weights_only=False)
        starts = checkpoint["starts"]
        trials = checkpoint["trials"]
        selected = checkpoint["selected"]
        trace_rows = checkpoint["trace_rows"]
        optimization = checkpoint["optimization"]
        del model
    else:
        model = model.to(device=device, dtype=torch.float64).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=args.cutoff,
            skin=args.skin,
            device=device,
            dtype=torch.float64,
            force_mode="autograd",
        )
        raw_events: list[dict[str, object]] = []
        result = batched_bfgs_line_search_relax(
            calculator.create_state([atoms]),
            calculator,
            cell_filter=FrechetCellFilter(),
            active_compaction=True,
            fmax=args.fmax,
            smax=None,
            max_steps=args.max_steps,
            max_step=0.2,
            alpha=10.0,
            optimizer_dtype="float64",
            callback_interval=args.max_steps + 1,
            trace_callback=raw_events.append,
        )
        starts = {
            int(event["optimizer_step"]): event
            for event in raw_events
            if event["event"] == "search_start"
        }
        trials: dict[int, list[dict[str, object]]] = defaultdict(list)
        for event in raw_events:
            if event["event"] == "trial":
                trials[int(event["optimizer_step"])].append(event)
        selected = _select_steps(
            starts, trials, high_trial_threshold=args.high_trial_threshold
        )
        final_fmax = float(result.max_force[0].item())
        trace_rows = _trace_summary(starts, trials, final_fmax)
        task_counts = Counter(
            str(row["final_task"]).split(":", 1)[0] for row in trace_rows
        )
        trial_counts = np.asarray([row["trials"] for row in trace_rows], dtype=float)
        optimization = {
            "converged": bool(result.converged[0].item()),
            "steps": result.steps,
            "model_evaluations": result.model_evaluations,
            "final_fmax_eV_per_A": final_fmax,
            "trial_count_min": int(trial_counts.min()),
            "trial_count_median": float(np.median(trial_counts)),
            "trial_count_max": int(trial_counts.max()),
            "non_descent_directions": sum(
                not bool(row["descent_direction"]) for row in trace_rows
            ),
            "final_task_prefix_counts": dict(task_counts),
        }
        torch.save(
            {
                "starts": starts,
                "trials": dict(trials),
                "selected": selected,
                "trace_rows": trace_rows,
                "optimization": optimization,
            },
            trace_path,
        )
        del calculator, model, result, raw_events

    snapshots = {label: starts[step] for label, step in selected.items()}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    profiles = _scan_profiles(
        args.checkpoint,
        atoms,
        snapshots,
        trials,
        selected,
        device=device,
        cutoff=args.cutoff,
        skin=args.skin,
    )
    plot_files = _write_plots(profiles, args.plot_dir)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "structure": args.structure.name,
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "parameters": {
            "trace_dtype": "float64",
            "profile_dtypes": ["float32", "float64"],
            "fmax_eV_per_A": args.fmax,
            "max_steps": args.max_steps,
            "alpha": 10.0,
            "c1": 0.23,
            "c2": 0.46,
            "max_step_A": 0.2,
            "cutoff_A": args.cutoff,
            "skin_A": args.skin,
            "deterministic_algorithms": True,
        },
        "optimization": optimization,
        "selected_steps": selected,
        "trace": trace_rows,
        "profiles": profiles,
        "plot_files": plot_files,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
