"""Command-line interface for configured batched simulations."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from ase import Atoms
from ase.io import read, write

from ..core.state import AseGraphBatch
from ..dynamics.integrators import (
    batched_langevin_baoab,
    batched_velocity_verlet,
    initialize_maxwell_boltzmann,
)
from ..models.loaders import build_model, infer_cutoff, load_e0, parse_dtype
from ..models.potential import AtomBitBatchCalculator
from ..optimization.cell_filters import FrechetCellFilter
from ..optimization.registry import create_optimizer
from .config import load_yaml, required
from .reporting import build_reporter


def _as_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return dict(value)


def _prepare(config: Mapping[str, Any]):
    runtime = _as_mapping(config.get("runtime"), "runtime")
    model_cfg = _as_mapping(required(config, "model", "config"), "model")

    device = torch.device(runtime.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device requests CUDA, but torch.cuda.is_available() is false")
    dtype = parse_dtype(runtime.get("dtype", "float64"))

    model = build_model(
        str(required(model_cfg, "factory", "model")),
        _as_mapping(model_cfg.get("kwargs"), "model.kwargs"),
    )
    cutoff = infer_cutoff(model, model_cfg.get("cutoff"))
    e0_dict = load_e0(model_cfg.get("e0"))
    potential = AtomBitBatchCalculator(
        model,
        device=device,
        dtype=dtype,
        force_mode=str(model_cfg.get("force_mode", "autograd")),
        e0_dict=e0_dict,
        model_call_kwargs=_as_mapping(model_cfg.get("call_kwargs"), "model.call_kwargs"),
        cutoff=cutoff,
        skin=float(runtime.get("skin", 0.0)),
        neighbor_backend=str(runtime.get("neighbor_backend", "auto")),
    )

    input_file = Path(str(required(config, "input", "config")))
    systems = read(input_file, index=":")
    if isinstance(systems, Atoms):
        systems = [systems]
    if not systems:
        raise RuntimeError(f"no structures found in {input_file}")

    state = AseGraphBatch.from_ase(
        systems,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        skin=float(runtime.get("skin", 0.0)),
        neighbor_backend=str(runtime.get("neighbor_backend", "auto")),
    )
    return state, potential, model, input_file


def _reporter_from_config(config: Mapping[str, Any]):
    reporting = _as_mapping(config.get("reporting"), "reporting")
    return build_reporter(
        trajectory=reporting.get("trajectory"),
        diagnostics=reporting.get("diagnostics"),
        checkpoint=reporting.get("checkpoint"),
        wrap=bool(reporting.get("wrap", False)),
    )


def _write_final(output: Path, state, evaluation, extras: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = state.to_ase(evaluation, wrap=bool(extras.get("wrap", True)))
    for system_id, atoms in enumerate(frames):
        atoms.info["system_id"] = system_id
        for key, values in extras.items():
            if key == "wrap":
                continue
            if isinstance(values, torch.Tensor) and values.ndim == 1:
                atoms.info[key] = values[system_id].detach().cpu().item()
    write(output, frames, format="extxyz")


def _summary_base(config: Mapping[str, Any], state, model, input_file: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": config.get("task"),
        "input": str(input_file),
        "n_systems": state.n_systems,
        "n_atoms": state.n_atoms,
        "counts": state.counts.detach().cpu().tolist(),
        "cutoff": state.cutoff,
        "skin": state.skin,
        "neighbor_backend": state.neighbor_backend,
        "neighbor_rebuild_count": state.neighbor_rebuild_count,
        "device": str(state.device),
        "dtype": str(state.dtype),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }


def run_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    task = str(required(config, "task", "config")).lower()
    output = Path(str(required(config, "output", "config")))

    started = time.perf_counter()
    state, potential, model, input_file = _prepare(config)
    reporter = _reporter_from_config(config)

    extras: dict[str, Any] = {"wrap": True}
    if task == "relax":
        options = _as_mapping(config.get("relax"), "relax")
        optimizer = str(options.pop("optimizer", "fire")).lower()
        cell_filter_cfg = options.pop("cell_filter", None)
        if cell_filter_cfg is not None:
            cell_options = _as_mapping(cell_filter_cfg, "relax.cell_filter")
            filter_type = str(cell_options.pop("type", "frechet")).lower()
            if filter_type != "frechet":
                raise ValueError(
                    f"unsupported relaxation cell filter {filter_type!r}"
                )
            options["cell_filter"] = FrechetCellFilter(**cell_options)
        result = create_optimizer(optimizer).run(
            state,
            potential,
            callback=reporter,
            **options,
        )
        extras.update(
            {
                "relax_converged": result.converged,
                "relax_converged_step": result.converged_step,
                "relax_fmax_eV_per_A": result.max_force,
            }
        )
        if result.max_stress is not None:
            extras["relax_smax_eV_per_A3"] = result.max_stress
        evaluation = result.evaluation
        steps = result.steps
        result_summary = {
            "converged": result.converged.detach().cpu().tolist(),
            "converged_step": result.converged_step.detach().cpu().tolist(),
            "max_force": result.max_force.detach().cpu().tolist(),
            "max_stress": (
                None
                if result.max_stress is None
                else result.max_stress.detach().cpu().tolist()
            ),
            "model_evaluations": result.model_evaluations,
            "graph_evaluations": result.graph_evaluations,
            "active_batch_sizes": list(result.active_batch_sizes),
        }
    elif task in ("nve", "nvt_langevin", "nvt"):
        options = _as_mapping(config.get("md"), "md")
        initialize = bool(options.pop("initialize_velocities", True))
        initial_temperature = options.pop(
            "initial_temperature_K", options.get("temperature_K", 300.0)
        )
        if initialize:
            initialize_maxwell_boltzmann(
                state,
                initial_temperature,
                seed=options.pop("initialization_seed", 1234),
                remove_com=bool(options.pop("remove_initial_com", True)),
                force_exact_temperature=bool(
                    options.pop("force_exact_initial_temperature", True)
                ),
            )
        if task == "nve":
            # temperature_K and friction are thermostat-only keys.
            options.pop("temperature_K", None)
            options.pop("friction_per_fs", None)
            result = batched_velocity_verlet(
                state, potential, callback=reporter, **options
            )
        else:
            result = batched_langevin_baoab(
                state, potential, callback=reporter, **options
            )
        evaluation = result.evaluation
        steps = result.steps
        extras.update(
            {
                "kinetic_energy_eV": result.kinetic_energy,
                "temperature_K": result.temperature,
            }
        )
        result_summary = {
            "kinetic_energy": result.kinetic_energy.detach().cpu().tolist(),
            "temperature": result.temperature.detach().cpu().tolist(),
        }
    else:
        raise ValueError("task must be one of: relax, nve, nvt_langevin")

    _write_final(output, state, evaluation, extras)
    elapsed = time.perf_counter() - started

    summary = _summary_base(config, state, model, input_file)
    summary.update(
        {
            "output": str(output),
            "steps": int(steps),
            "wall_seconds": elapsed,
            "neighbor_rebuild_count": state.neighbor_rebuild_count,
            "final_energy": evaluation.energy.detach().cpu().tolist(),
            **result_summary,
        }
    )

    reporting = _as_mapping(config.get("reporting"), "reporting")
    summary_path = Path(
        str(reporting.get("summary", output.with_suffix(output.suffix + ".summary.json")))
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def validate_batch(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    state, potential, model, input_file = _prepare(config)
    batch_eval = potential(state, neighbor_policy="auto")

    single_energies: list[float] = []
    single_forces: list[torch.Tensor] = []
    frames = state.to_ase(evaluation=None, wrap=False)
    for atoms in frames:
        single_state = AseGraphBatch.from_ase(
            [atoms],
            cutoff=state.cutoff,
            device=state.device,
            dtype=state.dtype,
            skin=state.skin,
        )
        single_eval = potential(single_state, neighbor_policy="auto")
        single_energies.append(float(single_eval.energy[0].cpu()))
        single_forces.append(single_eval.forces.cpu())

    reference_energy = torch.tensor(single_energies, dtype=batch_eval.energy.cpu().dtype)
    reference_forces = torch.cat(single_forces, dim=0)
    energy_error = torch.max(torch.abs(batch_eval.energy.cpu() - reference_energy)).item()
    force_error = torch.max(torch.abs(batch_eval.forces.cpu() - reference_forces)).item()

    result = {
        "input": str(input_file),
        "n_systems": state.n_systems,
        "n_atoms": state.n_atoms,
        "max_abs_energy_error": energy_error,
        "max_abs_force_error": force_error,
        "cross_system_edges": False,
        "passed": bool(energy_error < 1e-9 and force_error < 1e-8),
    }
    output = Path(str(config.get("validation_output", "validation.json")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def make_demo_data(output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]),
        Atoms(
            "H3",
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 1.4, 0.0]],
        ),
        Atoms(
            "He4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.3, 0.0, 0.0],
                [0.0, 1.3, 0.0],
                [1.3, 1.3, 0.0],
            ],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        ),
    ]
    write(output, frames, format="extxyz")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="batch-mlip")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a YAML-configured simulation")
    run_parser.add_argument("config")

    validate_parser = subparsers.add_parser(
        "validate", help="compare batched and single-system predictions"
    )
    validate_parser.add_argument("config")

    demo_parser = subparsers.add_parser("make-demo", help="write demo extxyz structures")
    demo_parser.add_argument("output", nargs="?", default="data/demo.extxyz")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_config(args.config)
    elif args.command == "validate":
        result = validate_batch(args.config)
    elif args.command == "make-demo":
        make_demo_data(args.output)
        result = {"output": args.output}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
