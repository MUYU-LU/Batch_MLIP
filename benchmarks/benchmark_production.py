"""Benchmark production AtomBit batches against sequential ASE inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
import types
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.io import read

from batch_mlip import AseGraphBatch, AtomBitBatchCalculator
from src.Calculator import AtomBitCalculator
from src.model import AtomBitModel
from src.utils import AtomBitConfig

ATOM_LINE = re.compile(r"^\s*\d+\s+[A-Z][a-z]?\s+", re.MULTILINE)


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, relative_path: str) -> str:
    return hashlib.sha256(f"{seed}:{relative_path}".encode()).hexdigest()


def count_cif_atom_rows(path: Path) -> int:
    """Count rows in the simple P1 atom loop used by the T2 packet."""

    return len(ATOM_LINE.findall(path.read_text(encoding="utf-8")))


def create_sample_manifest(
    dataset_dir: Path,
    output: Path,
    atom_counts: list[int],
    samples_per_size: int,
    seed: int,
) -> dict[str, Any]:
    candidates: dict[int, list[str]] = {count: [] for count in atom_counts}
    distribution: Counter[int] = Counter()
    for path in sorted(dataset_dir.glob("*.cif")):
        atom_count = count_cif_atom_rows(path)
        distribution[atom_count] += 1
        if atom_count in candidates:
            candidates[atom_count].append(path.name)

    selected: dict[str, list[str]] = {}
    for atom_count in atom_counts:
        ranked = sorted(candidates[atom_count], key=lambda name: stable_rank(seed, name))
        if len(ranked) < samples_per_size:
            raise ValueError(
                f"requested {samples_per_size} samples with {atom_count} atoms, "
                f"but found {len(ranked)}"
            )
        selected[str(atom_count)] = ranked[:samples_per_size]

    payload = {
        "schema_version": 1,
        "selection": "sha256(seed:filename), ascending",
        "seed": seed,
        "samples_per_atom_count": samples_per_size,
        "atom_counts": atom_counts,
        "dataset_distribution": dict(sorted(distribution.items())),
        "samples": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_manifest(path: Path, required_batch_size: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for atom_count in payload["atom_counts"]:
        if len(payload["samples"][str(atom_count)]) < required_batch_size:
            raise ValueError(f"manifest has too few {atom_count}-atom samples")
    return payload


def install_legacy_config_alias() -> None:
    """Support checkpoints pickled with the old src.utils.Utils module path."""

    module = types.ModuleType("src.utils.Utils")
    module.AtomBitConfig = AtomBitConfig
    sys.modules["src.utils.Utils"] = module


def load_production_model(checkpoint: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    install_legacy_config_alias()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("production checkpoint must be a dictionary")
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise KeyError("checkpoint must contain model_config and model_state_dict")

    config = payload["model_config"]
    model = AtomBitModel(config)
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    metadata = {
        "epoch": payload.get("epoch"),
        "label_mode": payload.get("label_mode"),
        "precision_dtype": payload.get("precision_dtype"),
        "model_config": dict(vars(config)),
        "state_tensor_count": len(state_dict),
    }
    metadata["model_config"]["active_paths"] = {
        str(key): value for key, value in config.active_paths.items()
    }
    return model, metadata


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(
    fn: Callable[[], Any],
    repeats: int,
    warmup: int,
    device: torch.device,
) -> tuple[list[float], int | None]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return samples, peak


def timing_summary(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "std_seconds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def evaluate_ase(
    calculator: AtomBitCalculator,
    systems: list[Any],
    forces: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    energies = []
    force_blocks = []
    properties = ("energy", "forces") if forces else ("energy",)
    for atoms in systems:
        calculator.calculate(atoms, properties=properties, system_changes=all_changes)
        energies.append(float(calculator.results["energy"]))
        if forces:
            force_blocks.append(np.asarray(calculator.results["forces"]).copy())
    return np.asarray(energies), force_blocks


def raw_model_forward(model: torch.nn.Module, state: AseGraphBatch) -> torch.Tensor:
    with torch.no_grad():
        output = model(state.as_model_data())
    return output["energy"] if isinstance(output, dict) else output


def transfer_state_tensors(state: AseGraphBatch, device: torch.device) -> list[torch.Tensor]:
    fields = (
        state.z,
        state.positions,
        state.cells,
        state.pbc,
        state.system_idx,
        state.ptr,
        state.masses,
        state.fixed,
        state.velocities,
        state.edge_index,
        state.shifts_int,
    )
    return [tensor.to(device=device) for tensor in fields]


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ase": ase.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return metadata


def benchmark_group(
    *,
    atom_count: int,
    systems: list[Any],
    batch_sizes: list[int],
    model: torch.nn.Module,
    device: torch.device,
    cutoff: float,
    repeats: int,
    reference_repeats: int,
    warmup: int,
    energy_atol: float,
    energy_per_atom_atol: float,
    force_atol: float,
    result: dict[str, Any],
    output: Path,
) -> None:
    calculator = AtomBitCalculator(
        model,
        cutoff=cutoff,
        device=device,
        enable_stress=False,
        add_e0=False,
    )
    potential = AtomBitBatchCalculator(
        model,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )

    reference_energies, reference_forces = evaluate_ase(calculator, systems, forces=True)
    reference_energy_samples, reference_energy_peak = timed(
        lambda: evaluate_ase(calculator, systems, forces=False),
        reference_repeats,
        warmup,
        device,
    )
    reference_force_samples, reference_force_peak = timed(
        lambda: evaluate_ase(calculator, systems, forces=True),
        reference_repeats,
        warmup,
        device,
    )
    reference_energy = timing_summary(reference_energy_samples)
    reference_force = timing_summary(reference_force_samples)
    group_result: dict[str, Any] = {
        "atom_count": atom_count,
        "sample_count": len(systems),
        "sample_files": [atoms.info["benchmark_source"] for atoms in systems],
        "points": [],
    }
    result["groups"][str(atom_count)] = group_result

    for batch_size in batch_sizes:
        if len(systems) % batch_size:
            raise ValueError(
                f"fixed pool of {len(systems)} is not divisible by batch size {batch_size}"
            )
        chunks = [
            systems[start : start + batch_size]
            for start in range(0, len(systems), batch_size)
        ]
        point: dict[str, Any] = {"batch_size": batch_size, "status": "running"}
        group_result["points"].append(point)
        write_result(output, result)
        try:
            states = [
                AseGraphBatch.from_ase(
                    chunk,
                    cutoff=cutoff,
                    skin=0.0,
                    device=device,
                    dtype=torch.float32,
                )
                for chunk in chunks
            ]
            evaluations = [
                potential(state, neighbor_policy="never") for state in states
            ]
            synchronize(device)
            batch_energy = np.concatenate(
                [evaluation.energy.cpu().numpy() for evaluation in evaluations]
            )
            batch_forces = np.concatenate(
                [evaluation.forces.cpu().numpy() for evaluation in evaluations], axis=0
            )
            expected_forces = np.concatenate(reference_forces, axis=0)
            energy_error = np.abs(batch_energy - reference_energies)
            force_error = np.abs(batch_forces - expected_forces)
            max_energy_error = float(energy_error.max())
            max_energy_error_per_atom = max_energy_error / atom_count
            energy_passed = bool(
                np.allclose(
                    batch_energy,
                    reference_energies,
                    atol=energy_atol,
                    rtol=1e-6,
                )
                or max_energy_error_per_atom <= energy_per_atom_atol
            )
            force_passed = bool(
                np.allclose(
                    batch_forces,
                    expected_forces,
                    atol=force_atol,
                    rtol=1e-5,
                )
            )
            validation = {
                "max_abs_energy_error_ev": max_energy_error,
                "max_abs_energy_error_ev_per_atom": max_energy_error_per_atom,
                "max_abs_force_error_ev_per_a": float(force_error.max()),
                "energy_atol_ev": energy_atol,
                "energy_per_atom_atol_ev": energy_per_atom_atol,
                "force_atol_ev_per_a": force_atol,
                "energy_passed": energy_passed,
                "force_passed": force_passed,
                "passed": energy_passed and force_passed,
            }

            forward_samples, forward_peak = timed(
                lambda states=states: [
                    raw_model_forward(model, state) for state in states
                ],
                repeats,
                warmup,
                device,
            )
            force_samples, force_peak = timed(
                lambda states=states: [
                    potential(state, neighbor_policy="never") for state in states
                ],
                repeats,
                warmup,
                device,
            )
            end_to_end_samples, end_to_end_peak = timed(
                lambda chunks=chunks: [
                    potential(
                        AseGraphBatch.from_ase(
                            chunk,
                            cutoff=cutoff,
                            skin=0.0,
                            device=device,
                            dtype=torch.float32,
                        ),
                        neighbor_policy="never",
                    )
                    for chunk in chunks
                ],
                repeats,
                1,
                device,
            )
            cpu_build_samples, _ = timed(
                lambda chunks=chunks: [
                    AseGraphBatch.from_ase(
                        chunk,
                        cutoff=cutoff,
                        skin=0.0,
                        device="cpu",
                        dtype=torch.float32,
                    )
                    for chunk in chunks
                ],
                repeats,
                0,
                torch.device("cpu"),
            )
            cpu_states = [
                AseGraphBatch.from_ase(
                    chunk,
                    cutoff=cutoff,
                    skin=0.0,
                    device="cpu",
                    dtype=torch.float32,
                )
                for chunk in chunks
            ]
            transfer_samples, _ = timed(
                lambda cpu_states=cpu_states: [
                    transfer_state_tensors(cpu_state, device)
                    for cpu_state in cpu_states
                ],
                repeats,
                warmup,
                device,
            )

            forward = timing_summary(forward_samples)
            force = timing_summary(force_samples)
            end_to_end = timing_summary(end_to_end_samples)
            total_edges = sum(int(state.edge_index.shape[1]) for state in states)
            max_batch_edges = max(int(state.edge_index.shape[1]) for state in states)
            point.update(
                {
                    "status": "passed" if validation["passed"] else "validation_failed",
                    "sample_count": len(systems),
                    "total_atoms_processed": atom_count * len(systems),
                    "atoms_per_batch": atom_count * batch_size,
                    "total_edges_processed": total_edges,
                    "max_edges_per_batch": max_batch_edges,
                    "validation": validation,
                    "batch_energy_forward": forward,
                    "batch_energy_force": force,
                    "batch_end_to_end_energy_force": end_to_end,
                    "ase_sequential_energy": reference_energy,
                    "ase_sequential_energy_force": reference_force,
                    "cpu_batch_build": timing_summary(cpu_build_samples),
                    "host_to_device_transfer": timing_summary(transfer_samples),
                    "batch_energy_peak_memory_bytes": forward_peak,
                    "batch_force_peak_memory_bytes": force_peak,
                    "batch_end_to_end_peak_memory_bytes": end_to_end_peak,
                    "ase_energy_peak_memory_bytes": reference_energy_peak,
                    "ase_force_peak_memory_bytes": reference_force_peak,
                    "model_only_energy_speedup_vs_ase_end_to_end": (
                        reference_energy["median_seconds"] / forward["median_seconds"]
                    ),
                    "model_only_force_speedup_vs_ase_end_to_end": (
                        reference_force["median_seconds"] / force["median_seconds"]
                    ),
                    "end_to_end_force_speedup_vs_ase": (
                        reference_force["median_seconds"]
                        / end_to_end["median_seconds"]
                    ),
                    "systems_per_second_model_energy_force": (
                        len(systems) / force["median_seconds"]
                    ),
                    "atoms_per_second_model_energy_force": (
                        (atom_count * len(systems)) / force["median_seconds"]
                    ),
                    "systems_per_second_end_to_end_energy_force": (
                        len(systems) / end_to_end["median_seconds"]
                    ),
                    "atoms_per_second_end_to_end_energy_force": (
                        (atom_count * len(systems)) / end_to_end["median_seconds"]
                    ),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            point.update({"status": "oom", "error": str(error)})
        except Exception as error:
            point.update(
                {"status": "error", "error": f"{type(error).__name__}: {error}"}
            )
        finally:
            write_result(output, result)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument("--create-manifest", action="store_true")
    parser.add_argument("--atom-counts", type=parse_int_list, default=[46, 92, 184, 276])
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--selection-seed", type=int, default=20260717)
    parser.add_argument("--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt"))
    parser.add_argument("--e0", type=Path, default=Path("../AtomBit-OMC-s/meta_e0_data_OMC_r6_single.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference-repeats", type=int, default=3)
    parser.add_argument("--energy-atol", type=float, default=1e-5)
    parser.add_argument("--energy-per-atom-atol", type=float, default=5e-7)
    parser.add_argument("--force-atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("runs/production_batch_scaling.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    max_batch_size = max(args.batch_sizes)
    if args.create_manifest:
        payload = create_sample_manifest(
            args.dataset_dir,
            args.manifest,
            args.atom_counts,
            max_batch_size,
            args.selection_seed,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    manifest = load_manifest(args.manifest, max_batch_size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()

    # Validate the residual model itself. E0 is coordinate independent and is
    # intentionally excluded from all timings and batch/single error checks.
    e0_payload = torch.load(args.e0, map_location="cpu", weights_only=False)
    e0_dict = {int(key): float(value) for key, value in e0_payload["e0_dict"].items()}
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "hypothesis": (
            "True graph batching increases production energy/force throughput over "
            "sequential ASE inference, with gains depending on batch size and atom count."
        ),
        "timing_scope": {
            "primary": "energy plus autograd forces",
            "workload": (
                "the same fixed 64 structures per atom count are processed at every "
                "batch size"
            ),
            "batch_model": "prebuilt neighbor list, GPU model forward and autograd",
            "ase_reference": "per-system CPU neighbor list, transfer, forward and autograd",
            "cpu_batch_build": "host tensor packing and per-system neighbor lists",
            "host_to_device_transfer": "batched state tensors only",
            "e0": "excluded because it is coordinate-independent constant bookkeeping",
        },
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "e0": {
            "path": str(args.e0),
            "sha256": sha256_file(args.e0),
            "values": e0_dict,
        },
        "sample_manifest": {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
            "seed": manifest["seed"],
        },
        "parameters": {
            "atom_counts": args.atom_counts,
            "batch_sizes": args.batch_sizes,
            "cutoff": args.cutoff,
            "dtype": "float32",
            "force_mode": "autograd",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "reference_repeats": args.reference_repeats,
            "energy_atol_ev": args.energy_atol,
            "energy_per_atom_atol_ev": args.energy_per_atom_atol,
            "force_atol_ev_per_a": args.force_atol,
        },
        "groups": {},
    }
    write_result(args.output, result)

    for atom_count in args.atom_counts:
        names = manifest["samples"][str(atom_count)][:max_batch_size]
        systems = []
        for name in names:
            atoms = read(args.dataset_dir / name)
            if len(atoms) != atom_count:
                raise ValueError(f"{name} has {len(atoms)} atoms, expected {atom_count}")
            atoms.info["benchmark_source"] = name
            systems.append(atoms)
        benchmark_group(
            atom_count=atom_count,
            systems=systems,
            batch_sizes=args.batch_sizes,
            model=model,
            device=device,
            cutoff=args.cutoff,
            repeats=args.repeats,
            reference_repeats=args.reference_repeats,
            warmup=args.warmup,
            energy_atol=args.energy_atol,
            energy_per_atom_atol=args.energy_per_atom_atol,
            force_atol=args.force_atol,
            result=result,
            output=args.output,
        )

    statuses = [
        point["status"]
        for group in result["groups"].values()
        for point in group["points"]
    ]
    result["status"] = "passed" if statuses and all(s == "passed" for s in statuses) else "completed_with_failures"
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
