#!/usr/bin/env python3
"""Test AtomBit hard-degree cutoff effects across independent structures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    write_result,
)
from diagnose_atombit_degree_discontinuity import (  # noqa: E402
    _edge_distances,
    _edge_keys,
    _energy,
    _fixed_degree,
    _force_effect,
    _inverse_sqrt_degree,
)

from batch_mlip import AtomBitBatchCalculator  # noqa: E402
from batch_mlip.core.types import GraphData  # noqa: E402


def _without_keys(
    data: GraphData,
    removed_keys: set[tuple[int, int, int, int, int]],
) -> GraphData:
    keys = _edge_keys(data)
    keep = torch.tensor(
        [key not in removed_keys for key in keys],
        device=data.edge_index.device,
        dtype=torch.bool,
    )
    return GraphData(
        z=data.z,
        pos=data.pos,
        cell=data.cell,
        edge_index=data.edge_index[:, keep],
        shifts_int=data.shifts_int[keep],
        batch=data.batch,
        num_graphs=data.num_graphs,
    )


def _reverse_key(
    key: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    center, neighbor, shift_x, shift_y, shift_z = key
    return neighbor, center, -shift_x, -shift_y, -shift_z


def _probe_structure(
    path: Path,
    calculator: AtomBitBatchCalculator,
    model: torch.nn.Module,
    cutoff: float,
    cutoff_offset: float,
) -> dict[str, object]:
    atoms = read(path)
    initial = calculator.create_state([atoms]).as_model_data()
    distances = _edge_distances(initial)
    if distances.numel() == 0:
        raise RuntimeError(f"{path.name} has no physical neighbor edges")
    selected_index = int(torch.argmax(distances).item())
    selected_distance = float(distances[selected_index].item())
    selected_key = _edge_keys(initial)[selected_index]
    reverse_key = _reverse_key(selected_key)

    target_distance = cutoff - cutoff_offset
    scale = target_distance / selected_distance
    strained_atoms = atoms.copy()
    strained_atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    present = calculator.create_state([strained_atoms]).as_model_data()
    present_keys = set(_edge_keys(present))
    removed_keys = {selected_key, reverse_key}
    missing = removed_keys - present_keys
    if missing:
        raise RuntimeError(
            f"target edges missing after strain for {path.name}: {missing}"
        )
    removed = _without_keys(present, removed_keys)
    if present.edge_index.shape[1] - removed.edge_index.shape[1] != 2:
        raise RuntimeError(f"expected two directed target edges for {path.name}")

    present_distances = _edge_distances(present)
    present_lookup = {
        key: index for index, key in enumerate(_edge_keys(present))
    }
    target_distances = [
        float(present_distances[present_lookup[key]].item())
        for key in sorted(removed_keys)
    ]
    x = target_distances[0] / cutoff
    envelope = 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)

    energy_present = _energy(model, present)
    energy_removed = _energy(model, removed)
    normal_force = _force_effect(model, present, removed)
    fixed_degree = _inverse_sqrt_degree(present)
    with _fixed_degree(model, fixed_degree):
        fixed_energy_present = _energy(model, present)
        fixed_energy_removed = _energy(model, removed)
        fixed_force = _force_effect(model, present, removed)

    center, neighbor = selected_key[:2]
    degrees = torch.bincount(
        present.edge_index[0], minlength=present.z.shape[0]
    )
    return {
        "source": path.name,
        "atom_count": len(atoms),
        "uniform_length_scale": scale,
        "absolute_strain": abs(scale - 1.0),
        "selected_edge": list(selected_key),
        "selected_edge_reverse": list(reverse_key),
        "selected_distance_before_strain_A": selected_distance,
        "target_distances_A": target_distances,
        "target_envelope": envelope,
        "physical_edges": int(present.edge_index.shape[1]),
        "affected_degrees_before_removal": {
            str(center): int(degrees[center].item()),
            str(neighbor): int(degrees[neighbor].item()),
        },
        "normal": {
            "energy_effect_removed_minus_present_eV": (
                energy_removed - energy_present
            ),
            **normal_force,
        },
        "fixed_degree": {
            "energy_effect_removed_minus_present_eV": (
                fixed_energy_removed - fixed_energy_present
            ),
            **fixed_force,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples-per-size", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("native", "float32", "float64"),
        default="native",
    )
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--cutoff-offset", type=float, default=1e-8)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.samples_per_size <= 0:
        raise ValueError("samples-per-size must be positive")
    if not 0.0 < args.cutoff_offset < args.cutoff:
        raise ValueError("cutoff-offset must lie between zero and cutoff")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    native_dtype = next(model.parameters()).dtype
    dtype = {
        "native": native_dtype,
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.dtype]
    model = model.to(device=device, dtype=dtype).eval()
    calculator = AtomBitBatchCalculator(
        model,
        cutoff=args.cutoff,
        skin=args.skin,
        device=device,
        dtype=dtype,
        force_mode="autograd",
    )
    manifest = json.loads(args.manifest.read_text())
    selected = []
    for atom_count in manifest["atom_counts"]:
        names = manifest["samples"][str(atom_count)][: args.samples_per_size]
        selected.extend(args.dataset_dir / name for name in names)

    probes = [
        _probe_structure(
            path,
            calculator,
            model,
            args.cutoff,
            args.cutoff_offset,
        )
        for path in selected
    ]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "atom_counts": manifest["atom_counts"],
            "samples_per_size": args.samples_per_size,
            "sample_count": len(probes),
            "selection_manifest": str(args.manifest),
            "selection_manifest_sha256": sha256_file(args.manifest),
            "cutoff_A": args.cutoff,
            "cutoff_offset_A": args.cutoff_offset,
            "checkpoint_dtype": str(native_dtype),
            "evaluation_dtype": str(dtype),
            "control": "same geometry with hard degree held fixed",
        },
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "probes": probes,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
