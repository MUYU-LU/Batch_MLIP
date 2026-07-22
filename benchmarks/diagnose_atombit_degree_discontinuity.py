#!/usr/bin/env python3
"""Isolate AtomBit cutoff discontinuity from hard degree normalization."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import torch
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_production import (  # noqa: E402
    environment_metadata,
    load_production_model,
    sha256_file,
    write_result,
)

from batch_mlip import AtomBitBatchCalculator  # noqa: E402
from batch_mlip.core.types import GraphData  # noqa: E402


def _state_at_parameter(
    calculator: AtomBitBatchCalculator,
    atoms: Any,
    snapshot: dict[str, object],
    parameter: float,
):
    dtype = calculator.dtype
    device = calculator.device
    base = snapshot["base_coordinates"].to(device=device, dtype=dtype).reshape(-1, 3)
    direction = snapshot["direction"].to(device=device, dtype=dtype).reshape(-1, 3)
    coordinate = base + parameter * direction
    atom_count = len(atoms)
    factor = torch.as_tensor(snapshot["cell_factor"], device=device, dtype=dtype)
    log_deformation = coordinate[atom_count:] / factor
    deformation = torch.matrix_exp(log_deformation)
    reference_cell = snapshot["reference_cell"].to(device=device, dtype=dtype)
    state = calculator.create_state([atoms], build_neighbors=False)
    state.cells = (reference_cell @ deformation.T).unsqueeze(0)
    state.positions = coordinate[:atom_count] @ deformation.T
    state.rebuild_neighbor_list()
    return state


def _with_topology(geometry: GraphData, topology: GraphData) -> GraphData:
    return GraphData(
        z=geometry.z,
        pos=geometry.pos,
        cell=geometry.cell,
        edge_index=topology.edge_index,
        shifts_int=topology.shifts_int,
        batch=geometry.batch,
        num_graphs=geometry.num_graphs,
    )


def _edge_keys(data: GraphData) -> list[tuple[int, int, int, int, int]]:
    edges = data.edge_index.detach().cpu().T.tolist()
    shifts = data.shifts_int.detach().cpu().tolist()
    return [
        (int(edge[0]), int(edge[1]), *(int(value) for value in shift))
        for edge, shift in zip(edges, shifts, strict=True)
    ]


def _edge_distances(data: GraphData) -> torch.Tensor:
    center, neighbor = data.edge_index
    if data.cell is None:
        shifts = torch.zeros_like(data.pos[center])
    else:
        cells = data.cell[data.batch[center]]
        shifts = torch.bmm(
            data.shifts_int.unsqueeze(1).to(data.pos.dtype), cells
        ).squeeze(1)
    vectors = data.pos[center] - data.pos[neighbor] - shifts
    return torch.linalg.vector_norm(vectors, dim=1)


def _inverse_sqrt_degree(data: GraphData) -> torch.Tensor:
    center = data.edge_index[0]
    degree = torch.zeros(data.z.shape[0], device=data.z.device, dtype=data.pos.dtype)
    degree.index_add_(0, center, torch.ones_like(center, dtype=data.pos.dtype))
    return torch.rsqrt(degree.clamp_min(1.0))


@contextlib.contextmanager
def _fixed_degree(model: torch.nn.Module, value: torch.Tensor):
    """Temporarily make every interaction block use the same node degree."""
    originals = []
    for block in model.blocks:
        density = block.density
        original = density.forward
        originals.append((density, original))

        def replacement(
            messages,
            index,
            num_nodes,
            inv_sqrt_deg=None,
            *,
            _original=original,
        ):
            del inv_sqrt_deg
            return _original(
                messages,
                index,
                num_nodes,
                inv_sqrt_deg=value,
            )

        density.forward = replacement
    try:
        yield
    finally:
        for density, original in originals:
            density.forward = original


def _energy(model: torch.nn.Module, data: GraphData) -> float:
    with torch.no_grad():
        output = model(data)
    if isinstance(output, dict):
        output = output["energy"]
    return float(output.reshape(-1)[0].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--trace-checkpoint", type=Path, required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    parser.add_argument("--parameter-below", type=float, required=True)
    parser.add_argument("--parameter-above", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.parameter_below >= args.parameter_above:
        raise ValueError("parameter-below must be smaller than parameter-above")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    dtype = torch.float64
    trace = torch.load(args.trace_checkpoint, map_location="cpu", weights_only=False)
    snapshot = trace["starts"][args.optimizer_step]
    atoms = read(args.structure)
    model, checkpoint_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=dtype).eval()
    calculator = AtomBitBatchCalculator(
        model,
        cutoff=args.cutoff,
        skin=args.skin,
        device=device,
        dtype=dtype,
        force_mode="autograd",
    )

    below_state = _state_at_parameter(
        calculator, atoms, snapshot, args.parameter_below
    )
    above_state = _state_at_parameter(
        calculator, atoms, snapshot, args.parameter_above
    )
    below = below_state.as_model_data()
    above = above_state.as_model_data()
    below_keys = _edge_keys(below)
    above_keys = _edge_keys(above)
    below_key_set = set(below_keys)
    above_key_set = set(above_keys)
    removed = sorted(below_key_set - above_key_set)
    added = sorted(above_key_set - below_key_set)

    below_with_below = _with_topology(below, below)
    below_with_above = _with_topology(below, above)
    above_with_below = _with_topology(above, below)
    above_with_above = _with_topology(above, above)
    normal = {
        "below_geometry_below_topology": _energy(model, below_with_below),
        "below_geometry_above_topology": _energy(model, below_with_above),
        "above_geometry_below_topology": _energy(model, above_with_below),
        "above_geometry_above_topology": _energy(model, above_with_above),
    }

    fixed_results = {}
    for label, topology in (("below", below), ("above", above)):
        with _fixed_degree(model, _inverse_sqrt_degree(topology)):
            fixed_results[label] = {
                "below_geometry_below_topology": _energy(model, below_with_below),
                "below_geometry_above_topology": _energy(model, below_with_above),
                "above_geometry_below_topology": _energy(model, above_with_below),
                "above_geometry_above_topology": _energy(model, above_with_above),
            }

    below_distances = _edge_distances(below)
    below_lookup = {key: index for index, key in enumerate(below_keys)}
    removed_distances = [
        float(below_distances[below_lookup[key]].item()) for key in removed
    ]
    removed_envelopes = []
    for distance in removed_distances:
        x = min(max(distance / args.cutoff, 0.0), 1.0)
        removed_envelopes.append(1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2))

    normal_jump = (
        normal["above_geometry_above_topology"]
        - normal["below_geometry_below_topology"]
    )
    topology_effect_above = (
        normal["above_geometry_above_topology"]
        - normal["above_geometry_below_topology"]
    )
    fixed_below = fixed_results["below"]
    fixed_topology_effect_above = (
        fixed_below["above_geometry_above_topology"]
        - fixed_below["above_geometry_below_topology"]
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "structure": args.structure.name,
        "optimizer_step": args.optimizer_step,
        "parameters": {
            "parameter_below": args.parameter_below,
            "parameter_above": args.parameter_above,
            "interval": args.parameter_above - args.parameter_below,
            "cutoff_A": args.cutoff,
            "skin_A": args.skin,
            "dtype": "float64",
        },
        "environment": environment_metadata(device),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "topology": {
            "edges_below": len(below_keys),
            "edges_above": len(above_keys),
            "removed": [list(key) for key in removed],
            "added": [list(key) for key in added],
            "removed_distances_below_A": removed_distances,
            "removed_envelopes_below": removed_envelopes,
        },
        "normal_energy_eV": normal,
        "fixed_degree_energy_eV": fixed_results,
        "effects_eV": {
            "normal_crossing_jump": normal_jump,
            "topology_effect_at_above_geometry": topology_effect_above,
            "topology_effect_at_above_geometry_with_fixed_below_degree": (
                fixed_topology_effect_above
            ),
        },
        "hard_degree_normalization_is_causal": (
            abs(topology_effect_above) > 1e-6
            and abs(fixed_topology_effect_above) < 1e-10
        ),
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
