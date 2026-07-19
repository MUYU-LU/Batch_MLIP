#!/usr/bin/env python3
"""Validate variable-cell neighbor caching against skin-zero AtomBit BFGS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_production import (  # noqa: E402
    load_manifest,
    load_production_model,
    sha256_file,
    write_result,
)

from batch_mlip import (  # noqa: E402
    AtomBitBatchCalculator,
    FrechetCellFilter,
    RuntimeProfiler,
    relax,
)


def snapshot(results) -> dict[str, torch.Tensor]:
    return {
        "cells": torch.cat([result.state.cells.detach().cpu() for result in results]),
        "converged": torch.cat(
            [result.converged.detach().cpu() for result in results]
        ),
        "energies": torch.cat(
            [result.evaluation.energy.detach().cpu() for result in results]
        ),
        "forces": torch.cat(
            [result.evaluation.forces.detach().cpu() for result in results]
        ),
        "positions": torch.cat(
            [result.state.positions.detach().cpu() for result in results]
        ),
        "steps": torch.cat(
            [result.converged_step.detach().cpu() for result in results]
        ),
        "stresses": torch.cat(
            [result.evaluation.stress.detach().cpu() for result in results]
        ),
    }


def compare(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]):
    fields: dict[str, Any] = {}
    passed = True
    tolerances = {
        "cells": (1e-6, 1e-6),
        "energies": (1e-5, 1e-6),
        "forces": (1e-5, 1e-5),
        "positions": (1e-6, 1e-6),
        "stresses": (1e-6, 1e-5),
    }
    for name, reference_value in reference.items():
        candidate_value = candidate[name]
        exact = torch.equal(reference_value, candidate_value)
        if reference_value.dtype in (torch.bool, torch.int32, torch.int64):
            close = exact
            max_abs = 0.0 if exact else None
        else:
            atol, rtol = tolerances[name]
            close = torch.allclose(
                reference_value, candidate_value, atol=atol, rtol=rtol
            )
            max_abs = float((reference_value - candidate_value).abs().max())
        fields[name] = {
            "exact": exact,
            "close": bool(close),
            "max_abs_difference": max_abs,
        }
        passed &= bool(close)
    return {"passed": passed, "fields": fields, "tolerances": tolerances}


def run_mode(
    systems,
    calculator,
    *,
    mode: str,
    max_steps: int,
    optimizer_dtype: torch.dtype,
):
    groups = [[atoms] for atoms in systems] if mode == "b1" else [systems]
    with RuntimeProfiler(device=calculator.device) as profiler:
        results = [
            relax(
                group,
                calculator,
                optimizer="bfgs",
                cell_filter=FrechetCellFilter(),
                fmax=1e-30,
                smax=None,
                max_steps=max_steps,
                optimizer_dtype=optimizer_dtype,
            )
            for group in groups
        ]
    profile = profiler.summary(include_samples=False)
    neighbor_events = [
        event for event in profile["events"] if event["name"] == "neighbor_rebuild"
    ]
    model_events = [
        event
        for event in profile["events"]
        if event["name"] == "model_evaluation" and event["adapter"] == "atombit"
    ]
    physical_edges = sum(int(event["edges"]) for event in model_events)
    candidate_edges = sum(int(event["candidate_edges"]) for event in model_events)
    return {
        "snapshot": snapshot(results),
        "metrics": {
            "model_evaluations": sum(result.model_evaluations for result in results),
            "graph_evaluations": sum(result.graph_evaluations for result in results),
            "neighbor_rebuild_calls": sum(
                result.state.neighbor_rebuild_count for result in results
            ),
            "profile_seconds": profile["total_seconds"],
            "phase_totals_seconds": {
                name: values["total_seconds"]
                for name, values in profile["phases"].items()
            },
            "neighbor_events": len(neighbor_events),
            "rebuilt_systems": sum(
                int(event["rebuilt_systems"]) for event in neighbor_events
            ),
            "physical_edges_evaluated": physical_edges,
            "candidate_edges_filtered": candidate_edges,
            "candidate_to_physical_edge_ratio": (
                candidate_edges / physical_edges if physical_edges else 1.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/T2_test/structures")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.skin <= 0.0:
        raise ValueError("skin must be positive for cache validation")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    manifest = load_manifest(args.manifest, 1)
    names = [manifest["samples"][str(count)][0] for count in (46, 92, 184, 276)]
    systems = [read(args.dataset_dir / name) for name in names]

    reference_model, checkpoint_metadata = load_production_model(args.checkpoint)
    candidate_model, _ = load_production_model(args.checkpoint)
    reference = AtomBitBatchCalculator(
        reference_model,
        cutoff=6.0,
        skin=0.0,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )
    candidate = AtomBitBatchCalculator(
        candidate_model,
        cutoff=6.0,
        skin=args.skin,
        device=device,
        dtype=torch.float32,
        force_mode="autograd",
    )

    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "samples": names,
        "atom_counts": [len(atoms) for atoms in systems],
        "parameters": {
            "cutoff_A": 6.0,
            "candidate_skin_A": args.skin,
            "max_steps": args.max_steps,
            "model_dtype": "float32",
            "optimizer_dtype": "float64",
            "deterministic_algorithms": True,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            **checkpoint_metadata,
        },
        "modes": {},
    }
    write_result(args.output, output)

    for mode in ("b1", "heterogeneous_b4"):
        reference_run = run_mode(
            systems,
            reference,
            mode=mode,
            max_steps=args.max_steps,
            optimizer_dtype=torch.float64,
        )
        candidate_run = run_mode(
            systems,
            candidate,
            mode=mode,
            max_steps=args.max_steps,
            optimizer_dtype=torch.float64,
        )
        validation = compare(reference_run["snapshot"], candidate_run["snapshot"])
        output["modes"][mode] = {
            "reference": reference_run["metrics"],
            "candidate": candidate_run["metrics"],
            "validation": validation,
        }
        write_result(args.output, output)

    output["status"] = (
        "passed"
        if all(value["validation"]["passed"] for value in output["modes"].values())
        else "failed"
    )
    write_result(args.output, output)
    print(json.dumps({"status": output["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
