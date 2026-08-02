#!/usr/bin/env python3
"""Isolate batched-versus-ASE BFGS endpoints at resident batch size one."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]

from benchmark_production import (  # noqa: E402
    load_production_model,
    sha256_file,
)
from benchmark_variable_cell_scaling import run_batch  # noqa: E402

from batch_mlip import (  # noqa: E402
    ReproducibilityConfig,
    configure_reproducibility,
)
from batch_mlip.workloads import read_workload_manifest  # noqa: E402


def _energy_per_atom_difference(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    atom_count = len(left["positions_A"])
    return abs(float(left["energy_eV"]) - float(right["energy_eV"])) / atom_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reproducibility = configure_reproducibility(
        ReproducibilityConfig(
            seed=args.seed,
            cpu_threads=1,
            interop_threads=1,
        ),
        require_preconfigured_python_hash=True,
    )
    manifest = read_workload_manifest(args.manifest)
    jobs = {job.system_id: job for job in manifest.jobs}
    missing = sorted(set(args.source) - jobs.keys())
    if missing:
        raise ValueError(f"sources are absent from the signed manifest: {missing}")
    systems = []
    for source in args.source:
        job = jobs[source]
        atoms = read(
            args.dataset_dir / job.source_path,
            index=job.frame_index,
        )
        atoms.info["benchmark_source"] = source
        systems.append(atoms)

    device = torch.device(args.device)
    model, model_metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()
    output = run_batch(
        model,
        systems=systems,
        batch_size=1,
        active_compaction=True,
        device=device,
        cutoff=6.0,
        skin=0.5,
        optimizer_dtype="float64",
        model_dtype=torch.float32,
        neighbor_backend="auto",
        fmax=0.05,
        max_steps=500,
        dt_start=0.1,
        dt_max=1.0,
        max_step=0.2,
        optimizer_name="bfgs",
        alpha=70.0,
        linear_algebra_backend="auto",
    )

    reference_payload = json.loads(args.reference.read_text(encoding="utf-8"))
    reference = {record["source"]: record for record in reference_payload["records"]}
    actual = {record["source"]: record for record in output["records"]}
    rows = []
    for source in args.source:
        left = actual[source]
        right = reference[source]
        rows.append(
            {
                "source": source,
                "absolute_energy_difference_eV_per_atom": (
                    _energy_per_atom_difference(left, right)
                ),
                "position_max_abs_difference_A": float(
                    np.max(
                        np.abs(np.asarray(left["positions_A"]) - np.asarray(right["positions_A"]))
                    )
                ),
                "cell_max_abs_difference_A": float(
                    np.max(np.abs(np.asarray(left["cell_A"]) - np.asarray(right["cell_A"])))
                ),
                "batched_b1_steps": int(left["steps"]),
                "ase_steps": int(right["steps"]),
                "batched_b1_converged": bool(left["converged"]),
                "ase_converged": bool(right["converged"]),
            }
        )
    energy_differences = [row["absolute_energy_difference_eV_per_atom"] for row in rows]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "B1 numerical-equivalence isolation",
        "workload_id": manifest.workload_id,
        "workload_manifest_sha256": manifest.manifest_sha256,
        "sources": args.source,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            **model_metadata,
        },
        "contract": {
            "optimizer": "BatchedBFGS",
            "resident_batch_size": 1,
            "optimizer_dtype": "torch.float64",
            "cell_filter": "FrechetCellFilter",
            "fmax_eV_per_A": 0.05,
            "smax_eV_per_A3": None,
            "max_steps": 500,
            "max_step_A": 0.2,
            "alpha": 70.0,
        },
        "reproducibility": reproducibility,
        "comparison": {
            "maximum_absolute_energy_difference_eV_per_atom": max(energy_differences),
            "count_over_5meV_per_atom": sum(value > 0.005 for value in energy_differences),
            "count_over_1meV_per_atom": sum(value > 0.001 for value in energy_differences),
            "rows": rows,
        },
        "records": output["records"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
