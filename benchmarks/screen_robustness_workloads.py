#!/usr/bin/env python3
"""Screen fixed robustness representatives for MLIP and batching compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmark_production import load_production_model, sha256_file  # noqa: E402

from batch_mlip import AtomBitBatchCalculator, MACEBatchCalculator  # noqa: E402
from batch_mlip.workloads import read_workload_manifest, topology_key  # noqa: E402


def _representatives(index_path: Path) -> list[dict[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    edge_key = topology_key(6.0, 0.0)
    representatives = []
    for workload_id, metadata in sorted(index["workloads"].items()):
        if "CROSS-MIX" in workload_id:
            continue
        manifest = read_workload_manifest(metadata["manifest_json"])
        unique = {}
        for job in manifest.jobs:
            unique.setdefault(job.normalized_structure_sha256, job)
        ranked = sorted(
            unique.values(),
            key=lambda job: (
                job.topology_edge_counts[edge_key] / job.atom_count,
                job.source_path,
            ),
        )
        for stratum, index_value in (
            ("low", 0),
            ("median", len(ranked) // 2),
            ("high", len(ranked) - 1),
        ):
            job = ranked[index_value]
            representatives.append(
                {
                    "workload_id": workload_id,
                    "stratum": stratum,
                    "source_path": job.source_path,
                    "atom_count": job.atom_count,
                    "chemical_formula": job.chemical_formula,
                    "edges": job.topology_edge_counts[edge_key],
                    "edges_per_atom": (
                        job.topology_edge_counts[edge_key] / job.atom_count
                    ),
                }
            )
    return representatives


def _evaluate(calculator, systems) -> dict[str, np.ndarray]:
    state = calculator.create_state(systems)
    result = calculator(state, compute_stress=True)
    if result.stress is None:
        raise RuntimeError("compatibility screen requires stress")
    return {
        "energy": result.energy.detach().cpu().numpy(),
        "forces": result.forces.detach().cpu().numpy(),
        "stress": result.stress.detach().cpu().numpy(),
    }


def _single_outputs(calculator, systems) -> dict[str, np.ndarray]:
    outputs = [_evaluate(calculator, [atoms]) for atoms in systems]
    return {
        "energy": np.concatenate([item["energy"] for item in outputs]),
        "forces": np.concatenate([item["forces"] for item in outputs]),
        "stress": np.concatenate([item["stress"] for item in outputs]),
    }


def _errors(reference, candidate) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlip", choices=("atombit", "mace"), required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("runs/robustness/workloads/index.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/public/home/lmy/Batch_imple_project/test_set"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/public/home/lmy/Batch_imple_project/AtomBit-OMC-s/checkpoints/"
            "smooth_rms_finetune/"
            "AtomBit-OMC-s_smooth_rms_fp32_7gpu_epoch5.pt"
        ),
    )
    parser.add_argument("--mace-model", default="small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    representatives = _representatives(args.index)
    systems = []
    for record in representatives:
        atoms = read(args.dataset_dir / record["source_path"])
        atoms.info["benchmark_source"] = record["source_path"]
        systems.append(atoms)

    device = torch.device(args.device)
    if args.mlip == "atombit":
        model, checkpoint_metadata = load_production_model(args.checkpoint)
        model = model.to(device=device, dtype=torch.float32).eval()
        calculator = AtomBitBatchCalculator(
            model,
            cutoff=6.0,
            skin=0.5,
            device=device,
            dtype=torch.float32,
            force_mode="autograd",
            neighbor_backend="auto",
        )
        supported = sorted(int(value) for value in model.used_atomic_numbers)
        tolerance = {
            "max_energy_error_eV": 2e-4,
            "max_force_error_eV_per_A": 2e-4,
            "max_stress_error_eV_per_A3": 2e-5,
        }
        model_metadata = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
        }
    else:
        calculator = MACEBatchCalculator.from_off(
            model=args.mace_model,
            device=device,
            dtype=torch.float64,
            graph_mode="rebuild",
            skin=0.0,
            neighbor_backend="auto",
        )
        supported = sorted(int(value) for value in calculator.z_table.zs)
        tolerance = {
            "max_energy_error_eV": 1e-8,
            "max_force_error_eV_per_A": 1e-8,
            "max_stress_error_eV_per_A3": 1e-8,
        }
        model_metadata = {"model": args.mace_model}

    present = sorted(
        {
            int(number)
            for atoms in systems
            for number in atoms.numbers
        }
    )
    unsupported = sorted(set(present) - set(supported))
    if unsupported:
        raise ValueError(f"selected positive controls contain unsupported Z: {unsupported}")

    _evaluate(calculator, systems[:1])
    torch.cuda.reset_peak_memory_stats(device)
    single = _single_outputs(calculator, systems)
    batch = _evaluate(calculator, systems)
    errors = _errors(single, batch)
    finite = all(
        bool(np.isfinite(values).all())
        for output in (single, batch)
        for values in output.values()
    )
    records = []
    atom_offset = 0
    for index, (metadata, atoms) in enumerate(zip(representatives, systems, strict=True)):
        atom_count = len(atoms)
        force_block = batch["forces"][atom_offset : atom_offset + atom_count]
        records.append(
            {
                **metadata,
                "energy_eV": float(batch["energy"][index]),
                "max_force_eV_per_A": float(
                    np.linalg.vector_norm(force_block, axis=1).max()
                ),
                "max_abs_stress_eV_per_A3": float(
                    np.abs(batch["stress"][index]).max()
                ),
            }
        )
        atom_offset += atom_count
    passed = finite and all(
        errors[name] <= limit for name, limit in tolerance.items()
    )
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "mlip": args.mlip,
        "device": str(device),
        "index": str(args.index),
        "index_sha256": sha256_file(args.index),
        "representative_count": len(systems),
        "supported_atomic_numbers": supported,
        "present_atomic_numbers": present,
        "unsupported_atomic_numbers": unsupported,
        "finite": finite,
        "batch_vs_single": errors,
        "tolerances": tolerance,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "records": records,
        **model_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "mlip": args.mlip,
                "output": str(args.output),
                "batch_vs_single": errors,
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
