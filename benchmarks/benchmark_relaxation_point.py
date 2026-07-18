"""Run one production relaxation point against an expanded base reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ase.io import read
from benchmark_production import load_manifest, load_production_model, synchronize, write_result
from benchmark_relaxation import (
    run_ase_pool,
    run_batched_pool,
    timed_runs,
    timing_summary,
    validate_relaxations,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--atom-count", type=int, required=True)
    result.add_argument("--batch-size", type=int, required=True)
    result.add_argument("--pool-size", type=int, default=1024)
    result.add_argument("--base-sample-count", type=int, default=16)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--active-compaction", action="store_true")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--dataset-dir", type=Path, default=Path("data/T2_test/structures"))
    result.add_argument("--manifest", type=Path, default=Path("benchmarks/t2_fixed_samples.json"))
    result.add_argument("--checkpoint", type=Path, default=Path("../AtomBit-OMC-s/model_epoch_15.pt"))
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.pool_size % args.batch_size:
        raise ValueError("batch size must divide pool size")
    manifest = load_manifest(args.manifest, args.base_sample_count)
    names = manifest["samples"][str(args.atom_count)][: args.base_sample_count]
    base_systems = []
    for name in names:
        atoms = read(args.dataset_dir / name)
        atoms.info["benchmark_source"] = name
        base_systems.append(atoms)

    device = torch.device(args.device)
    model, _metadata = load_production_model(args.checkpoint)
    model = model.to(device=device, dtype=torch.float32).eval()
    common = {
        "model": model,
        "device": device,
        "cutoff": 6.0,
        "fmax": 0.05,
        "max_steps": 500,
        "dt_start": 0.1,
        "dt_max": 1.0,
        "max_step": 0.2,
        "alpha_start": 0.1,
        "n_min": 5,
        "f_inc": 1.1,
        "f_dec": 0.5,
        "f_alpha": 0.99,
    }
    base_reference = run_ase_pool(systems=base_systems, **common)

    systems = []
    reference_records = []
    for pool_index in range(args.pool_size):
        base_index = pool_index % len(base_systems)
        source = f"{names[base_index]}#pool-{pool_index:04d}"
        atoms = base_systems[base_index].copy()
        atoms.info["benchmark_source"] = source
        systems.append(atoms)
        record = dict(base_reference["records"][base_index])
        record["source"] = source
        reference_records.append(record)

    synchronize(device)
    result = {
        "status": "running",
        "atom_count": args.atom_count,
        "batch_size": args.batch_size,
        "pool_size": args.pool_size,
        "active_compaction": args.active_compaction,
    }
    write_result(args.output, result)
    try:
        batch, samples, peak = timed_runs(
            lambda: run_batched_pool(
                systems=systems,
                batch_size=args.batch_size,
                skin=0.0,
                active_compaction=args.active_compaction,
                **common,
            ),
            repeats=args.repeats,
            device=device,
        )
        validation = validate_relaxations(
            {"records": reference_records},
            batch,
            atom_count=args.atom_count,
            target_fmax=0.05,
            energy_per_atom_atol=5e-5,
            force_atol=2e-2,
            position_rmsd_atol=1e-2,
            step_atol=15,
        )
        result.update(
            {
                "status": "passed" if validation["passed"] else "validation_failed",
                "timing": timing_summary(samples),
                "peak_memory_bytes": peak,
                "model_evaluations": batch["model_evaluations"],
                "graph_evaluations": batch["graph_evaluations"],
                "validation": validation,
            }
        )
    except torch.cuda.OutOfMemoryError as error:
        result.update({"status": "oom", "error": str(error)})
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
