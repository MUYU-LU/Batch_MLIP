#!/usr/bin/env python3
"""Summarize active-drain and active-refill BFGS measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_variable_cell_scaling import validate


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point(path: Path) -> dict[str, Any]:
    artifact = load(path)
    result = artifact["points"][0]
    if artifact["status"] != "complete" or result["status"] != "passed":
        raise ValueError(f"benchmark point did not pass: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--atom-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    atom_reference = load(args.atom_reference)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "workload_size": 256,
        "measured_repeats": 1,
        "groups": {},
    }
    for model in ("atombit", "mace"):
        model_groups = summary["groups"].setdefault(model, {})
        for atom_count in (46, 92, 184, 276):
            if model == "atombit" and atom_count == 46:
                root = args.pilot_dir
                prefix = "atombit"
            else:
                root = args.raw_dir
                prefix = f"{model}_atoms{atom_count}"

            if model == "atombit":
                ase_32_seconds = atom_reference["groups"][str(atom_count)][
                    "ase"
                ]["median_seconds"]
                ase_source = str(args.atom_reference)
            else:
                ase_artifact = point(root / f"{prefix}_ase.json")
                ase_32_seconds = ase_artifact["timing"]["median_seconds"]
                ase_source = str(root / f"{prefix}_ase.json")

            group: dict[str, Any] = {
                "ase": {
                    "measured_pool_size": 32,
                    "measured_seconds": ase_32_seconds,
                    "extrapolation_factor": 8,
                    "equivalent_256_seconds": ase_32_seconds * 8,
                    "source": ase_source,
                },
                "points": {},
            }
            for resident_size, drain_factor in ((64, 4), (128, 2)):
                drain = point(root / f"{prefix}_active_b{resident_size}.json")
                refill = point(root / f"{prefix}_refill_b{resident_size}.json")
                drain_seconds = drain["timing"]["median_seconds"]
                drain_256_seconds = drain_seconds * drain_factor
                refill_seconds = refill["timing"]["median_seconds"]
                drain_records = drain["records"]
                refill_records = refill["records"]
                comparison = validate(
                    drain_records,
                    refill_records[:resident_size],
                    atom_count,
                )
                group["points"][str(resident_size)] = {
                    "active_drain": {
                        "measured_pool_size": resident_size,
                        "measured_seconds": drain_seconds,
                        "extrapolation_factor": drain_factor,
                        "equivalent_256_seconds": drain_256_seconds,
                        "peak_memory_bytes": drain["peak_memory_bytes"],
                        "converged_count_equivalent_256": sum(
                            record["converged"] for record in drain_records
                        )
                        * drain_factor,
                    },
                    "active_refill": {
                        "measured_pool_size": 256,
                        "measured_seconds": refill_seconds,
                        "peak_memory_bytes": refill["peak_memory_bytes"],
                        "converged_count": sum(
                            record["converged"] for record in refill_records
                        ),
                        "model_evaluations": refill["model_evaluations"],
                        "graph_evaluations": refill["graph_evaluations"],
                    },
                    "refill_speedup_vs_drain": drain_256_seconds
                    / refill_seconds,
                    "refill_speedup_vs_ase": (
                        ase_32_seconds * 8 / refill_seconds
                    ),
                    "validation_first_resident_block": comparison,
                }
            model_groups[str(atom_count)] = group

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
