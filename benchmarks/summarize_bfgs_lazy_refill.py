#!/usr/bin/env python3
"""Compare lazy pending graphs with the eager BFGS refill baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_variable_cell_scaling import validate


def load_point(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    point = artifact["points"][0]
    if artifact["status"] != "complete" or point["status"] != "passed":
        raise ValueError(f"benchmark point did not pass: {path}")
    return point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-dir", type=Path, required=True)
    parser.add_argument("--eager-pilot-dir", type=Path, required=True)
    parser.add_argument("--lazy-dir", type=Path, required=True)
    parser.add_argument("--lazy-pilot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
            atom_group: dict[str, Any] = {}
            for resident_size in (64, 128):
                if model == "atombit" and atom_count == 46:
                    eager_path = args.eager_pilot_dir / (
                        f"atombit_refill_b{resident_size}.json"
                    )
                else:
                    eager_path = args.eager_dir / (
                        f"{model}_atoms{atom_count}_refill_b{resident_size}.json"
                    )
                lazy_root = (
                    args.lazy_pilot_dir if atom_count == 46 else args.lazy_dir
                )
                lazy_path = lazy_root / (
                    f"{model}_atoms{atom_count}_refill_b{resident_size}.json"
                )
                eager = load_point(eager_path)
                lazy = load_point(lazy_path)
                eager_seconds = eager["timing"]["median_seconds"]
                lazy_seconds = lazy["timing"]["median_seconds"]
                atom_group[str(resident_size)] = {
                    "eager_seconds": eager_seconds,
                    "lazy_seconds": lazy_seconds,
                    "eager_over_lazy_time_ratio": eager_seconds / lazy_seconds,
                    "eager_peak_memory_bytes": eager["peak_memory_bytes"],
                    "lazy_peak_memory_bytes": lazy["peak_memory_bytes"],
                    "memory_saved_bytes": (
                        eager["peak_memory_bytes"] - lazy["peak_memory_bytes"]
                    ),
                    "validation": validate(
                        eager["records"], lazy["records"], atom_count
                    ),
                }
            model_groups[str(atom_count)] = atom_group

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
