#!/usr/bin/env python3
"""Build reusable task-aware pilots from frozen H46/H276 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batch_mlip import (  # noqa: E402
    BatchTimingPoint,
    OptimizationPilot,
    PilotRegime,
)


def _mps_rate(
    summary: dict,
    *,
    model: str,
    optimizer: str,
    atom_count: int,
) -> float:
    row = next(
        row
        for row in summary["rows"]
        if row["model"] == model
        and row["optimizer"] == optimizer
        and row["atoms"] == atom_count
        and row["method"] == "mps32"
    )
    return float(row["systems_per_second"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "experiments/refill-batch-optimizer-factorial/"
            "results/results.json"
        ),
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("runs/task_aware_policy/pilots"),
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=Path("runs/task_aware_policy/eval_timing"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("benchmarks/workloads/profiles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/task_aware_policy/policy_inputs"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for model, profile_model in (
        ("atombit", "atombit"),
        ("mace", "mace_off_small"),
    ):
        for optimizer in ("bfgs", "fire"):
            regimes = []
            for atom_count in (46, 276):
                raw_path = (
                    args.pilot_dir
                    / f"{model}_{optimizer}_h{atom_count}.json"
                )
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                point = raw["points"][0]
                if point["status"] != "passed":
                    raise RuntimeError(f"pilot failed: {raw_path}")
                steps = tuple(int(record["steps"]) for record in point["records"])
                profile_path = (
                    args.profile_dir
                    / profile_model
                    / f"OPT-H{atom_count}-R256-v1.json"
                )
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                timing_points = []
                for batch_size in (1, 8, 16, 32, 64, 128):
                    timing_path = (
                        args.timing_dir
                        / f"{model}_h{atom_count}_b{batch_size}.json"
                    )
                    timing = json.loads(
                        timing_path.read_text(encoding="utf-8")
                    )
                    if (
                        timing["status"] != "passed"
                        or not timing["compute_stress"]
                    ):
                        raise RuntimeError(
                            f"invalid stress timing point: {timing_path}"
                        )
                    timing_points.append(
                        BatchTimingPoint(
                            batch_size=batch_size,
                            seconds=float(timing["wall_time_s"]),
                        )
                    )
                regimes.append(
                    PilotRegime(
                        label=f"H{atom_count}",
                        atom_count=atom_count,
                        edge_count=round(profile["active_edges_mean"]),
                        sampled_steps=steps,
                        timing_points=tuple(timing_points),
                        mps_systems_per_second=_mps_rate(
                            summary,
                            model=model,
                            optimizer=optimizer,
                            atom_count=atom_count,
                        ),
                    )
                )
            pilot = OptimizationPilot(
                optimizer=optimizer,
                regimes=tuple(regimes),
                source=(
                    "H46/H276 R32 step pilots, isolated warmed stress timing "
                    "curves, and prior signed R256 MPS32 results"
                ),
            )
            output = args.output_dir / f"{model}_{optimizer}.json"
            output.write_text(
                json.dumps(pilot.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            outputs[f"{model}_{optimizer}"] = str(output)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
