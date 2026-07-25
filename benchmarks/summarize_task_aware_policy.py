#!/usr/bin/env python3
"""Summarize held-out task-aware scheduling against measured candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _policy_candidate(path: Path) -> dict[str, Any]:
    data = _load(path)
    return {
        "mode": "policy_tensor",
        "seconds": float(data["wall_time_s"]),
        "peak_bytes": int(data["peak_reserved_bytes"]),
        "converged": int(data["converged"]),
        "schedule": data["schedule"]["batches"],
    }


def _mps_candidate(path: Path) -> dict[str, Any]:
    data = _load(path)
    return {
        "mode": "mps32",
        "seconds": float(data["timing"]["wall_seconds"]),
        "peak_bytes": int(data["peak_gpu_memory_bytes_nvidia_smi"]),
        "converged": int(data["converged"]),
    }


def _homogeneous_candidate(path: Path, mode: str) -> dict[str, Any]:
    point = _load(path)["points"][0]
    return {
        "mode": mode,
        "seconds": float(point["timing"]["median_seconds"]),
        "peak_bytes": int(point["peak_reserved_memory_bytes"]),
        "converged": sum(
            bool(record["converged"]) for record in point["records"]
        ),
    }


def _mixed_candidate(path: Path, mode: str) -> dict[str, Any]:
    data = _load(path)
    return {
        "mode": mode,
        "seconds": float(data["optimization_seconds"]),
        "peak_bytes": int(data["peak_reserved_bytes"]),
        "converged": int(data["converged"]),
    }


def _alternative_candidates(
    root: Path,
    *,
    model: str,
    optimizer: str,
    distribution: str,
    pool_size: int,
) -> list[dict[str, Any]]:
    candidates = []
    if pool_size in (64, 256):
        batch_size = 32 if pool_size == 64 else 64
        if distribution == "MIX4":
            path = (
                root
                / f"{model}_bfgs_MIX4_r{pool_size}_refill_b{batch_size}.json"
            )
            candidates.append(
                _mixed_candidate(path, f"refill_b{batch_size}")
            )
        else:
            atom_count = distribution.removeprefix("H")
            path = (
                root
                / f"{model}_{optimizer}_H{atom_count}_r{pool_size}"
                f"_refill_b{batch_size}.json"
            )
            candidates.append(
                _homogeneous_candidate(path, f"refill_b{batch_size}")
            )
    if (
        model == "atombit"
        and optimizer == "bfgs"
        and distribution == "H92"
        and pool_size == 256
    ):
        path = root / "atombit_bfgs_H92_r256_active_b128.json"
        candidates.append(_homogeneous_candidate(path, "drain_b128"))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/task_aware_policy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/task-aware-policy-validation/results"),
    )
    args = parser.parse_args()

    rows = []
    specifications = (
        ("atombit", "bfgs", "H92"),
        ("mace", "bfgs", "H92"),
        ("atombit", "fire", "H184"),
        ("mace", "fire", "H184"),
        ("atombit", "bfgs", "MIX4"),
        ("mace", "bfgs", "MIX4"),
    )
    for model, optimizer, distribution in specifications:
        for pool_size in (32, 64, 256):
            stem = f"{model}_{optimizer}_{distribution}_r{pool_size}"
            policy_data = _load(args.run_dir / "policy" / f"{stem}.json")
            tensor = _policy_candidate(
                args.run_dir / "policy" / f"{stem}.json"
            )
            mps = _mps_candidate(args.run_dir / "mps" / f"{stem}.json")
            candidates = [
                tensor,
                mps,
                *_alternative_candidates(
                    args.run_dir / "candidates",
                    model=model,
                    optimizer=optimizer,
                    distribution=distribution,
                    pool_size=pool_size,
                ),
            ]
            valid = [
                candidate
                for candidate in candidates
                if candidate["converged"] == pool_size
            ]
            fastest = (
                min(valid, key=lambda candidate: candidate["seconds"])
                if valid
                else None
            )
            recommendation = policy_data["schedule"][
                "recommended_worker_mode"
            ]
            selected = mps if recommendation == "mps" else tensor
            selected_valid = selected["converged"] == pool_size
            regret = (
                selected["seconds"] / fastest["seconds"]
                if selected_valid and fastest is not None
                else None
            )
            rows.append(
                {
                    "model": model,
                    "optimizer": optimizer,
                    "distribution": distribution,
                    "pool_size": pool_size,
                    "recommendation": recommendation,
                    "selected_mode": selected["mode"],
                    "selected_seconds": selected["seconds"],
                    "selected_peak_GiB": selected["peak_bytes"] / 2**30,
                    "selected_converged": selected["converged"],
                    "mps_seconds": mps["seconds"],
                    "speedup_vs_mps32": (
                        mps["seconds"] / selected["seconds"]
                    ),
                    "fastest_measured_mode": (
                        None if fastest is None else fastest["mode"]
                    ),
                    "fastest_measured_seconds": (
                        None if fastest is None else fastest["seconds"]
                    ),
                    "policy_regret": regret,
                    "quality_valid": selected_valid,
                    "candidates": candidates,
                }
            )

    valid_regrets = [
        row["policy_regret"]
        for row in rows
        if row["policy_regret"] is not None
    ]
    result = {
        "schema_version": 1,
        "scope": (
            "restricted oracle: policy tensor fallback, MPS32, smaller refill; "
            "one deterministic timing per candidate"
        ),
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "quality_valid_rows": sum(
                bool(row["quality_valid"]) for row in rows
            ),
            "rows_with_defined_regret": len(valid_regrets),
            "geometric_mean_regret": (
                None
                if not valid_regrets
                else float(
                    math.prod(valid_regrets) ** (1.0 / len(valid_regrets))
                )
            ),
            "maximum_regret": (
                None if not valid_regrets else max(valid_regrets)
            ),
            "within_five_percent_of_oracle": sum(
                regret <= 1.05 for regret in valid_regrets
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat_keys = [key for key in rows[0] if key != "candidates"]
    with (args.output_dir / "results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=flat_keys,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in flat_keys} for row in rows
        )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
