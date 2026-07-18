"""Merge paired masked/active FIRE benchmark shards into one compact result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("runs")
ATOM_COUNTS = (46, 92, 184, 276)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_points(atom_count: int, mode: str) -> tuple[dict[int, dict], list[Path]]:
    points: dict[int, dict] = {}
    paths = []
    for shard in ("small", "large"):
        suffix = "active_vectorized" if mode == "active" else mode
        path = ROOT / f"compaction_paired_atoms{atom_count}_{shard}_{suffix}.json"
        payload = json.loads(path.read_text())
        paths.append(path)
        for point in payload["groups"][str(atom_count)]["points"]:
            points[point["batch_size"]] = point
    return points, paths


def main() -> None:
    summary = {
        "schema_version": 1,
        "experiment": "active-batch-compaction",
        "hypothesis": (
            "Removing converged graphs eliminates wasted graph evaluations and "
            "reduces batched FIRE wall time without changing scientific outcomes."
        ),
        "groups": {},
        "source_files": [],
    }
    active_validation_passes = []

    for atom_count in ATOM_COUNTS:
        masked, masked_paths = load_points(atom_count, "masked")
        active, active_paths = load_points(atom_count, "active")
        for path in masked_paths + active_paths:
            summary["source_files"].append(
                {"path": str(path), "sha256": sha256_file(path)}
            )

        group_points = []
        for batch_size in sorted(masked):
            baseline = masked[batch_size]
            candidate = active[batch_size]
            masked_seconds = baseline["timing"]["median_seconds"]
            active_seconds = candidate["timing"]["median_seconds"]
            active_uncompacted = sum(
                len(trace) * batch_size
                for trace in candidate["active_batch_sizes"]
            )
            validation_passed = bool(candidate["validation"]["passed"])
            active_validation_passes.append(validation_passed)
            group_points.append(
                {
                    "batch_size": batch_size,
                    "masked_status": baseline["status"],
                    "active_status": candidate["status"],
                    "masked_median_seconds": masked_seconds,
                    "active_median_seconds": active_seconds,
                    "active_speedup_vs_masked": masked_seconds / active_seconds,
                    "masked_graph_evaluations": baseline["graph_evaluations"],
                    "active_graph_evaluations": candidate["graph_evaluations"],
                    "active_uncompacted_graph_evaluations": active_uncompacted,
                    "masked_wasted_graph_evaluations": baseline[
                        "wasted_graph_evaluations"
                    ],
                    "active_wasted_graph_evaluations": candidate[
                        "wasted_graph_evaluations"
                    ],
                    "graph_evaluation_reduction_fraction": 1.0
                    - candidate["graph_evaluations"]
                    / active_uncompacted,
                    "masked_peak_memory_bytes": baseline["peak_memory_bytes"],
                    "active_peak_memory_bytes": candidate["peak_memory_bytes"],
                    "active_validation": candidate["validation"],
                }
            )
        summary["groups"][str(atom_count)] = {"points": group_points}

    summary["status"] = (
        "passed"
        if all(active_validation_passes)
        else "completed_with_validation_variation"
    )
    output = ROOT / "active_batch_compaction_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
