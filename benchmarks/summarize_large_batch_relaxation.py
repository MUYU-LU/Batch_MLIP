"""Merge the 1,024-system large-batch masked and active benchmark results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path("runs")
ATOM_COUNTS = (46, 92, 184, 276)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(atom_count: int, mode: str) -> tuple[dict, Path]:
    path = ROOT / f"large_pool1024_atoms{atom_count}_{mode}.json"
    return json.loads(path.read_text()), path


def point_summary(point: dict, ase_seconds: float) -> dict:
    result = {
        "status": point["status"],
        "error": point.get("error"),
    }
    if "timing" in point:
        seconds = point["timing"]["median_seconds"]
        result.update(
            {
                "median_seconds": seconds,
                "speedup_vs_ase": ase_seconds / seconds,
                "systems_per_second": point.get(
                    "systems_per_second", 1024 / seconds
                ),
                "peak_memory_bytes": point["peak_memory_bytes"],
                "graph_evaluations": point["graph_evaluations"],
                "wasted_graph_evaluations": point.get(
                    "wasted_graph_evaluations"
                ),
                "validation": point["validation"],
            }
        )
    return result


def load_replicated_point(batch_size: int) -> tuple[dict, list[Path]]:
    paths = [
        ROOT / f"large_pool1024_atoms92_masked_b{batch_size}_rep{replica}.json"
        for replica in range(3)
    ]
    payloads = [json.loads(path.read_text()) for path in paths]
    samples = [item["timing"]["median_seconds"] for item in payloads]
    median_seconds = median(samples)
    representative = min(
        payloads,
        key=lambda item: abs(item["timing"]["median_seconds"] - median_seconds),
    )
    point = dict(representative)
    point["timing"] = {
        "samples_seconds": samples,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "mean_seconds": mean(samples),
        "median_seconds": median_seconds,
        "std_seconds": pstdev(samples),
    }
    point["replica_statuses"] = [item["status"] for item in payloads]
    return point, paths


def main() -> None:
    output = {
        "schema_version": 1,
        "experiment": "large-batch-relaxation-1024-pool",
        "pool": {
            "systems": 1024,
            "base_systems": 16,
            "expansion": "64 cyclic repetitions of the fixed base systems",
            "batch_sizes": [64, 128, 256, 512, 1024],
        },
        "groups": {},
        "source_files": [],
    }

    for atom_count in ATOM_COUNTS:
        masked_payload, masked_path = load(atom_count, "masked")
        active_payload, active_path = load(atom_count, "active")
        for path in (masked_path, active_path):
            output["source_files"].append(
                {"path": str(path), "sha256": sha256_file(path)}
            )

        masked_group = masked_payload["groups"][str(atom_count)]
        active_group = active_payload["groups"][str(atom_count)]
        masked_ase = masked_group["ase_reference"]["timing"]["median_seconds"]
        active_ase = active_group["ase_reference"]["timing"]["median_seconds"]
        common_ase = median((masked_ase, active_ase))
        masked_points = {p["batch_size"]: p for p in masked_group["points"]}
        active_points = {p["batch_size"]: p for p in active_group["points"]}
        if atom_count == 92:
            for batch_size in (128, 256):
                masked_points[batch_size], replica_paths = load_replicated_point(
                    batch_size
                )
                for path in replica_paths:
                    output["source_files"].append(
                        {"path": str(path), "sha256": sha256_file(path)}
                    )
            for batch_size in (512, 1024):
                path = ROOT / f"large_pool1024_atoms92_masked_b{batch_size}_aux.json"
                masked_points[batch_size] = json.loads(path.read_text())
                output["source_files"].append(
                    {"path": str(path), "sha256": sha256_file(path)}
                )

        points = []
        for batch_size in sorted(masked_points):
            masked = point_summary(masked_points[batch_size], masked_ase)
            active = point_summary(active_points[batch_size], active_ase)
            if "median_seconds" in masked:
                masked["speedup_vs_common_ase"] = (
                    common_ase / masked["median_seconds"]
                )
            if "median_seconds" in active:
                active["speedup_vs_common_ase"] = (
                    common_ase / active["median_seconds"]
                )
            item = {
                "batch_size": batch_size,
                "masked": masked,
                "active": active,
            }
            if "median_seconds" in masked and "median_seconds" in active:
                item["active_speedup_vs_masked"] = (
                    masked["median_seconds"] / active["median_seconds"]
                )
            points.append(item)

        output["groups"][str(atom_count)] = {
            "ase_seconds_masked_worker": masked_ase,
            "ase_seconds_active_worker": active_ase,
            "ase_median_seconds": common_ase,
            "points": points,
        }

    statuses = [
        point[mode]["status"]
        for group in output["groups"].values()
        for point in group["points"]
        for mode in ("masked", "active")
    ]
    output["status"] = (
        "passed"
        if all(status == "passed" for status in statuses)
        else "completed_with_failures_or_oom"
    )
    path = ROOT / "large_batch_relaxation_1024_summary.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
