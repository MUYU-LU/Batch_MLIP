#!/usr/bin/env python3
"""Validate and summarize the refill pool-size and multi-GPU matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import read_workload_manifest  # noqa: E402

CHECKPOINT_SHA256 = (
    "b27c372cb2c2848ae1e54a9ffd7a2aa0b0401d9cc0ed922e3be250ff63e44486"
)
FIT_SINGLE = ("XATMOV88", "XAFPAY172")
HELDOUT_SINGLE = ("SOXLEX48",)
FIT_MULTI = ("XATMOV88",)
HELDOUT_MULTI = ("BOQWIN116",)
SINGLE_POINTS = ((128, 32), (128, 64), (512, 64), (512, 128))
MULTI_POINTS = ((512, 2, 64), (1024, 4, 64))


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "passed":
        raise ValueError(f"{path} did not pass")
    return payload


def _flat(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        if isinstance(value, list):
            output.extend(_flat(value))
        else:
            output.append(float(value))
    return output


def _rms(left: list[Any], right: list[Any]) -> float:
    a = _flat(left)
    b = _flat(right)
    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a)
    )


def _endpoint(active: dict[str, Any], refill: dict[str, Any]) -> dict[str, Any]:
    left = {record["source"]: record for record in active["records"]}
    right = {record["source"]: record for record in refill["records"]}
    if left.keys() != right.keys():
        raise ValueError("active/refill sources differ")
    energies = []
    positions = []
    cells = []
    mismatches = 0
    for source, control in left.items():
        current = right[source]
        count = len(control["positions_A"])
        energies.append(
            abs(control["energy_eV"] - current["energy_eV"])
            * 1000.0
            / count
        )
        positions.append(
            _rms(control["positions_A"], current["positions_A"])
        )
        cells.append(_rms(control["cell_A"], current["cell_A"]))
        mismatches += control["converged"] != current["converged"]
    return {
        "endpoint_max_energy_difference_meV_per_atom": max(energies),
        "endpoint_energy_outliers_above_5_meV_per_atom": sum(
            value > 5.0 for value in energies
        ),
        "endpoint_max_position_rmsd_A": max(positions),
        "endpoint_max_cell_rmsd_A": max(cells),
        "endpoint_convergence_mismatches": mismatches,
    }


def _descriptors(manifest) -> dict[str, Any]:
    atoms = [job.atom_count for job in manifest.jobs]
    edges = [
        job.topology_edge_counts["cutoff=6.000_skin=0.500"]
        for job in manifest.jobs
    ]

    def cv(values):
        mean = sum(values) / len(values)
        return math.sqrt(
            sum((value - mean) ** 2 for value in values) / len(values)
        ) / mean

    return {
        "mean_atom_count": sum(atoms) / len(atoms),
        "atom_count_cv": cv(atoms),
        "mean_edge_count": sum(edges) / len(edges),
        "edge_count_cv": cv(edges),
    }


def _validate_pair(
    active: dict[str, Any],
    refill: dict[str, Any],
    *,
    manifest,
    pool_size: int,
    gpu_count: int,
    resident_capacity: int,
) -> None:
    for payload, method in ((active, "active"), (refill, "refill")):
        if payload["method"] != method:
            raise ValueError("method label mismatch")
        if payload.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError("checkpoint mismatch")
        if payload["workload_manifest_sha256"] != manifest.manifest_sha256:
            raise ValueError("manifest mismatch")
        actual_pool = payload.get("pool_size", payload.get("jobs"))
        if actual_pool != pool_size:
            raise ValueError("pool size mismatch")
        actual_capacity = payload.get(
            "resident_capacity",
            payload.get("batch_size"),
        )
        if actual_capacity != resident_capacity:
            raise ValueError("resident capacity mismatch")
    if gpu_count > 1:
        if active["gpu_count"] != gpu_count or refill["gpu_count"] != gpu_count:
            raise ValueError("GPU count mismatch")
        if active["worker_system_indices"] != refill["worker_system_indices"]:
            raise ValueError("active/refill worker shards differ")


def _row(
    active: dict[str, Any],
    refill: dict[str, Any],
    *,
    family: str,
    split: str,
    manifest,
    pool_size: int,
    gpu_count: int,
    resident_capacity: int,
) -> dict[str, Any]:
    timing_key = (
        "timing_seconds"
        if gpu_count == 1
        else "optimization_wall_seconds"
    )
    endpoint = _endpoint(active, refill)
    speedup = active[timing_key] / refill[timing_key]
    peak_key = (
        "peak_reserved_bytes"
        if gpu_count == 1
        else "peak_reserved_bytes_max_per_worker"
    )
    total_memory = (
        active["environment"]["gpu_total_memory_bytes"]
        if gpu_count == 1
        else refill["workers"][0]["device_metadata"][
            "gpu_total_memory_bytes"
        ]
    )
    if gpu_count == 1:
        predicted = "refill"
    else:
        predicted = "active" if gpu_count == 2 else "refill"
    measured = "refill" if speedup >= 1.05 else "active"
    endpoint_passed = (
        endpoint["endpoint_max_energy_difference_meV_per_atom"] <= 5.0
        and endpoint["endpoint_convergence_mismatches"] == 0
    )
    memory_passed = refill[peak_key] / total_memory <= 0.85
    converged = (
        int(active["converged"]) == pool_size
        and int(refill["converged"]) == pool_size
    )
    return {
        "family": family,
        "split": split,
        "pool_size": pool_size,
        "gpu_count": gpu_count,
        "worker_pool_size": pool_size // gpu_count,
        "resident_capacity": resident_capacity,
        "pool_to_resident_ratio_per_worker": (
            pool_size / gpu_count / resident_capacity
        ),
        **_descriptors(manifest),
        "active_wall_seconds": active[timing_key],
        "refill_wall_seconds": refill[timing_key],
        "refill_speedup": speedup,
        "active_systems_per_second": pool_size / active[timing_key],
        "refill_systems_per_second": pool_size / refill[timing_key],
        "active_converged": int(active["converged"]),
        "refill_converged": int(refill["converged"]),
        "refill_peak_reserved_GiB_per_worker": refill[peak_key] / 2**30,
        "refill_peak_reserved_fraction": refill[peak_key] / total_memory,
        **endpoint,
        "endpoint_gate_passed": endpoint_passed,
        "memory_gate_passed": memory_passed,
        "all_jobs_converged": converged,
        "predicted_mode": predicted,
        "measured_speed_mode": measured,
        "speed_prediction_correct": predicted == measured,
        "selected_refill_scientifically_valid": (
            predicted != "refill"
            or (endpoint_passed and memory_passed and converged)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for split, families, raw_dir in (
        ("fit", FIT_SINGLE, args.fit_dir),
        ("heldout", HELDOUT_SINGLE, args.heldout_dir),
    ):
        for family in families:
            for pool_size, capacity in SINGLE_POINTS:
                stem = f"{family}_R{pool_size}_B{capacity}_G1"
                active = _load(raw_dir / f"{stem}_active.json")
                refill = _load(raw_dir / f"{stem}_refill.json")
                manifest = read_workload_manifest(
                    args.manifest_dir
                    / f"OPT-RFT-{family}-R{pool_size}-v1.json"
                )
                _validate_pair(
                    active,
                    refill,
                    manifest=manifest,
                    pool_size=pool_size,
                    gpu_count=1,
                    resident_capacity=capacity,
                )
                rows.append(
                    _row(
                        active,
                        refill,
                        family=family,
                        split=split,
                        manifest=manifest,
                        pool_size=pool_size,
                        gpu_count=1,
                        resident_capacity=capacity,
                    )
                )
    for split, families, raw_dir in (
        ("fit", FIT_MULTI, args.fit_dir),
        ("heldout", HELDOUT_MULTI, args.heldout_dir),
    ):
        for family in families:
            for pool_size, gpu_count, capacity in MULTI_POINTS:
                stem = (
                    f"{family}_R{pool_size}_B{capacity}_G{gpu_count}"
                )
                active = _load(raw_dir / f"{stem}_active.json")
                refill = _load(raw_dir / f"{stem}_refill.json")
                manifest = read_workload_manifest(
                    args.manifest_dir
                    / f"OPT-RFT-{family}-R{pool_size}-v1.json"
                )
                _validate_pair(
                    active,
                    refill,
                    manifest=manifest,
                    pool_size=pool_size,
                    gpu_count=gpu_count,
                    resident_capacity=capacity,
                )
                rows.append(
                    _row(
                        active,
                        refill,
                        family=family,
                        split=split,
                        manifest=manifest,
                        pool_size=pool_size,
                        gpu_count=gpu_count,
                        resident_capacity=capacity,
                    )
                )

    heldout_single = [
        row
        for row in rows
        if row["split"] == "heldout" and row["gpu_count"] == 1
    ]
    heldout_multi = [
        row
        for row in rows
        if row["split"] == "heldout" and row["gpu_count"] > 1
    ]
    summary = {
        "schema_version": 1,
        "baseline_commit": "fcc8fcb",
        "contract_validation": "passed",
        "timing_repeats": 1,
        "rows": rows,
        "validation": {
            "single_gpu_speed_predictions_correct": sum(
                row["speed_prediction_correct"]
                for row in heldout_single
            ),
            "single_gpu_speed_predictions_total": len(heldout_single),
            "single_gpu_scientific_gate_failures": sum(
                not row["selected_refill_scientifically_valid"]
                for row in heldout_single
            ),
            "multi_gpu_speed_predictions_correct": sum(
                row["speed_prediction_correct"]
                for row in heldout_multi
            ),
            "multi_gpu_speed_predictions_total": len(heldout_multi),
            "multi_gpu_scientific_gate_failures": sum(
                not row["selected_refill_scientifically_valid"]
                for row in heldout_multi
            ),
            "multi_gpu_policy_accepted": False,
        },
        "limitations": [
            "One timing observation per point.",
            "G4 methods ran sequentially on the same devices.",
            "Multi-GPU transfer failed held-out validation and remains active drain.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
