#!/usr/bin/env python3
"""Validate and summarize the contract-identical refill calibration matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from batch_mlip.workloads import read_workload_manifest  # noqa: E402

FIT_FAMILIES = (
    "GUFJOG44",
    "XATMOV88",
    "XAFPAY172",
    "OBEQIX220",
    "ROFB296",
    "ROFA-MIX",
)
HELD_OUT_FAMILIES = ("SOXLEX48", "AXOSOW64", "BOQWIN116")
FAMILIES = FIT_FAMILIES + HELD_OUT_FAMILIES
BATCH_SIZES = (32, 64, 128)
CHECKPOINT_SHA256 = (
    "b27c372cb2c2848ae1e54a9ffd7a2aa0b0401d9cc0ed922e3be250ff63e44486"
)
GPU_NAME = "NVIDIA H100 80GB HBM3"


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _validate_contract(
    result: dict[str, Any],
    *,
    family: str,
    method: str,
    batch_size: int,
    manifest_sha256: str,
) -> None:
    expected = {
        "status": "passed",
        "mlip": "atombit",
        "method": method,
        "optimizer": "bfgs",
        "jobs": 256,
        "workload_jobs": 256,
        "job_limit": None,
        "batch_size": batch_size,
        "fmax_eV_per_A": 0.05,
        "max_steps": 500,
        "linear_algebra_backend": "auto",
        "deterministic_algorithms": True,
        "cpu_threads": 1,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "workload_manifest_sha256": manifest_sha256,
    }
    for key, value in expected.items():
        _require_equal(f"{family}/B{batch_size}/{method}/{key}", result.get(key), value)

    contract = result["execution_contract"]
    expected_contract = {
        "model_dtype": "torch.float32",
        "optimizer_dtype": "torch.float64",
        "cutoff_A": 6.0,
        "skin_A": 0.5,
        "cell_filter": "BatchedFrechetCellFilter",
        "active_compaction": True,
        "timing_scope": "optimizer_only_after_one_model_warmup",
        "cuda_allocator_config": "expandable_segments:True",
        "cuda_allocator_environment": {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    }
    _require_equal(
        f"{family}/B{batch_size}/{method}/execution_contract",
        contract,
        expected_contract,
    )
    _require_equal(
        f"{family}/B{batch_size}/{method}/gpu_name",
        result["environment"]["gpu_name"],
        GPU_NAME,
    )
    _require_equal(
        f"{family}/B{batch_size}/{method}/gpu_total_memory_bytes",
        result["environment"]["gpu_total_memory_bytes"],
        85017886720,
    )
    _require_equal(
        f"{family}/B{batch_size}/{method}/allocator_environment",
        result["allocator_metrics"]["cuda_allocator_environment"],
        expected_contract["cuda_allocator_environment"],
    )
    _require_equal(
        f"{family}/B{batch_size}/{method}/allocation_retries",
        result["allocator_metrics"]["allocation_retries"],
        0,
    )
    _require_equal(
        f"{family}/B{batch_size}/{method}/out_of_memory_count",
        result["allocator_metrics"]["out_of_memory_count"],
        0,
    )
    if method == "active":
        for key in (
            "refill_storage",
            "refill_interval",
            "convergence_check_interval",
        ):
            _require_equal(
                f"{family}/B{batch_size}/{method}/{key}",
                result.get(key),
                None,
            )
    else:
        _require_equal(
            f"{family}/B{batch_size}/{method}/refill_storage",
            result["refill_storage"],
            "repack" if family == "ROFA-MIX" else "slots",
        )
        _require_equal(
            f"{family}/B{batch_size}/{method}/refill_interval",
            result["refill_interval"],
            1,
        )
        _require_equal(
            f"{family}/B{batch_size}/{method}/convergence_check_interval",
            result["convergence_check_interval"],
            1,
        )


def _flatten(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if isinstance(value, list):
            output.extend(_flatten(value))
        else:
            output.append(float(value))
    return output


def _rms(left: list[Any], right: list[Any]) -> float:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left_flat, right_flat, strict=True))
        / len(left_flat)
    )


def _endpoint_difference(
    refill: dict[str, Any], active: dict[str, Any]
) -> dict[str, Any]:
    active_records = {
        record["source"]: record for record in active["records"]
    }
    refill_records = {
        record["source"]: record for record in refill["records"]
    }
    _require_equal("active/refill sources", refill_records.keys(), active_records.keys())
    energy = []
    position = []
    cell = []
    convergence_mismatches = 0
    for source, current in refill_records.items():
        control = active_records[source]
        atom_count = len(current["positions_A"])
        energy.append(
            abs(current["energy_eV"] - control["energy_eV"])
            * 1000.0
            / atom_count
        )
        position.append(_rms(current["positions_A"], control["positions_A"]))
        cell.append(_rms(current["cell_A"], control["cell_A"]))
        convergence_mismatches += current["converged"] != control["converged"]
    return {
        "convergence_mismatches": convergence_mismatches,
        "max_energy_difference_meV_per_atom": max(energy),
        "energy_difference_above_5_meV_per_atom": sum(
            value > 5.0 for value in energy
        ),
        "max_position_rmsd_A": max(position),
        "max_cell_rmsd_A": max(cell),
    }


def _coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _manifest_features(path: Path) -> dict[str, Any]:
    manifest = read_workload_manifest(path)
    atoms = [float(job.atom_count) for job in manifest.jobs]
    edges = [
        float(job.topology_edge_counts["cutoff=6.000_skin=0.500"])
        for job in manifest.jobs
    ]
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "mean_atom_count": statistics.fmean(atoms),
        "atom_count_cv": _coefficient_of_variation(atoms),
        "mean_edge_count": statistics.fmean(edges),
        "edge_count_cv": _coefficient_of_variation(edges),
        "mean_edges_per_atom": statistics.fmean(
            edge / atom for edge, atom in zip(edges, atoms, strict=True)
        ),
        "homogeneous_atom_count": len(set(atoms)) == 1,
        "unique_structure_count": len(
            {job.normalized_structure_sha256 for job in manifest.jobs}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=(
            ROOT
            / "experiments"
            / "refill-offline-predictor"
            / "workloads"
            / "manifests"
        ),
    )
    parser.add_argument("--manifest-prefix", default="OPT-RF-U256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    pair_details = []
    contract_versions: set[tuple[str, ...]] = set()
    for family in FAMILIES:
        manifest_path = (
            args.manifest_dir
            / f"{args.manifest_prefix}-{family}-R256-v1.json"
        )
        features = _manifest_features(manifest_path)
        if features["unique_structure_count"] != 256:
            raise ValueError(
                f"{family} contains {features['unique_structure_count']} "
                "unique structures; predictor calibration requires 256"
            )
        for batch_size in BATCH_SIZES:
            pair = {}
            for method in ("active", "refill"):
                path = args.raw_dir / f"{family}_B{batch_size}_{method}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                _validate_contract(
                    result,
                    family=family,
                    method=method,
                    batch_size=batch_size,
                    manifest_sha256=features["manifest_sha256"],
                )
                contract_versions.add(
                    (
                        result["environment"]["torch"],
                        result["environment"]["cuda_version"],
                        result["environment"]["ase"],
                        result["environment"]["numpy"],
                    )
                )
                pair[method] = result
            active = pair["active"]
            refill = pair["refill"]
            endpoint = _endpoint_difference(refill, active)
            active_sizes = _flatten(active["active_batch_sizes"])
            refill_sizes = _flatten(refill["active_batch_sizes"])
            refill_speedup = active["timing_seconds"] / refill["timing_seconds"]
            row = {
                "family": family,
                "split": "fit" if family in FIT_FAMILIES else "heldout",
                "batch_size": batch_size,
                "pool_size": 256,
                "pool_to_resident_ratio": 256 / batch_size,
                "resident_atoms": (
                    batch_size * features["mean_atom_count"]
                ),
                **{
                    key: value
                    for key, value in features.items()
                    if key != "manifest_sha256"
                },
                "refill_storage": (
                    "repack" if family == "ROFA-MIX" else "slots"
                ),
                "active_wall_seconds": active["timing_seconds"],
                "refill_wall_seconds": refill["timing_seconds"],
                "refill_speedup": refill_speedup,
                "active_systems_per_second": active["systems_per_second"],
                "refill_systems_per_second": refill["systems_per_second"],
                "active_atoms_per_second": (
                    active["systems_per_second"]
                    * features["mean_atom_count"]
                ),
                "refill_atoms_per_second": (
                    refill["systems_per_second"]
                    * features["mean_atom_count"]
                ),
                "active_mean_occupancy": (
                    statistics.fmean(active_sizes) / batch_size
                ),
                "refill_mean_occupancy": (
                    statistics.fmean(refill_sizes) / batch_size
                ),
                "active_model_evaluations": active["model_evaluations"],
                "refill_model_evaluations": refill["model_evaluations"],
                "active_graph_evaluations": active["graph_evaluations"],
                "refill_graph_evaluations": refill["graph_evaluations"],
                "active_peak_reserved_GiB": (
                    active["peak_reserved_bytes"] / 2**30
                ),
                "refill_peak_reserved_GiB": (
                    refill["peak_reserved_bytes"] / 2**30
                ),
                "refill_peak_reserved_fraction": (
                    refill["peak_reserved_bytes"]
                    / refill["environment"]["gpu_total_memory_bytes"]
                ),
                "active_converged": active["converged"],
                "refill_converged": refill["converged"],
                "endpoint_max_energy_difference_meV_per_atom": endpoint[
                    "max_energy_difference_meV_per_atom"
                ],
                "endpoint_energy_outliers_above_5_meV_per_atom": endpoint[
                    "energy_difference_above_5_meV_per_atom"
                ],
                "endpoint_convergence_mismatches": endpoint[
                    "convergence_mismatches"
                ],
                "endpoint_gate_passed": (
                    endpoint["convergence_mismatches"] == 0
                    and endpoint["max_energy_difference_meV_per_atom"] <= 5.0
                ),
                "memory_gate_passed": (
                    refill["peak_reserved_bytes"]
                    / refill["environment"]["gpu_total_memory_bytes"]
                    <= 0.85
                ),
            }
            rows.append(row)
            pair_details.append(
                {
                    "family": family,
                    "batch_size": batch_size,
                    "endpoint_difference": endpoint,
                }
            )

    if len(contract_versions) != 1:
        raise ValueError(
            f"software environment differs across runs: {sorted(contract_versions)}"
        )
    excluded_families = sorted(
        {
            row["family"]
            for row in rows
            if not row["endpoint_gate_passed"]
        }
    )
    for row in rows:
        row["training_eligible"] = (
            row["split"] == "fit"
            and row["family"] not in excluded_families
            and row["memory_gate_passed"]
            and row["active_converged"] == 256
            and row["refill_converged"] == 256
        )
        row["predicted_mode"] = (
            "refill"
            if (
                row["refill_storage"] == "slots"
                and row["resident_atoms"] <= 12000.0
                and row["memory_gate_passed"]
            )
            else "active"
        )
        row["measured_speed_mode"] = (
            "refill" if row["refill_speedup"] >= 1.05 else "active"
        )
        row["speed_prediction_correct"] = (
            row["predicted_mode"] == row["measured_speed_mode"]
        )
        row["selected_refill_scientifically_valid"] = (
            row["predicted_mode"] != "refill"
            or (
                row["endpoint_gate_passed"]
                and row["memory_gate_passed"]
                and row["active_converged"] == 256
                and row["refill_converged"] == 256
            )
        )

    held_out = [row for row in rows if row["split"] == "heldout"]
    summary = {
        "schema_version": 1,
        "baseline_commit": "4d30a44",
        "contract_validation": "passed",
        "software_environment": list(next(iter(contract_versions))),
        "timing_repeats": 1,
        "rows": rows,
        "pair_details": pair_details,
        "excluded_families": excluded_families,
        "training_rows": sum(row["training_eligible"] for row in rows),
        "held_out_rows": len(held_out),
        "locked_policy_validation": {
            "speed_predictions_correct": sum(
                row["speed_prediction_correct"] for row in held_out
            ),
            "speed_predictions_total": len(held_out),
            "selected_refill_cases": sum(
                row["predicted_mode"] == "refill" for row in held_out
            ),
            "selected_refill_speed_losses": sum(
                row["predicted_mode"] == "refill"
                and row["refill_speedup"] < 1.05
                for row in held_out
            ),
            "selected_refill_scientific_gate_failures": sum(
                not row["selected_refill_scientifically_valid"]
                for row in held_out
            ),
        },
        "limitations": [
            "One timing observation per matrix point.",
            "Each family contains 256 unique, deterministically selected structures.",
            "Prediction is restricted to the validated execution contract.",
            "Descriptor matching does not extrapolate beyond measured pool sizes and capacities.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
