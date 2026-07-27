#!/usr/bin/env python3
"""Summarize automatic, manual, and MPS optimization policy results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _case(
    raw: Path,
    *,
    family: str,
    pool_size: int,
    mlip: str,
    optimizer: str,
) -> dict[str, Any]:
    suffix = f"{mlip}_{optimizer}_{family}_R{pool_size}_G1.json"
    paths = {
        "automatic": raw / f"auto_{suffix}",
        "manual": raw / f"manual_{suffix}",
        "mps": raw / f"mps_{suffix}",
    }
    automatic = _read(paths["automatic"])
    manual = _read(paths["manual"])
    mps = _read(paths["mps"])
    call = automatic["call_records"][0]
    schedule = call["schedule"]
    production_seconds = float(schedule["production_run_seconds"])
    total_seconds = float(call["wall_time_s"])
    manual_seconds = float(manual["timing_seconds"])
    mps_seconds = float(mps["timing"]["wall_seconds"])
    gpu_total = int(automatic["environment"]["gpu_total_memory_bytes"])
    worker_peak = int(call["peak_reserved_bytes"])
    probe_peak = int(schedule["probe"]["peak_reserved_bytes"])
    conservative_peak = worker_peak + probe_peak
    converged = {
        "automatic": int(call["converged"]),
        "manual": int(manual["converged"]),
        "mps": int(mps["converged"]),
    }
    return {
        "case_id": f"{family}-R{pool_size}-{mlip}-{optimizer}",
        "family": family,
        "pool_size": pool_size,
        "mlip": mlip,
        "optimizer": optimizer,
        "jobs": int(automatic["jobs"]),
        "selected_chunks": [
            int(chunk["system_count"])
            for chunk in schedule["planned_chunks"]
        ],
        "timing_seconds": {
            "automatic_production": production_seconds,
            "automatic_total_api": total_seconds,
            "manual_active_drain": manual_seconds,
            "ase_cuda_mps_4": mps_seconds,
            "planning": float(schedule["planning_seconds"]),
            "worker_startup": float(
                schedule["worker_startup_seconds_this_call"]
            ),
        },
        "throughput": {
            "automatic_production_systems_per_second": (
                pool_size / production_seconds
            ),
            "automatic_total_systems_per_second": pool_size / total_seconds,
            "manual_systems_per_second": pool_size / manual_seconds,
            "mps_systems_per_second": pool_size / mps_seconds,
            "automatic_production_fraction_of_manual": (
                manual_seconds / production_seconds
            ),
            "automatic_total_fraction_of_manual": manual_seconds / total_seconds,
            "automatic_production_speedup_over_mps": (
                mps_seconds / production_seconds
            ),
            "automatic_total_speedup_over_mps": mps_seconds / total_seconds,
        },
        "memory": {
            "gpu_total_bytes": gpu_total,
            "worker_peak_allocated_bytes": int(
                call["peak_allocated_bytes"]
            ),
            "worker_peak_reserved_bytes": worker_peak,
            "parent_probe_peak_reserved_bytes": probe_peak,
            "conservative_device_peak_bound_bytes": conservative_peak,
            "conservative_device_peak_fraction": conservative_peak / gpu_total,
            "manual_peak_reserved_bytes": int(manual["peak_reserved_bytes"]),
            "mps_peak_device_bytes": int(
                mps["peak_gpu_memory_bytes_nvidia_smi"]
            ),
        },
        "convergence": {
            **converged,
            "match": len(set(converged.values())) == 1,
            "all_jobs_converged": all(
                value == pool_size for value in converged.values()
            ),
        },
        "gates": {
            "memory_at_most_85_percent": conservative_peak / gpu_total <= 0.85,
            "production_within_5_percent_of_manual": (
                manual_seconds / production_seconds >= 0.95
            ),
            "production_faster_than_mps": mps_seconds > production_seconds,
            "total_api_faster_than_mps": mps_seconds > total_seconds,
            "convergence_match": len(set(converged.values())) == 1,
        },
        "raw": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }


def _correctness_case(
    raw: Path,
    correctness: Path,
    *,
    family: str,
    mlip: str,
    optimizer: str,
) -> dict[str, Any]:
    automatic_path = correctness / (
        f"auto_{mlip}_{optimizer}_{family}_R64.json"
    )
    manual_path = raw / f"manual_{mlip}_{optimizer}_{family}_R64_G1.json"
    automatic = _read(automatic_path)["call_records"][0]
    manual = _read(manual_path)
    tensors = automatic["final_tensors"]
    records = manual["records"]
    atoms_per_system = len(records[0]["positions_A"])
    positions = np.asarray(tensors["positions_A"]).reshape(
        len(records),
        atoms_per_system,
        3,
    )
    reference_positions = np.asarray(
        [record["positions_A"] for record in records]
    )
    cells = np.asarray(tensors["cells_A"])
    reference_cells = np.asarray([record["cell_A"] for record in records])
    energies = np.asarray(tensors["energies_eV"])
    reference_energies = np.asarray(
        [record["energy_eV"] for record in records]
    )
    energy_error_per_atom = (
        np.abs(energies - reference_energies) / atoms_per_system
    )
    position_rmsd = np.sqrt(
        np.mean(np.square(positions - reference_positions), axis=(1, 2))
    )
    cell_rmsd = np.sqrt(
        np.mean(np.square(cells - reference_cells), axis=(1, 2))
    )
    automatic_steps = automatic["converged_steps"]
    manual_steps = [record["steps"] for record in records]
    return {
        "case_id": f"{family}-R64-{mlip}-{optimizer}",
        "jobs": len(records),
        "convergence_flags_match": int(automatic["converged"])
        == int(manual["converged"]),
        "step_mismatches": sum(
            left != right
            for left, right in zip(
                automatic_steps,
                manual_steps,
                strict=True,
            )
        ),
        "max_energy_error_eV_per_atom": float(
            np.max(energy_error_per_atom)
        ),
        "median_energy_error_eV_per_atom": float(
            np.median(energy_error_per_atom)
        ),
        "jobs_above_1_meV_per_atom": int(
            np.sum(energy_error_per_atom > 1e-3)
        ),
        "max_position_rmsd_A": float(np.max(position_rmsd)),
        "max_cell_rmsd_A": float(np.max(cell_rmsd)),
        "strict_1_meV_per_atom_gate": bool(
            np.max(energy_error_per_atom) <= 1e-3
        ),
        "interpretation": (
            "long-run full-BFGS local-minimum sensitivity; fixed-step "
            "equations were validated separately"
            if optimizer == "bfgs"
            else "scheduler/reassembly agreement"
        ),
        "raw": {
            "automatic": {
                "path": str(automatic_path),
                "sha256": _sha256(automatic_path),
            },
            "manual": {
                "path": str(manual_path),
                "sha256": _sha256(manual_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [
        _case(
            args.raw_dir,
            family=family,
            pool_size=pool_size,
            mlip=mlip,
            optimizer=optimizer,
        )
        for family in ("AXOSOW", "XAFPAY")
        for pool_size in (64, 256)
        for mlip in ("atombit", "mace")
        for optimizer in ("fire", "bfgs")
    ]
    correctness_cases = [
        _correctness_case(
            args.raw_dir,
            args.correctness_dir,
            family=family,
            mlip=mlip,
            optimizer=optimizer,
        )
        for family in ("AXOSOW", "XAFPAY")
        for mlip in ("atombit", "mace")
        for optimizer in ("fire", "bfgs")
    ]
    production_manual = [
        case["throughput"]["automatic_production_fraction_of_manual"]
        for case in cases
    ]
    production_mps = [
        case["throughput"]["automatic_production_speedup_over_mps"]
        for case in cases
    ]
    total_mps = [
        case["throughput"]["automatic_total_speedup_over_mps"]
        for case in cases
    ]
    summary = {
        "schema_version": 1,
        "status": "stage_1_complete_stage_2_blocked",
        "cases": len(cases),
        "production_within_5_percent_of_manual": sum(
            case["gates"]["production_within_5_percent_of_manual"]
            for case in cases
        ),
        "production_faster_than_mps": sum(
            case["gates"]["production_faster_than_mps"] for case in cases
        ),
        "total_api_faster_than_mps": sum(
            case["gates"]["total_api_faster_than_mps"] for case in cases
        ),
        "memory_safe": sum(
            case["gates"]["memory_at_most_85_percent"] for case in cases
        ),
        "convergence_match": sum(
            case["gates"]["convergence_match"] for case in cases
        ),
        "all_jobs_converged": sum(
            case["convergence"]["all_jobs_converged"] for case in cases
        ),
        "geometric_mean": {
            "automatic_production_fraction_of_manual": _geometric_mean(
                production_manual
            ),
            "automatic_production_speedup_over_mps": _geometric_mean(
                production_mps
            ),
            "automatic_total_speedup_over_mps": _geometric_mean(total_mps),
        },
        "planning_seconds": {
            "median": statistics.median(
                case["timing_seconds"]["planning"] for case in cases
            ),
            "maximum": max(
                case["timing_seconds"]["planning"] for case in cases
            ),
        },
        "worker_startup_seconds": {
            "median": statistics.median(
                case["timing_seconds"]["worker_startup"] for case in cases
            ),
            "maximum": max(
                case["timing_seconds"]["worker_startup"] for case in cases
            ),
        },
        "maximum_conservative_memory_fraction": max(
            case["memory"]["conservative_device_peak_fraction"]
            for case in cases
        ),
        "failed_manual_throughput_cases": [
            case["case_id"]
            for case in cases
            if not case["gates"]["production_within_5_percent_of_manual"]
        ],
        "incomplete_convergence_cases": [
            case["case_id"]
            for case in cases
            if not case["convergence"]["all_jobs_converged"]
        ],
        "strict_numerical_cases_passed": sum(
            case["strict_1_meV_per_atom_gate"]
            and case["convergence_flags_match"]
            for case in correctness_cases
        ),
        "strict_numerical_cases_total": len(correctness_cases),
        "strict_numerical_failures": [
            case["case_id"]
            for case in correctness_cases
            if not case["strict_1_meV_per_atom_gate"]
            or not case["convergence_flags_match"]
        ],
    }
    output = {
        "schema_version": 1,
        "experiment": "end-to-end-optimization-policy",
        "summary": summary,
        "cases": cases,
        "correctness_cases": correctness_cases,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
