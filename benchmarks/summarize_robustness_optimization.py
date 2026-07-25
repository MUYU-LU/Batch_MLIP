#!/usr/bin/env python3
"""Summarize signed-workload ASE, MPS, and tensor BFGS benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TENSOR_POINTS = {
    "GUFJOG44": (("active", 128), ("active", 256)),
    "XATMOV88": (("active", 128), ("active", 256)),
    "OBEQIX220": (
        ("active", 128),
        ("active", 192),
        ("refill", 192),
    ),
    "ROFA-MIX": (
        ("active", 128),
        ("active", 192),
        ("active", 256),
        ("refill", 192),
    ),
}
SELECTED_POLICY = {
    "GUFJOG44": ("active", 256),
    "XATMOV88": ("active", 256),
    "OBEQIX220": ("active", 128),
    "ROFA-MIX": ("active", 256),
}
TOLERANCES = {
    "max_energy_error_eV_per_atom": 1e-4,
    "max_final_fmax_error_eV_per_A": 0.03,
    "max_stress_tensor_error_eV_per_A3": 0.01,
    "max_position_rmsd_A": 0.02,
    "max_cell_rmsd_A": 0.02,
    "max_step_difference": 25,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mps_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for worker in result["worker_results"]
        for record in worker["records"]
    ]


def _converged(result: dict[str, Any]) -> int:
    return int(
        result.get(
            "converged",
            sum(bool(record["converged"]) for record in result["records"]),
        )
    )


def _validate(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    references = {record["source"]: record for record in reference}
    candidates = {record["source"]: record for record in candidate}
    if references.keys() != candidates.keys():
        raise ValueError("signed job identifiers differ")

    values = {name: [] for name in TOLERANCES}
    convergence_flags_match = True
    failed_job_ids = []
    for source, expected in references.items():
        actual = candidates[source]
        atom_count = len(expected["positions_A"])
        convergence_matches = bool(expected["converged"]) == bool(
            actual["converged"]
        )
        convergence_flags_match &= convergence_matches
        job_metrics = {
            "max_energy_error_eV_per_atom": (
                abs(float(actual["energy_eV"]) - float(expected["energy_eV"]))
                / atom_count
            ),
            "max_final_fmax_error_eV_per_A": abs(
                float(actual["max_force_eV_per_A"])
                - float(expected["max_force_eV_per_A"])
            ),
        }
        stress_error = np.asarray(actual["stress_eV_per_A3"]) - np.asarray(
            expected["stress_eV_per_A3"]
        )
        job_metrics["max_stress_tensor_error_eV_per_A3"] = float(
            np.abs(stress_error).max()
        )
        position_error = np.asarray(actual["positions_A"]) - np.asarray(
            expected["positions_A"]
        )
        job_metrics["max_position_rmsd_A"] = float(
            np.sqrt(np.mean(np.square(position_error)))
        )
        cell_error = np.asarray(actual["cell_A"]) - np.asarray(
            expected["cell_A"]
        )
        job_metrics["max_cell_rmsd_A"] = float(
            np.sqrt(np.mean(np.square(cell_error)))
        )
        job_metrics["max_step_difference"] = abs(
            int(actual["steps"]) - int(expected["steps"])
        )
        for name, value in job_metrics.items():
            values[name].append(value)
        if not convergence_matches or any(
            job_metrics[name] > limit for name, limit in TOLERANCES.items()
        ):
            failed_job_ids.append(source)
    metrics = {name: max(samples) for name, samples in values.items()}
    failed = [
        name for name, limit in TOLERANCES.items() if metrics[name] > limit
    ]
    if not convergence_flags_match:
        failed.append("convergence_flags_match")
    return {
        **metrics,
        "distributions": {
            name: {
                "median": float(np.median(samples)),
                "percentile_95": float(np.percentile(samples, 95)),
                "maximum": float(np.max(samples)),
            }
            for name, samples in values.items()
        },
        "convergence_flags_match": convergence_flags_match,
        "failed_job_count": len(failed_job_ids),
        "failed_job_ids": failed_job_ids,
        "failed_checks": failed,
        "passed": not failed,
        "tolerances": TOLERANCES,
    }


def _tensor_path(
    raw_dir: Path,
    mlip: str,
    workload: str,
    method: str,
    batch_size: int,
) -> Path:
    return (
        raw_dir
        / f"tensor_{mlip}_{workload}_{method}_B{batch_size}.json"
    )


def _tensor_candidates(
    raw_dir: Path,
    mlip: str,
    workload: str,
) -> list[dict[str, Any]]:
    candidates = []
    for method, batch_size in TENSOR_POINTS[workload]:
        path = _tensor_path(raw_dir, mlip, workload, method, batch_size)
        data = _load(path)
        candidates.append(
            {
                "method": method,
                "batch_size": batch_size,
                "seconds": data["timing_seconds"],
                "systems_per_second": data["systems_per_second"],
                "converged": _converged(data),
                "optimizer_steps_total": data["optimizer_steps_total"],
                "model_evaluations": data["model_evaluations"],
                "peak_allocated_bytes": data["peak_allocated_bytes"],
                "peak_reserved_bytes": data["peak_reserved_bytes"],
                "path": str(path),
                "sha256": _sha256(path),
                "_data": data,
            }
        )
    return candidates


def summarize(raw_dir: Path) -> dict[str, Any]:
    rows = []
    for mlip in ("atombit", "mace"):
        for workload in TENSOR_POINTS:
            ase_path = raw_dir / f"confirm_ase1_{mlip}_{workload}.json"
            mps_path = raw_dir / f"mps32_{mlip}_{workload}.json"
            ase = _load(ase_path)
            mps = _load(mps_path)
            candidates = _tensor_candidates(raw_dir, mlip, workload)
            valid = [
                candidate
                for candidate in candidates
                if candidate["converged"] == 256
                and candidate["seconds"] is not None
            ]
            if not valid:
                raise ValueError(f"no fully converged tensor point for {mlip}/{workload}")
            fastest_screen = min(valid, key=lambda item: item["seconds"])
            selected_method, selected_batch_size = SELECTED_POLICY[workload]
            selected_path = (
                raw_dir
                / (
                    f"confirm_tensor_{mlip}_{workload}_"
                    f"{selected_method}_B{selected_batch_size}.json"
                )
            )
            tensor = _load(selected_path)
            for candidate in candidates:
                candidate.pop("_data", None)
            ase_seconds = float(ase["timing_seconds"])
            mps_seconds = float(mps["timing"]["wall_seconds"])
            tensor_seconds = float(tensor["timing_seconds"])
            rows.append(
                {
                    "mlip": mlip,
                    "workload": workload,
                    "jobs": 256,
                    "selected_tensor_method": selected_method,
                    "selected_batch_size": selected_batch_size,
                    "fastest_screen_method": fastest_screen["method"],
                    "fastest_screen_batch_size": fastest_screen["batch_size"],
                    "selected_screen_gap_fraction": (
                        float(
                            next(
                                candidate["seconds"]
                                for candidate in valid
                                if candidate["method"] == selected_method
                                and candidate["batch_size"] == selected_batch_size
                            )
                        )
                        / float(fastest_screen["seconds"])
                        - 1.0
                    ),
                    "ase1_seconds": ase_seconds,
                    "mps32_seconds": mps_seconds,
                    "tensor_seconds": tensor_seconds,
                    "tensor_speedup_vs_ase1": ase_seconds / tensor_seconds,
                    "tensor_speedup_vs_mps32": mps_seconds / tensor_seconds,
                    "ase1_peak_allocated_bytes": ase["peak_allocated_bytes"],
                    "ase1_peak_reserved_bytes": ase["peak_reserved_bytes"],
                    "mps32_peak_nvidia_smi_bytes": mps[
                        "peak_gpu_memory_bytes_nvidia_smi"
                    ],
                    "tensor_peak_allocated_bytes": tensor[
                        "peak_allocated_bytes"
                    ],
                    "tensor_peak_reserved_bytes": tensor[
                        "peak_reserved_bytes"
                    ],
                    "ase1_converged": _converged(ase),
                    "mps32_converged": mps["converged"],
                    "tensor_converged": _converged(tensor),
                    "tensor_vs_ase1_validation": _validate(
                        ase["records"],
                        tensor["records"],
                    ),
                    "mps32_vs_ase1_validation": _validate(
                        ase["records"],
                        _mps_records(mps),
                    ),
                    "tensor_candidates": candidates,
                    "raw_files": {
                        "ase1": {
                            "path": str(ase_path),
                            "sha256": _sha256(ase_path),
                        },
                        "mps32": {
                            "path": str(mps_path),
                            "sha256": _sha256(mps_path),
                        },
                        "selected_tensor": {
                            "path": str(selected_path),
                            "sha256": _sha256(selected_path),
                        },
                    },
                }
            )
    diagnostic_dir = raw_dir.parent / "diagnostics"
    b1_diagnostics = []
    for mlip in ("atombit", "mace"):
        for workload in ("GUFJOG44", "OBEQIX220"):
            selected_method, selected_batch_size = SELECTED_POLICY[workload]
            ase_path = raw_dir / f"confirm_ase1_{mlip}_{workload}.json"
            b1_path = diagnostic_dir / f"b1_{mlip}_{workload}_J32.json"
            tensor_path = (
                raw_dir
                / (
                    f"confirm_tensor_{mlip}_{workload}_"
                    f"{selected_method}_B{selected_batch_size}.json"
                )
            )
            ase_records = _load(ase_path)["records"][:32]
            b1_records = _load(b1_path)["records"]
            tensor_records = _load(tensor_path)["records"][:32]
            b1_diagnostics.append(
                {
                    "mlip": mlip,
                    "workload": workload,
                    "jobs": 32,
                    "b1_vs_ase1": _validate(ase_records, b1_records),
                    "selected_batch_vs_b1": _validate(
                        b1_records,
                        tensor_records,
                    ),
                    "raw_files": {
                        "ase1": {
                            "path": str(ase_path),
                            "sha256": _sha256(ase_path),
                        },
                        "b1": {
                            "path": str(b1_path),
                            "sha256": _sha256(b1_path),
                        },
                        "selected_tensor": {
                            "path": str(tensor_path),
                            "sha256": _sha256(tensor_path),
                        },
                    },
                }
            )
    return {
        "schema_version": 1,
        "status": "complete",
        "timing_repeats": 1,
        "timing_interpretation": (
            "single-run screen; differences below 2 percent are inconclusive"
        ),
        "rows": rows,
        "b1_diagnostics": b1_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = summarize(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "mlip",
            "workload",
            "selected_tensor_method",
            "selected_batch_size",
            "ase1_seconds",
            "mps32_seconds",
            "tensor_seconds",
            "tensor_speedup_vs_ase1",
            "tensor_speedup_vs_mps32",
            "ase1_peak_allocated_bytes",
            "ase1_peak_reserved_bytes",
            "mps32_peak_nvidia_smi_bytes",
            "tensor_peak_allocated_bytes",
            "tensor_peak_reserved_bytes",
        )
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(result["rows"])
    print(json.dumps({"output": str(args.output), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
