#!/usr/bin/env python3
"""Summarize compatible-refill timings, mechanisms, and endpoint gates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw"


def _identity(path: Path) -> tuple[str, int, str]:
    family, batch_mode = path.stem.split("_B", 1)
    batch, mode = batch_mode.split("_", 1)
    return family, int(batch), mode


def _phase_seconds(payload: dict, prefix: str) -> float:
    return sum(
        float(values.get("total_seconds", 0.0))
        for name, values in payload["runtime_profile"]["phases"].items()
        if name.startswith(prefix)
    )


def _endpoint(reference: dict, candidate: dict) -> dict[str, float | int]:
    expected = {record["source"]: record for record in reference["records"]}
    observed = {record["source"]: record for record in candidate["records"]}
    if expected.keys() != observed.keys():
        raise ValueError("endpoint source sets differ")
    energy = []
    position = []
    cell = []
    step_difference = []
    convergence_mismatch = 0
    for source, left in expected.items():
        right = observed[source]
        atom_count = len(left["positions_A"])
        energy.append(abs(left["energy_eV"] - right["energy_eV"]) * 1000 / atom_count)
        position.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["positions_A"])
                        - np.asarray(right["positions_A"])
                    )
                )
            )
        )
        cell.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left["cell_A"])
                        - np.asarray(right["cell_A"])
                    )
                )
            )
        )
        step_difference.append(abs(int(left["steps"]) - int(right["steps"])))
        convergence_mismatch += int(bool(left["converged"]) != bool(right["converged"]))
    return {
        "max_energy_difference_meV_per_atom": max(energy, default=0.0),
        "max_position_component_difference_A": max(position, default=0.0),
        "max_cell_component_difference_A": max(cell, default=0.0),
        "max_step_difference": max(step_difference, default=0),
        "convergence_mismatches": convergence_mismatch,
    }


def main() -> None:
    payloads = {
        _identity(path): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RAW.glob("*.json"))
    }
    rows = []
    for (family, batch, mode), payload in sorted(payloads.items()):
        active = payloads[(family, batch, "active")]
        fifo = payloads.get((family, batch, "fifo_slots"))
        events = [
            event
            for event in payload["runtime_profile"]["events"]
            if event.get("name") == "refill"
        ]
        storage_counts: dict[str, int] = {}
        for event in events:
            storage = str(event.get("storage"))
            storage_counts[storage] = storage_counts.get(storage, 0) + 1
        timing = float(payload["timing_seconds"])
        endpoint = _endpoint(active, payload)
        row = {
            "family": family,
            "batch_size": batch,
            "mode": mode,
            "seconds": timing,
            "systems_per_second": float(payload["systems_per_second"]),
            "speedup_over_active": float(active["timing_seconds"]) / timing,
            "speedup_over_fifo": (
                None
                if fifo is None or mode == "active"
                else float(fifo["timing_seconds"]) / timing
            ),
            "model_evaluations": int(payload["model_evaluations"]),
            "graph_evaluations": int(payload["graph_evaluations"]),
            "mean_active_occupancy": (
                sum(payload["active_batch_sizes"][0])
                / len(payload["active_batch_sizes"][0])
                / batch
            ),
            "peak_allocated_GiB": payload["peak_allocated_bytes"] / 2**30,
            "peak_reserved_GiB": payload["peak_reserved_bytes"] / 2**30,
            "refill_events": len(events),
            "slot_events": storage_counts.get("slots", 0),
            "repack_events": storage_counts.get("repack", 0),
            "arena_events": storage_counts.get("arena", 0),
            "complete_compatible_matches": sum(
                bool(event.get("compatible_match_complete")) for event in events
            ),
            "refill_phase_seconds": _phase_seconds(payload, "scheduler.refill"),
            "neighbor_phase_seconds": _phase_seconds(payload, "graph."),
            "all_jobs_converged": payload["converged"] == payload["jobs"],
            **endpoint,
        }
        row["endpoint_gate_pass"] = (
            row["all_jobs_converged"]
            and row["convergence_mismatches"] == 0
            and row["max_energy_difference_meV_per_atom"] <= 5.0
        )
        if not math.isfinite(row["seconds"]):
            raise ValueError("non-finite timing")
        rows.append(row)

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['family']:10s} B{row['batch_size']:<3d} "
            f"{row['mode']:18s} {row['seconds']:7.3f}s "
            f"active={row['speedup_over_active']:.3f}x "
            f"fifo={row['speedup_over_fifo'] or 0.0:.3f}x "
            f"slots/repack={row['slot_events']}/{row['repack_events']} "
            f"endpoint={row['endpoint_gate_pass']}"
        )


if __name__ == "__main__":
    main()
