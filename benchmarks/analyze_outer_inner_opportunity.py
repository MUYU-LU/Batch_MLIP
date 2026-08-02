#!/usr/bin/env python3
"""Classify passive multi-GPU scheduling opportunity from one frozen run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ShapeKey = tuple[int, int]


@dataclass(frozen=True)
class TaskTrace:
    task_index: int
    worker_id: int
    bucket_index: int
    system_indices: tuple[int, ...]
    dispatch: float
    completion: float
    profile_end: float
    input_ready_on_dispatch: bool | None
    events: tuple[dict[str, Any], ...]


def load_shape_keys(path: str | Path) -> tuple[ShapeKey, ...]:
    """Read fixed-slot compatibility keys from a signed planning sidecar."""

    payload = json.loads(Path(path).read_text())
    systems = payload.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError("planning profile contains no systems")
    keys: list[ShapeKey | None] = [None] * len(systems)
    for system in systems:
        structure = system["structure"]
        task = system["task_auxiliary"]
        index = int(structure["index"])
        if index != int(task["index"]):
            raise ValueError("planning profile layers have different indices")
        keys[index] = (
            int(structure["atom_count"]),
            int(task["generalized_dimension"]),
        )
    if any(key is None for key in keys):
        raise ValueError("planning profile indices are not contiguous")
    return tuple(key for key in keys if key is not None)


def _task_traces(payload: dict[str, Any]) -> tuple[TaskTrace, ...]:
    scheduling = payload["scheduling"]
    traces = []
    for worker in scheduling["workers"]:
        worker_id = int(worker["worker_id"])
        for chunk in worker["chunks"]:
            dispatch = chunk.get("dispatch_offset_seconds")
            completion = chunk.get("completion_offset_seconds")
            indices = chunk.get("system_indices")
            if dispatch is None or completion is None or indices is None:
                raise ValueError(
                    "run lacks passive task offsets or system indices; rerun with "
                    "outer-inner opportunity telemetry enabled"
                )
            runtime = chunk.get("runtime_profile") or {}
            total = float(runtime.get("total_seconds", 0.0))
            input_metadata = chunk.get("input") or {}
            traces.append(
                TaskTrace(
                    task_index=int(chunk["task_index"]),
                    worker_id=worker_id,
                    bucket_index=int(chunk["bucket_index"]),
                    system_indices=tuple(int(index) for index in indices),
                    dispatch=float(dispatch),
                    completion=float(completion),
                    profile_end=min(float(completion), float(dispatch) + total),
                    input_ready_on_dispatch=input_metadata.get("ready_on_dispatch"),
                    events=tuple(runtime.get("events", ())),
                )
            )
    ordered = tuple(sorted(traces, key=lambda trace: trace.task_index))
    if [trace.task_index for trace in ordered] != list(range(len(ordered))):
        raise ValueError("task indices are not contiguous")
    return ordered


def _pending_tasks(traces: tuple[TaskTrace, ...], at: float) -> tuple[TaskTrace, ...]:
    return tuple(trace for trace in traces if trace.dispatch > at)


def _busy_workers(traces: tuple[TaskTrace, ...], at: float) -> set[int]:
    return {
        trace.worker_id
        for trace in traces
        if trace.dispatch <= at < trace.completion
    }


def _pending_shapes(
    pending: tuple[TaskTrace, ...],
    shape_keys: tuple[ShapeKey, ...],
) -> Counter[ShapeKey]:
    return Counter(
        shape_keys[index]
        for trace in pending
        for index in trace.system_indices
    )


def _classify_free_slots(
    *,
    traces: tuple[TaskTrace, ...],
    shape_keys: tuple[ShapeKey, ...],
    gpu_count: int,
    at: float,
    free_by_shape: Counter[ShapeKey],
) -> Counter[str]:
    pending = _pending_tasks(traces, at)
    free_slots = sum(free_by_shape.values())
    if not pending:
        return Counter({"unavoidable_tail": free_slots})
    pending_shapes = _pending_shapes(pending, shape_keys)
    compatible_slots = sum(
        count
        for key, count in free_by_shape.items()
        if pending_shapes[key] > 0
    )
    classified: Counter[str] = Counter()
    incompatible_slots = free_slots - compatible_slots
    if incompatible_slots:
        classified["incompatible_pending"] = incompatible_slots
    if compatible_slots == 0:
        return classified
    if len(_busy_workers(traces, at)) < gpu_count:
        compatible_pending = [
            trace
            for trace in pending
            if any(shape_keys[index] in free_by_shape for index in trace.system_indices)
        ]
        next_compatible = min(
            compatible_pending,
            key=lambda trace: (trace.dispatch, trace.task_index),
        )
        ready = next_compatible.input_ready_on_dispatch is True
        category = (
            "idle_gpu_with_ready_work"
            if ready
            else "materialization_or_dispatch_wait"
        )
        classified[category] += compatible_slots
        return classified
    classified["refill_opportunity"] += compatible_slots
    return classified


def analyze_opportunity(
    payload: dict[str, Any],
    shape_keys: tuple[ShapeKey, ...],
) -> dict[str, Any]:
    """Return time-resolved passive opportunity without simulating refill."""

    traces = _task_traces(payload)
    scheduling = payload["scheduling"]
    gpu_count = int(scheduling["active_gpu_count"])
    if len(shape_keys) != int(payload["pool_size"]):
        raise ValueError("planning profile and result pool sizes differ")
    run_seconds = float(scheduling["production_run_seconds"])

    slot_seconds: Counter[str] = Counter()
    segment_count: Counter[str] = Counter()
    resident_slot_seconds = 0.0

    def accumulate_slot_interval(
        start: float,
        stop: float,
        free_by_shape: Counter[ShapeKey],
    ) -> None:
        boundaries = sorted(
            {
                start,
                stop,
                *(
                    boundary
                    for trace in traces
                    for boundary in (trace.dispatch, trace.completion)
                    if start < boundary < stop
                ),
            }
        )
        for left, right in zip(boundaries, boundaries[1:], strict=False):
            midpoint = 0.5 * (left + right)
            classified = _classify_free_slots(
                traces=traces,
                shape_keys=shape_keys,
                gpu_count=gpu_count,
                at=midpoint,
                free_by_shape=free_by_shape,
            )
            for category, slots in classified.items():
                slot_seconds[category] += slots * (right - left)
                segment_count[category] += 1

    for trace in traces:
        capacity_by_shape = Counter(shape_keys[index] for index in trace.system_indices)
        capacity = sum(capacity_by_shape.values())
        resident_slot_seconds += capacity * max(0.0, trace.profile_end - trace.dispatch)
        free_by_shape: Counter[ShapeKey] = Counter()
        previous = trace.dispatch
        slot_events = sorted(
            (
                event
                for event in trace.events
                if event.get("name") == "active_compaction_slots"
            ),
            key=lambda event: float(event["elapsed_seconds"]),
        )
        for event in slot_events:
            boundary = min(
                trace.profile_end,
                trace.dispatch + float(event["elapsed_seconds"]),
            )
            if boundary > previous and free_by_shape:
                accumulate_slot_interval(previous, boundary, free_by_shape)
            key = (
                int(event["atom_count"]),
                int(event["generalized_dimension"]),
            )
            free_by_shape[key] += int(event["slots"])
            previous = boundary
        if trace.profile_end > previous and free_by_shape:
            accumulate_slot_interval(previous, trace.profile_end, free_by_shape)

    boundaries = sorted(
        {
            0.0,
            run_seconds,
            *(trace.dispatch for trace in traces),
            *(trace.completion for trace in traces),
        }
    )
    gpu_seconds: Counter[str] = Counter()
    for start, stop in zip(boundaries, boundaries[1:], strict=False):
        if stop <= start:
            continue
        midpoint = 0.5 * (start + stop)
        idle = gpu_count - len(_busy_workers(traces, midpoint))
        if idle <= 0:
            continue
        pending = _pending_tasks(traces, midpoint)
        if not pending:
            category = "unavoidable_tail"
        elif min(
            pending,
            key=lambda trace: (trace.dispatch, trace.task_index),
        ).input_ready_on_dispatch is True:
            category = "idle_with_ready_work"
        else:
            category = "materialization_or_dispatch_wait"
        gpu_seconds[category] += idle * (stop - start)

    empty_slot_seconds = sum(slot_seconds.values())
    result = {
        "schema_version": 1,
        "status": "passive_observation_only",
        "workload_id": payload.get("workload_id"),
        "pool_size": int(payload["pool_size"]),
        "gpu_count": gpu_count,
        "task_count": len(traces),
        "production_run_seconds": run_seconds,
        "resident_slot_seconds": resident_slot_seconds,
        "empty_slot_seconds": empty_slot_seconds,
        "empty_slot_fraction": (
            0.0 if resident_slot_seconds == 0.0 else empty_slot_seconds / resident_slot_seconds
        ),
        "slot_seconds": dict(sorted(slot_seconds.items())),
        "slot_segment_count": dict(sorted(segment_count.items())),
        "gpu_seconds": dict(sorted(gpu_seconds.items())),
        "interpretation": {
            "refill_opportunity": (
                "compatible empty resident slots while every GPU was busy and "
                "unstarted work remained"
            ),
            "unavoidable_tail": (
                "empty slots or idle GPUs after every structure had been dispatched"
            ),
            "not_a_speedup_prediction": True,
        },
    }
    for mapping in (slot_seconds, gpu_seconds):
        if any(not math.isfinite(value) or value < 0.0 for value in mapping.values()):
            raise RuntimeError("opportunity accounting produced invalid time")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--planning-profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze_opportunity(
        json.loads(Path(args.result).read_text()),
        load_shape_keys(args.planning_profile),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
