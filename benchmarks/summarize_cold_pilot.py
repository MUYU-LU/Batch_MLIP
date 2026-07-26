#!/usr/bin/env python3
"""Summarize matched C256 and C32 persistent-executor sessions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from summarize_persistent_executor import _numerical_comparison


def _timing(record: dict[str, Any]) -> dict[str, Any]:
    calls = [float(item["wall_time_s"]) for item in record["call_records"]]
    return {
        "calls_s": calls,
        "first_call_s": calls[0],
        "post_first_mean_s": math.fsum(calls[1:]) / len(calls[1:]),
        "last_call_s": calls[-1],
        "session_s": math.fsum(calls),
        "peak_reserved_GiB": max(
            item["peak_reserved_bytes"] for item in record["call_records"]
        )
        / 2**30,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pair(
    label: str,
    c256_path: Path,
    c32_path: Path,
) -> dict[str, Any]:
    c256 = _load(c256_path)
    c32 = _load(c32_path)
    identity_fields = (
        "mlip",
        "workload_id",
        "jobs",
        "atoms",
        "resident_batch_size",
        "gpu_count",
        "max_steps",
    )
    mismatches = [
        field
        for field in identity_fields
        if c256.get(field) != c32.get(field)
    ]
    if mismatches:
        raise ValueError(f"{label}: mismatched fields: {', '.join(mismatches)}")
    if len(c256["call_records"]) != len(c32["call_records"]):
        raise ValueError(f"{label}: call counts differ")

    c256_timing = _timing(c256)
    c32_timing = _timing(c32)
    speedup = {
        key: c256_timing[key] / c32_timing[key]
        for key in (
            "first_call_s",
            "post_first_mean_s",
            "last_call_s",
            "session_s",
        )
    }
    return {
        "label": label,
        "identity": {field: c256.get(field) for field in identity_fields},
        "c256": c256_timing,
        "c32": c32_timing,
        "speedup_c256_over_c32": speedup,
        "numerical": _numerical_comparison(c256, c32),
        "source": {"c256": str(c256_path), "c32": str(c32_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "C256_JSON", "C32_JSON"),
        required=True,
    )
    parser.add_argument(
        "--shape-warmup-pair",
        nargs=2,
        metavar=("NO_WARMUP_JSON", "WARMUP_JSON"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "distributed-cold-pilot",
        "pairs": [
            _pair(label, Path(c256), Path(c32))
            for label, c256, c32 in args.pair
        ],
    }
    if args.shape_warmup_pair is not None:
        no_warmup = _timing(_load(Path(args.shape_warmup_pair[0])))
        warmup = _timing(_load(Path(args.shape_warmup_pair[1])))
        output["rejected_shape_warmup"] = {
            "no_warmup": no_warmup,
            "warmup": warmup,
            "session_speedup_no_warmup_over_warmup": (
                warmup["session_s"] / no_warmup["session_s"]
            ),
            "source": {
                "no_warmup": args.shape_warmup_pair[0],
                "warmup": args.shape_warmup_pair[1],
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
