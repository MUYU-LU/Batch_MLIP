"""Compare two experiment manifests without hiding regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def command_map(manifest):
    return {entry["name"]: entry for entry in manifest.get("commands", [])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    bmap = command_map(baseline)
    cmap = command_map(candidate)

    print(f"baseline:  {baseline.get('id')} ({baseline.get('status')})")
    print(f"candidate: {candidate.get('id')} ({candidate.get('status')})")
    print()
    print(f"{'command':24s} {'base_s':>12s} {'cand_s':>12s} {'ratio':>10s} {'status':>10s}")
    for name in sorted(set(bmap) | set(cmap)):
        b = bmap.get(name)
        c = cmap.get(name)
        if b is None or c is None:
            print(f"{name:24s} {'-':>12s} {'-':>12s} {'-':>10s} {'missing':>10s}")
            continue
        ratio = c["wall_seconds"] / b["wall_seconds"] if b["wall_seconds"] else float("nan")
        status = "ok" if c["returncode"] == 0 else "failed"
        print(
            f"{name:24s} {b['wall_seconds']:12.4f} {c['wall_seconds']:12.4f} "
            f"{ratio:10.3f} {status:>10s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
