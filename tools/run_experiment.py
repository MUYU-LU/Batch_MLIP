"""Execute a reproducible experiment specification and write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(spec_path: Path, root: Path) -> Path:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise TypeError("experiment specification must be a mapping")
    experiment_id = str(spec.get("id", spec_path.stem))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "runs" / "experiments" / f"{experiment_id}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "experiment.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )

    tracked_files = [
        path
        for top in ("atombit_batch", "src", "tests", "benchmarks", "configs")
        for path in (root / top).rglob("*")
        if path.is_file()
    ]
    manifest: dict[str, Any] = {
        "id": experiment_id,
        "description": spec.get("description"),
        "hypothesis": spec.get("hypothesis"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(root),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cwd": str(root),
        },
        "source_hashes": {
            str(path.relative_to(root)): sha256(path) for path in sorted(tracked_files)
        },
        "commands": [],
    }

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(spec.get("environment", {})).items()})
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    for index, entry in enumerate(spec.get("commands", [])):
        if not isinstance(entry, dict):
            raise TypeError("each command entry must be a mapping")
        name = str(entry.get("name", f"command-{index:02d}"))
        command = entry.get("command")
        argv = shlex.split(command) if isinstance(command, str) else [str(x) for x in command]
        stdout_path = run_dir / f"{index:02d}-{name}.stdout.txt"
        stderr_path = run_dir / f"{index:02d}-{name}.stderr.txt"

        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                argv,
                cwd=root,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
                text=True,
            )
        elapsed = time.perf_counter() - started
        record = {
            "name": name,
            "argv": argv,
            "returncode": completed.returncode,
            "wall_seconds": elapsed,
            "stdout": str(stdout_path.relative_to(root)),
            "stderr": str(stderr_path.relative_to(root)),
        }
        manifest["commands"].append(record)
        if completed.returncode != 0 and not bool(entry.get("allow_failure", False)):
            manifest["status"] = "failed"
            break
    else:
        manifest["status"] = "passed"

    collected: dict[str, Any] = {}
    for item in spec.get("collect_json", []):
        path = root / str(item)
        if path.exists():
            collected[str(item)] = json.loads(path.read_text(encoding="utf-8"))
    manifest["collected_json"] = collected
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    path = run(Path(args.spec), Path(args.root).resolve())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
