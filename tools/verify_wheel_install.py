#!/usr/bin/env python3
"""Install a wheel into an isolated target and verify the released public API."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    if not args.wheel.is_file() or args.wheel.suffix != ".whl":
        parser.error("wheel must be an existing .whl file")
    return args


def main() -> None:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="batch-mlip-wheel-") as temporary:
        root = Path(temporary)
        target = root / "site-packages"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--target",
                str(target),
                str(args.wheel.resolve()),
            ],
            check=True,
            cwd=root,
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(target)
        environment.pop("PYTHONHOME", None)
        code = """
import json
from pathlib import Path

import batch_mlip
from batch_mlip import MACEBatchCalculator, optimize_pool
from batch_mlip.planning import load_packaged_hardware_capacity_policies

policies = load_packaged_hardware_capacity_policies()
package = Path(batch_mlip.__file__).resolve()
assert callable(optimize_pool)
assert callable(MACEBatchCalculator.from_off)
assert len(policies) == 2
assert any(policy.contract["model_id"] == "MACE-OFF23-Small" for policy in policies)
print(json.dumps({
    "package": str(package),
    "policy_ids": [policy.policy_id for policy in policies],
    "status": "pass",
}, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        if not Path(result["package"]).is_relative_to(target.resolve()):
            raise RuntimeError("verification imported the source tree, not the wheel")
        example = target / "examples" / "optimize_pool_mace.py"
        if not example.is_file():
            raise RuntimeError("wheel does not contain the canonical MACE example")
        result["example"] = str(example)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
