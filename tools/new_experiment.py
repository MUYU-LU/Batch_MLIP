"""Create a new experiment directory from the repository template."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    destination = root / "experiments" / args.experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    payload = yaml.safe_load((root / "experiments" / "TEMPLATE.yaml").read_text())
    payload["id"] = args.experiment_id
    (destination / "experiment.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        f"# {args.experiment_id}\n\nRecord the hypothesis, change, results, and conclusion here.\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
