from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from batch_mlip import ReproducibilityConfig, reproducibility_environment

ROOT = Path(__file__).resolve().parents[1]

_CHILD_SCRIPT = """
import json
import random
import numpy as np
import torch
from batch_mlip import (
    active_reproducibility_state,
    configure_reproducibility_from_environment,
)

state = configure_reproducibility_from_environment()
print(json.dumps({
    "hash": hash("batch-mlip"),
    "python": random.random(),
    "numpy": float(np.random.random()),
    "torch": float(torch.rand(())),
    "state": state,
    "active_state": active_reproducibility_state(),
}, sort_keys=True))
"""


def _run_seed(seed: int) -> dict:
    config = ReproducibilityConfig(seed=seed)
    environment = {
        **os.environ,
        **reproducibility_environment(config),
        "PYTHONPATH": str(ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def test_reproducibility_contract_repeats_all_global_rngs_and_hashes():
    first = _run_seed(20260729)
    second = _run_seed(20260729)

    assert first == second
    state = first["state"]
    assert first["active_state"] == state
    assert state["python_hash_seed_preconfigured"]
    assert state["torch_deterministic_algorithms"]
    assert state["cublas_workspace_config"] == ":4096:8"
    assert not state["cudnn_benchmark"]
    assert state["cudnn_deterministic"]
    assert not state["cudnn_allow_tf32"]
    assert not state["cuda_matmul_allow_tf32"]
    assert state["cpu_threads"] == 1
    assert state["interop_threads"] == 1


def test_reproducibility_contract_distinguishes_different_seeds():
    first = _run_seed(20260729)
    second = _run_seed(20260730)

    assert first["python"] != second["python"]
    assert first["numpy"] != second["numpy"]
    assert first["torch"] != second["torch"]
    assert first["hash"] != second["hash"]


def test_reproducibility_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        ReproducibilityConfig(seed=-1)
    with pytest.raises(ValueError, match="cublas"):
        ReproducibilityConfig(cublas_workspace_config="invalid")
    with pytest.raises(ValueError, match="cpu_threads"):
        ReproducibilityConfig(cpu_threads=0)
