"""Process-wide reproducibility controls for benchmarks and worker processes."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

_SEED_ENV = "BATCH_MLIP_REPRODUCIBILITY_SEED"
_DETERMINISTIC_ENV = "BATCH_MLIP_DETERMINISTIC_ALGORITHMS"
_WARN_ONLY_ENV = "BATCH_MLIP_DETERMINISTIC_WARN_ONLY"
_CUBLAS_DEFAULT = ":4096:8"
_ACTIVE_STATE: dict[str, Any] | None = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Complete process-level deterministic execution request."""

    seed: int = 20260729
    deterministic_algorithms: bool = True
    deterministic_warn_only: bool = False
    cublas_workspace_config: str = _CUBLAS_DEFAULT
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    allow_tf32: bool = False
    cpu_threads: int | None = 1
    interop_threads: int | None = 1

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.cublas_workspace_config not in (":16:8", ":4096:8"):
            raise ValueError(
                "cublas_workspace_config must be ':16:8' or ':4096:8'"
            )
        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive or None")
        if self.interop_threads is not None and self.interop_threads <= 0:
            raise ValueError("interop_threads must be positive or None")


def reproducibility_environment(
    config: ReproducibilityConfig,
) -> dict[str, str]:
    """Return environment values required before interpreter/CUDA startup."""

    environment = {
        "PYTHONHASHSEED": str(config.seed),
        "CUBLAS_WORKSPACE_CONFIG": config.cublas_workspace_config,
        _SEED_ENV: str(config.seed),
        _DETERMINISTIC_ENV: str(int(config.deterministic_algorithms)),
        _WARN_ONLY_ENV: str(int(config.deterministic_warn_only)),
    }
    if config.cpu_threads is not None:
        threads = str(config.cpu_threads)
        environment.update(
            {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
            }
        )
    return environment


def _set_threads(config: ReproducibilityConfig) -> None:
    if (
        config.cpu_threads is not None
        and torch.get_num_threads() != config.cpu_threads
    ):
        torch.set_num_threads(config.cpu_threads)
    if (
        config.interop_threads is not None
        and torch.get_num_interop_threads() != config.interop_threads
    ):
        try:
            torch.set_num_interop_threads(config.interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "PyTorch interop threads were initialized before the "
                "reproducibility contract was installed"
            ) from error


def configure_reproducibility(
    config: ReproducibilityConfig | None = None,
    *,
    require_preconfigured_python_hash: bool = False,
) -> dict[str, Any]:
    """Install and report deterministic controls in the current process.

    ``PYTHONHASHSEED`` only controls the current interpreter when it was present
    before startup. This function still installs it for subsequently spawned
    workers and can reject a late installation for strict benchmark launchers.
    """

    resolved = config or ReproducibilityConfig()
    requested_environment = reproducibility_environment(resolved)
    hash_seed_preconfigured = (
        os.environ.get("PYTHONHASHSEED") == str(resolved.seed)
    )
    if require_preconfigured_python_hash and not hash_seed_preconfigured:
        raise RuntimeError(
            "PYTHONHASHSEED must equal the requested seed before Python starts"
        )
    current_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if (
        torch.cuda.is_initialized()
        and current_cublas != resolved.cublas_workspace_config
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be installed before CUDA initializes"
        )
    os.environ.update(requested_environment)

    random.seed(resolved.seed)
    np.random.seed(resolved.seed % (2**32))
    torch.manual_seed(resolved.seed)
    torch.cuda.manual_seed_all(resolved.seed)
    torch.use_deterministic_algorithms(
        resolved.deterministic_algorithms,
        warn_only=resolved.deterministic_warn_only,
    )
    torch.backends.cudnn.benchmark = resolved.cudnn_benchmark
    torch.backends.cudnn.deterministic = resolved.cudnn_deterministic
    torch.backends.cudnn.allow_tf32 = resolved.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = resolved.allow_tf32
    if not resolved.allow_tf32:
        torch.set_float32_matmul_precision("highest")
    _set_threads(resolved)

    state = {
        "config": asdict(resolved),
        "python_hash_seed_preconfigured": hash_seed_preconfigured,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_random_seeded": True,
        "numpy_global_rng_seeded": True,
        "torch_cpu_rng_seeded": True,
        "torch_cuda_rngs_seeded": True,
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
    }
    global _ACTIVE_STATE
    _ACTIVE_STATE = deepcopy(state)
    return state


def configure_reproducibility_from_environment() -> dict[str, Any] | None:
    """Install an inherited contract in a newly spawned worker, when present."""

    seed = os.environ.get(_SEED_ENV)
    if seed is None:
        return None
    cpu_threads = os.environ.get("OMP_NUM_THREADS")
    return configure_reproducibility(
        ReproducibilityConfig(
            seed=int(seed),
            deterministic_algorithms=_env_bool(
                _DETERMINISTIC_ENV,
                True,
            ),
            deterministic_warn_only=_env_bool(_WARN_ONLY_ENV, False),
            cublas_workspace_config=os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG",
                _CUBLAS_DEFAULT,
            ),
            cpu_threads=(
                None if cpu_threads is None else int(cpu_threads)
            ),
            interop_threads=1,
        ),
        require_preconfigured_python_hash=True,
    )


def active_reproducibility_state() -> dict[str, Any] | None:
    """Return the installed process contract without changing random states."""

    return deepcopy(_ACTIVE_STATE)
