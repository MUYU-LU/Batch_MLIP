"""Low-overhead phase profiling with deferred CUDA synchronization."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _PendingCudaPhase:
    name: str
    device: torch.device
    start: torch.cuda.Event
    end: torch.cuda.Event
    host_seconds: float
    metadata: dict[str, Any]


_ACTIVE_PROFILER: ContextVar[RuntimeProfiler | None] = ContextVar(
    "batch_mlip_runtime_profiler", default=None
)


def _normalized_metadata(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(f"profile metadata tensor {key!r} must be scalar")
            value = value.detach().item()
        if isinstance(value, torch.device):
            value = str(value)
        if value is None or isinstance(value, (bool, int, float, str)):
            normalized[key] = value
        else:
            raise TypeError(
                f"profile metadata {key!r} must be JSON-scalar compatible"
            )
    return normalized


class RuntimeProfiler:
    """Collect internal phase timings without changing calculator signatures.

    CUDA phases use events and are synchronized once when the context exits.
    CPU phases use ``perf_counter``. Normal execution has only a context-variable
    lookup at each instrumentation point when no profiler is active.
    """

    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = None if device is None else torch.device(device)
        self._token: Token[RuntimeProfiler | None] | None = None
        self._started_at: float | None = None
        self._total_seconds: float | None = None
        self._pending_cuda: list[_PendingCudaPhase] = []
        self._samples: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._cuda_devices: set[torch.device] = set()
        self._finalized = False

    def __enter__(self) -> RuntimeProfiler:
        if self._token is not None or self._finalized:
            raise RuntimeError("RuntimeProfiler instances cannot be reused")
        self._started_at = time.perf_counter()
        self._token = _ACTIVE_PROFILER.set(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._token is None:
            raise RuntimeError("RuntimeProfiler context was not entered")
        _ACTIVE_PROFILER.reset(self._token)
        self._token = None
        self.finalize()

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        device: str | torch.device | None = None,
        **metadata: Any,
    ):
        if self._finalized:
            raise RuntimeError("cannot add phases after profiler finalization")
        resolved_device = self.device if device is None else torch.device(device)
        details = _normalized_metadata(metadata)
        host_started = time.perf_counter()

        if resolved_device is not None and resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(resolved_device):
                start.record(torch.cuda.current_stream(resolved_device))
            try:
                yield
            finally:
                host_seconds = time.perf_counter() - host_started
                with torch.cuda.device(resolved_device):
                    end.record(torch.cuda.current_stream(resolved_device))
                self._cuda_devices.add(resolved_device)
                self._pending_cuda.append(
                    _PendingCudaPhase(
                        name=name,
                        device=resolved_device,
                        start=start,
                        end=end,
                        host_seconds=host_seconds,
                        metadata=details,
                    )
                )
            return

        try:
            yield
        finally:
            elapsed = time.perf_counter() - host_started
            self._samples.append(
                {
                    "name": name,
                    "seconds": elapsed,
                    "host_seconds": elapsed,
                    "device": "cpu" if resolved_device is None else str(resolved_device),
                    **details,
                }
            )

    def event(self, name: str, **values: Any) -> None:
        if self._finalized:
            raise RuntimeError("cannot add events after profiler finalization")
        self._events.append({"name": name, **_normalized_metadata(values)})

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._started_at is None:
            raise RuntimeError("RuntimeProfiler must be entered before finalization")

        for device in self._cuda_devices:
            torch.cuda.synchronize(device)
        for pending in self._pending_cuda:
            cuda_seconds = pending.start.elapsed_time(pending.end) / 1000.0
            self._samples.append(
                {
                    "name": pending.name,
                    "seconds": cuda_seconds,
                    "host_seconds": pending.host_seconds,
                    "device": str(pending.device),
                    **pending.metadata,
                }
            )
        self._pending_cuda.clear()
        self._total_seconds = time.perf_counter() - self._started_at
        self._finalized = True

    def summary(self, *, include_samples: bool = True) -> dict[str, Any]:
        if not self._finalized:
            raise RuntimeError("profile summary is available after the context exits")

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in self._samples:
            grouped[sample["name"]].append(sample)
        phases = {}
        for name, samples in sorted(grouped.items()):
            seconds = [float(sample["seconds"]) for sample in samples]
            host_seconds = [float(sample["host_seconds"]) for sample in samples]
            phases[name] = {
                "count": len(samples),
                "total_seconds": sum(seconds),
                "mean_seconds": sum(seconds) / len(seconds),
                "max_seconds": max(seconds),
                "total_host_seconds": sum(host_seconds),
            }

        result: dict[str, Any] = {
            "schema_version": 1,
            "total_seconds": self._total_seconds,
            "phases": phases,
            "events": list(self._events),
        }
        if include_samples:
            result["samples"] = list(self._samples)
        if self.device is not None and self.device.type == "cuda":
            result["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device)
        return result


def profile_phase(
    name: str,
    *,
    device: str | torch.device | None = None,
    **metadata: Any,
):
    """Return an active profiler phase or a no-op context manager."""

    profiler = _ACTIVE_PROFILER.get()
    if profiler is None:
        return nullcontext()
    return profiler.phase(name, device=device, **metadata)


def profile_event(name: str, **values: Any) -> None:
    """Record a scalar runtime event when profiling is active."""

    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.event(name, **values)
