"""Opt-in runtime profiling for batch calculations and optimizers."""

from .runtime import RuntimeProfiler, profile_event, profile_phase
from .telemetry import (
    RUN_TELEMETRY_FIELDS,
    RunTelemetry,
    append_run_telemetry_csv,
    runtime_profile_registry_fields,
    write_run_telemetry_json,
)

__all__ = [
    "RUN_TELEMETRY_FIELDS",
    "RunTelemetry",
    "RuntimeProfiler",
    "append_run_telemetry_csv",
    "profile_event",
    "profile_phase",
    "runtime_profile_registry_fields",
    "write_run_telemetry_json",
]
