"""Optimizer-independent active-refill scheduling helpers."""

from __future__ import annotations

import torch

from ..core.state import AseGraphBatch

REFILL_POLICIES = frozenset(("drain", "immediate", "threshold"))
REFILL_STORAGE_MODES = frozenset(("repack", "slots"))


def refill_insert_count(
    *,
    policy: str,
    capacity: int,
    survivors: int,
    pending: int,
    low_watermark: float,
    min_chunk: int,
) -> int:
    """Return how many pending systems to insert after active compaction."""

    slots = capacity - survivors
    if slots <= 0 or pending <= 0:
        return 0
    if policy == "immediate":
        return min(slots, pending)
    if survivors == 0:
        return min(slots, pending)
    if policy == "drain":
        return 0
    low_water_count = int(capacity * low_watermark)
    if survivors > low_water_count or slots < min_chunk:
        return 0
    return min(slots, pending)


def global_atom_ids(
    state: AseGraphBatch,
    system_ids: torch.Tensor,
) -> torch.Tensor:
    """Map ordered system IDs in ``state`` to their ordered global atom IDs."""

    blocks = [
        torch.arange(
            state.ptr[system_id],
            state.ptr[system_id + 1],
            device=state.device,
            dtype=torch.long,
        )
        for system_id in system_ids.tolist()
    ]
    if not blocks:
        return torch.empty(0, device=state.device, dtype=torch.long)
    return torch.cat(blocks)
