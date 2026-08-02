"""Optimizer-independent active-refill scheduling helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..core.state import AseGraphBatch

REFILL_POLICIES = frozenset(("drain", "immediate", "threshold"))
REFILL_STORAGE_MODES = frozenset(("repack", "slots"))


@dataclass(frozen=True)
class CompatibleSlotAssignment:
    """Deterministic pending jobs matched to equal-atom resident slots."""

    destination_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    unmatched_destination_ids: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Return whether every destination has a compatible source."""

        return not self.unmatched_destination_ids


def match_compatible_slots(
    destination_ids: Sequence[int],
    destination_counts: Sequence[int],
    pending_ids: Sequence[int],
    pending_counts: Sequence[int],
) -> CompatibleSlotAssignment:
    """Match each slot to the earliest unused pending job of equal atom count.

    The function does not mutate the pending queue. Callers must accept only a
    complete assignment or use their existing repack fallback; partial fixed
    batches would otherwise keep completed graphs resident.
    """

    destinations = tuple(int(value) for value in destination_ids)
    destination_sizes = tuple(int(value) for value in destination_counts)
    pending = tuple(int(value) for value in pending_ids)
    pending_sizes = tuple(int(value) for value in pending_counts)
    if len(destinations) != len(destination_sizes):
        raise ValueError("destination IDs and counts must have equal length")
    if len(pending) != len(pending_sizes):
        raise ValueError("pending IDs and counts must have equal length")
    if len(set(destinations)) != len(destinations):
        raise ValueError("destination IDs must be unique")
    if len(set(pending)) != len(pending):
        raise ValueError("pending IDs must be unique")
    if any(value <= 0 for value in (*destination_sizes, *pending_sizes)):
        raise ValueError("atom counts must be positive")

    used: set[int] = set()
    matched_destinations = []
    matched_sources = []
    unmatched = []
    for destination, destination_count in zip(
        destinations,
        destination_sizes,
        strict=True,
    ):
        source_offset = next(
            (
                offset
                for offset, source_count in enumerate(pending_sizes)
                if offset not in used and source_count == destination_count
            ),
            None,
        )
        if source_offset is None:
            unmatched.append(destination)
            continue
        used.add(source_offset)
        matched_destinations.append(destination)
        matched_sources.append(pending[source_offset])
    return CompatibleSlotAssignment(
        destination_ids=tuple(matched_destinations),
        source_ids=tuple(matched_sources),
        unmatched_destination_ids=tuple(unmatched),
    )


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
