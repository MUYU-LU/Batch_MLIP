"""Compatibility utilities for the uploaded AtomBit model namespace.

This module intentionally lives at ``src.utils`` because existing pickled
checkpoints may reference that import path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch

DEFAULT_FLOAT_DTYPE = torch.float32
DEFAULT_NP_FLOAT_DTYPE = np.float32

PathKey = Tuple[int, int, int, str]


def default_active_paths() -> Dict[PathKey, bool]:
    """Return a broad set of Cartesian tensor-product paths."""

    return {
        (0, 0, 0, "prod"): True,
        (0, 1, 1, "prod"): True,
        (1, 0, 1, "prod"): True,
        (0, 2, 2, "prod"): True,
        (2, 0, 2, "prod"): True,
        (1, 1, 0, "dot"): True,
        (1, 1, 1, "cross"): True,
        (1, 1, 2, "outer"): True,
        (2, 1, 1, "mat_vec"): True,
        (1, 2, 1, "vec_mat"): True,
        (2, 2, 0, "double_dot"): True,
        (2, 2, 2, "mat_mul_sym"): True,
        (1, 2, 2, "vec_cross_tensor"): True,
        (2, 1, 2, "tensor_cross_vector"): True,
        (2, 2, 1, "tensor_commutator"): True,
    }


@dataclass
class AtomBitConfig:
    """Configuration fields required by the uploaded model implementation.

    Extra attributes from older serialized instances remain supported because
    this dataclass does not use slots.
    """

    cutoff: float = 6.0
    num_rbf: int = 8
    hidden_dim: int = 128
    num_layers: int = 4
    num_atom_types: int = 100
    atom_types_map: list[int] = field(default_factory=list)
    use_L1: bool = True
    use_L2: bool = True
    use_gating: bool = True
    use_direct_force: bool = False
    active_paths: Dict[PathKey, bool] = field(default_factory=default_active_paths)
    block_impls: Dict[str, str] = field(default_factory=dict)
    outer_impl: int = 1
    mat_mul_sym_impl: int = 1
    gating_impl: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomBitConfig":
        data = dict(payload)
        paths = data.get("active_paths")
        if isinstance(paths, list):
            parsed: Dict[PathKey, bool] = {}
            for item in paths:
                if isinstance(item, Mapping):
                    key = (
                        int(item["l_in"]),
                        int(item["l_edge"]),
                        int(item["l_out"]),
                        str(item["op"]),
                    )
                    parsed[key] = bool(item.get("active", True))
                else:
                    l_in, l_edge, l_out, op = item
                    parsed[(int(l_in), int(l_edge), int(l_out), str(op))] = True
            data["active_paths"] = parsed
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["active_paths"] = [
            {
                "l_in": key[0],
                "l_edge": key[1],
                "l_out": key[2],
                "op": key[3],
                "active": active,
            }
            for key, active in self.active_paths.items()
        ]
        return data


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Small ``torch_scatter.scatter_add``-compatible implementation."""

    if index.dtype != torch.long:
        index = index.long()
    if dim < 0:
        dim += src.ndim
    if not 0 <= dim < src.ndim:
        raise ValueError(f"invalid dim={dim} for source with ndim={src.ndim}")

    if dim != 0:
        src = src.movedim(dim, 0)
    if index.ndim != 1 or index.shape[0] != src.shape[0]:
        raise ValueError("index must be one-dimensional and match src along scatter dim")

    if dim_size is None:
        dim_size = 0 if index.numel() == 0 else int(index.max().item()) + 1
    out = torch.zeros(
        (int(dim_size),) + tuple(src.shape[1:]),
        dtype=src.dtype,
        device=src.device,
    )
    if src.numel() > 0:
        out.index_add_(0, index, src)
    if dim != 0:
        out = out.movedim(0, dim)
    return out
