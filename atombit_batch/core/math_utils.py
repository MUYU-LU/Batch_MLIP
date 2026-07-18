"""Core scatter and shape helpers without torch-scatter."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out_shape = (size,) + tuple(values.shape[1:])
    out = torch.zeros(out_shape, device=values.device, dtype=values.dtype)
    if values.numel() != 0:
        out.index_add_(0, index, values)
    return out


def scatter_max(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.full((size,), -torch.inf, device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return torch.zeros_like(out)
    if hasattr(out, "scatter_reduce_"):
        out.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    else:  # pragma: no cover - for old PyTorch
        for graph_idx in range(size):
            mask = index == graph_idx
            if bool(mask.any()):
                out[graph_idx] = values[mask].max()
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def system_l2_norm(vectors: torch.Tensor, system_idx: torch.Tensor, n_systems: int) -> torch.Tensor:
    squared = (vectors * vectors).sum(dim=-1)
    return torch.sqrt(scatter_sum(squared, system_idx, n_systems).clamp_min(0.0))


def as_system_parameter(
    value: float | Sequence[float] | torch.Tensor,
    *,
    n_systems: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(n_systems).clone()
    if tensor.shape == (n_systems,):
        return tensor.clone()
    raise ValueError(
        f"{name} must be a scalar or have shape ({n_systems},), "
        f"got {tuple(tensor.shape)}"
    )


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.get_default_dtype()
