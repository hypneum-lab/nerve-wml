"""Spike encoders for neuromorphic export.

rate_encode: float input in [0, 1] → binary spike tensor by
             Bernoulli sampling at each timestep.
temporal_encode: time-to-first-spike — high values fire early,
                 low values fire late.

Both use a local torch.Generator so they never mutate the global RNG.
Plan 6 Task 2-3.
"""
from __future__ import annotations

import torch
from torch import Tensor


def rate_encode(
    x: Tensor,
    *,
    n_timesteps: int = 32,
    seed: int = 0,
) -> Tensor:
    """Rate coding: Bernoulli sampling at each of n_timesteps.

    Args:
        x:           [..., dim] float in approximately [0, 1]. Values are
                     clamped to that range.
        n_timesteps: number of spike windows to produce.
        seed:        local Generator seed.

    Returns:
        [..., n_timesteps, dim] binary spike tensor.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    p = x.clamp(0.0, 1.0)
    p_expanded = p.unsqueeze(-2).expand(*p.shape[:-1], n_timesteps, p.shape[-1])
    spikes = torch.bernoulli(p_expanded.contiguous(), generator=gen)
    return spikes


def temporal_encode(x: Tensor, *, n_timesteps: int = 32) -> Tensor:
    """Time-to-first-spike: value 1 fires at t=0, value 0 fires never.

    Args:
        x:           [..., dim] float in [0, 1].
        n_timesteps: number of timesteps.

    Returns:
        [..., n_timesteps, dim] binary tensor with at most one spike
        per (batch, dim).
    """
    p = x.clamp(0.0, 1.0)
    fire_t = ((1.0 - p) * (n_timesteps - 1)).round().long()
    out = torch.zeros(*p.shape[:-1], n_timesteps, p.shape[-1])
    for idx in torch.nonzero(p > 0.0, as_tuple=False):
        flat = idx.tolist()
        dim_idx = flat[-1]
        batch_dims = flat[:-1]
        t = int(fire_t[tuple(batch_dims + [dim_idx])].item())
        if t < n_timesteps:
            out[tuple(batch_dims + [t, dim_idx])] = 1.0
    return out
