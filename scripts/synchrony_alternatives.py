"""Candidate replacements for `_synchrony_index` in GTM ablation analysis.

The current top-PC variance fraction is non-monotone across arms (null > gtm
> simple_gating > akorn-min). This module evaluates 4 alternative metrics
that each summarise the spectrum of a carrier batch differently:

    spectral_entropy  -- Shannon entropy of normalised eigenvalues.
                          Range [0, log D]. Higher = more spread.
    participation_ratio -- (sum sigma_i^2)^2 / sum sigma_i^4. Effective
                          dimension. Range (0, D].
    effective_rank    -- exp(spectral_entropy / log e). Robust to scale.
                          Range [1, D].
    top3_dispersion   -- 1 - (sum top 3 eigvals) / (sum all eigvals).
                          Range [0, 1]. Higher = more spread.

All take a `[B, D]` carrier tensor, return a single float per carrier.
"""
from __future__ import annotations

import torch


def _normalised_eigvals(carrier: torch.Tensor) -> torch.Tensor:
    centred = carrier - carrier.mean(dim=0, keepdim=True)
    if centred.shape[0] < 2:
        return torch.tensor([1.0], device=carrier.device)
    s = torch.linalg.svdvals(centred)
    eigs = s.pow(2)
    total = eigs.sum()
    if total <= 1e-12:
        return torch.full_like(eigs, 1.0 / eigs.numel())
    return eigs / total


def spectral_entropy(carrier: torch.Tensor) -> float:
    """Shannon entropy of normalised eigenvalues, in nats."""
    p = _normalised_eigvals(carrier)
    p = p.clamp_min(1e-12)
    return float((-(p * p.log()).sum()).item())


def participation_ratio(carrier: torch.Tensor) -> float:
    """(sum p_i)^2 / sum p_i^2 in normalised form = 1 / sum p_i^2."""
    p = _normalised_eigvals(carrier)
    return float((1.0 / (p * p).sum()).item())


def effective_rank(carrier: torch.Tensor) -> float:
    """exp(spectral entropy in nats)."""
    return float(torch.tensor(spectral_entropy(carrier)).exp().item())


def top3_dispersion(carrier: torch.Tensor) -> float:
    p = _normalised_eigvals(carrier)
    p_sorted, _ = p.sort(descending=True)
    return float((1.0 - p_sorted[: min(3, p_sorted.numel())].sum()).item())
