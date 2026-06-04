"""EWC — Elastic Weight Consolidation (diagonal Fisher approximation).

API
---
estimate_fisher(wml, data_loader) -> dict[str, Tensor]
    Compute diagonal Fisher information for each named parameter of `wml`
    from the squared gradients of the cross-entropy log-likelihood.

penalty(wml, fisher, theta_star, lam) -> Tensor
    Return the scalar EWC penalty  lam/2 * Σ_i F_i (θ_i − θ*_i)².

Invariants
----------
W-1: No mutation of another WML's parameters — this module only reads
     and computes gradients on the wml passed in.
W-2: penalty ranges over wml.parameters() which includes the codebook,
     so the codebook is regularised alongside the MLP weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from track_w.mlp_wml import MlpWML


def estimate_fisher(
    wml: MlpWML,
    data_loader: list[tuple[Tensor, Tensor]],
) -> dict[str, Tensor]:
    """Diagonal Fisher information from squared log-likelihood gradients.

    Parameters
    ----------
    wml : MlpWML
        The model after Task 0 training (weights = θ*).
    data_loader : list of (x, y) pairs
        Batches from Task 0. Labels must be in [0, n_classes).

    Returns
    -------
    dict mapping parameter name → non-negative diagonal Fisher tensor
        (same shape as the parameter).
    """
    # Infer n_classes from all batches to avoid IndexError when the first batch
    # does not contain the highest class id (can happen at arbitrary seeds).
    n_classes = max(y.max().item() for _, y in data_loader) + 1
    fisher: dict[str, Tensor] = {
        name: torch.zeros_like(p) for name, p in wml.named_parameters()
    }
    wml.eval()
    n_batches = len(data_loader)
    for x, y in data_loader:
        logits = wml.emit_head_pi(wml.core(x))[:, : int(n_classes)]
        log_probs = F.log_softmax(logits, dim=-1)
        # Use true label log-likelihood for a tighter diagonal estimate.
        # NOTE: nll_loss(reduction='mean') gives a batch-averaged gradient, so this is
        # the squared mean, not the mean of squares — the standard empirical-Fisher (EWC)
        # convention. lam absorbs the scale factor.
        nll = F.nll_loss(log_probs, y)
        wml.zero_grad()
        nll.backward()
        for name, p in wml.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    for name in fisher:
        fisher[name] /= n_batches
    wml.zero_grad()  # clean residual grads from the last backward pass
    wml.train()
    return fisher


def penalty(
    wml: MlpWML,
    fisher: dict[str, Tensor],
    theta_star: dict[str, Tensor],
    lam: float,
) -> Tensor:
    """EWC quadratic penalty: lam/2 * Σ_i F_i (θ_i − θ*_i)².

    Parameters
    ----------
    wml : MlpWML
        Current model (θ, being trained on Task 1).
    fisher : dict[str, Tensor]
        Diagonal Fisher from estimate_fisher() — same keys as named_parameters.
    theta_star : dict[str, Tensor]
        Snapshot of wml parameters right after Task 0 (before Task 1 training).
    lam : float
        Regularisation strength (sweep and record in result JSON).

    Returns
    -------
    Scalar tensor, differentiable w.r.t. wml.parameters().
    """
    pen = torch.tensor(0.0)
    for name, p in wml.named_parameters():
        if name in fisher and name in theta_star:
            diff = p - theta_star[name]
            pen = pen + (fisher[name] * diff ** 2).sum()
    return lam / 2.0 * pen
