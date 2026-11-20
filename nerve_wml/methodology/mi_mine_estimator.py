"""Mutual Information Neural Estimation (Belghazi et al. 2018).

Trains a small critic network T(x, y) to maximise the
Donsker-Varadhan lower bound on mutual information:

    I(X; Y) >= E_P[T(x, y)] - log E_Q[exp(T(x, y'))]

where P is the joint distribution of (X, Y) and Q is the product
of marginals (obtained by shuffling y relative to x).

Complements the Kraskov KSG k-NN estimator as a fourth independent
MI measurement for cross-estimator robustness checks. Both operate
on continuous pre-VQ embeddings in R^d, but via distinct principles
(neighbour density vs neural variational bound).

Implementation uses the raw DV bound with tail-average of the last
50 epochs (MINE training oscillates around the true value; the
max is biased upward, the tail mean is more robust). No EMA / JSD
variants to keep the code auditable; for paper submission we report
the result and flag the known DV variance property.

Typical hyperparameters for d=16, N=2000:
  hidden=128, n_epochs=500, batch_size=256, lr=1e-3
  => ~15 seconds on commodity CPU, ~5 seconds on GPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Critic(nn.Module):
    """Small MLP critic T(x, y) = net(concat([x, y]))."""

    def __init__(self, d_x: int, d_y: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x + d_y, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1)).squeeze(-1)


def mi_mine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden:       int   = 128,
    n_epochs:     int   = 500,
    batch_size:   int   = 256,
    lr:           float = 1e-3,
    seed:         int   = 0,
    tail_average: int   = 50,
    device:       str   = "cpu",
) -> float:
    """MINE-DV estimator of MI(X; Y), returned in nats.

    Args:
        x:            (N, d_x) continuous array.
        y:            (N, d_y) continuous array, same N.
        hidden:       critic MLP hidden dim (default 128).
        n_epochs:     number of gradient steps (default 500).
        batch_size:   minibatch size (default 256).
        lr:           Adam learning rate (default 1e-3).
        seed:         RNG + critic init seed.
        tail_average: how many trailing epochs to average the
                      returned value over (default 50).
        device:       "cpu" or "cuda:N".

    Returns:
        MI estimate in nats. Clipped to >= 0 (DV bound can go
        slightly negative for near-independent data at finite sample).
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x.shape[0]={x.shape[0]} != y.shape[0]={y.shape[0]}"
        )
    n = x.shape[0]
    if n < batch_size:
        raise ValueError(
            f"need at least {batch_size} samples for MINE, got {n}"
        )
    if n_epochs < tail_average:
        raise ValueError(
            f"n_epochs ({n_epochs}) must exceed tail_average ({tail_average})"
        )

    torch.manual_seed(seed)
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    critic = _Critic(x_t.shape[1], y_t.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)

    history = np.empty(n_epochs, dtype=np.float64)
    for epoch in range(n_epochs):
        idx_joint = torch.randint(0, n, (batch_size,), device=device)
        idx_marg = torch.randint(0, n, (batch_size,), device=device)
        x_b = x_t[idx_joint]
        y_joint = y_t[idx_joint]
        y_marg = y_t[idx_marg]

        t_joint = critic(x_b, y_joint)
        t_marg = critic(x_b, y_marg)

        mi_bound = t_joint.mean() - (
            torch.logsumexp(t_marg, dim=0) - float(np.log(batch_size))
        )
        loss = -mi_bound

        opt.zero_grad()
        loss.backward()
        opt.step()

        history[epoch] = float(mi_bound.detach().item())

    tail = history[-tail_average:]
    # np.mean returns a numpy scalar; cast explicitly for mypy.
    tail_mean: float = float(np.mean(tail))
    return float(max(tail_mean, 0.0))
