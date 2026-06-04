"""GTM ablation variants for Q1 Task 14.

Three subclasses of GammaThetaMultiplexer that selectively disable
one mechanism each:

- GTMGammaOnly: theta envelope frozen at 1.0 (no theta gating).
  Implementation: override `forward` to reuse parent logic but
  replace the `theta_env` term with `ones_like(t)` before the final
  multiplication. The theta envelope is computed inline in the parent
  (no helper to monkey-patch), so we re-implement forward minus the
  theta term. Demodulation path is unaffected (this benchmark only
  exercises forward via the GTMBridge wire-through).

- GTMThetaOnly: constellation reset to canonical PSK (no perturbation)
  and frozen with `requires_grad_(False)`. Theta gating + plasticity
  schedule remain active, but constellation can no longer learn.

- GTMNoPlasticity: plasticity schedule callable returning constant 1.0
  passed at construction time. Equivalent (per multiplexer.py L171) to
  no hook at all, but explicit for the ablation matrix.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 14.
Pre-reg: docs/milestones/q1-multiplexer-benchmark-2026-05-10.md.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from track_p.multiplexer import (
    GammaThetaConfig,
    GammaThetaMultiplexer,
    NoiseModel,
)


class GTMGammaOnly(GammaThetaMultiplexer):
    """GTM with theta envelope frozen at 1.0 (gamma carries everything).

    Re-implements `forward` from the parent, dropping the
    `theta_env.unsqueeze(0)` multiplication on the carrier. Identical
    otherwise (Gaussian per-symbol windows, gamma quadratures, PSK
    normalization, noise hook).
    """

    def forward(  # type: ignore[override]
        self,
        codes: Tensor,
        *,
        theta_phase_offset: float = 0.0,  # noqa: ARG002 — kept for API parity
        noise: NoiseModel | None = None,
        role: Tensor | None = None,
    ) -> Tensor:
        if codes.shape[-1] > self.cfg.symbols_per_theta:
            raise ValueError(
                f"K={codes.shape[-1]} exceeds symbols_per_theta="
                f"{self.cfg.symbols_per_theta} (Lisman-Idiart capacity bound)"
            )
        if role is not None:
            raise NotImplementedError(
                "out-of-band role channel pending — bouba_sens v1.2 scope"
            )
        k_active = codes.shape[-1]
        t = self._t_grid
        n_samples = t.numel()
        k_cap = self.cfg.symbols_per_theta
        window_n = n_samples // k_cap

        raw_sym = self.constellation[codes]
        sym = raw_sym / raw_sym.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        two_pi_gamma_t = 2.0 * torch.pi * self.cfg.gamma_hz * t
        gamma_i = torch.cos(two_pi_gamma_t)
        gamma_q = torch.sin(two_pi_gamma_t)

        # ABLATION: theta envelope replaced by 1.0 throughout.
        # No theta_env computation at all.

        window_centers = (
            (torch.arange(k_cap, device=t.device, dtype=t.dtype) + 0.5)
            * window_n
        )
        sigma = window_n / 4.0
        idx = torch.arange(n_samples, device=t.device, dtype=t.dtype)
        gaussian_masks = torch.exp(
            -((idx.unsqueeze(0) - window_centers.unsqueeze(1)) ** 2)
            / (2.0 * sigma**2)
        )[:k_active]

        i_t = (sym[..., 0:1] * gaussian_masks.unsqueeze(0)).sum(dim=1)
        q_t = (sym[..., 1:2] * gaussian_masks.unsqueeze(0)).sum(dim=1)

        # No theta_env multiplication.
        carrier = i_t * gamma_i.unsqueeze(0) + q_t * gamma_q.unsqueeze(0)
        carrier = carrier.to(torch.float32)

        if noise is not None:
            carrier = noise.apply(carrier)

        return carrier


class GTMThetaOnly(GammaThetaMultiplexer):
    """GTM with constellation frozen at canonical PSK (no learning)."""

    def __init__(
        self,
        cfg: GammaThetaConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(cfg=cfg, seed=seed)
        with torch.no_grad():
            angles = (
                2.0
                * math.pi
                * torch.arange(self.cfg.alphabet_size, dtype=torch.float32)
                / self.cfg.alphabet_size
            )
            psk = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
            self.constellation.copy_(psk)
        self.constellation.requires_grad_(False)


class GTMNoPlasticity(GammaThetaMultiplexer):
    """GTM with plasticity schedule pinned at constant 1.0."""

    def __init__(
        self,
        cfg: GammaThetaConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            cfg=cfg,
            seed=seed,
            plasticity_schedule=lambda _step: 1.0,
        )
