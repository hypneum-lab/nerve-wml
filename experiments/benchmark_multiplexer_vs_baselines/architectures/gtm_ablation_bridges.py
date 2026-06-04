"""Bridge wrappers for the 3 GTM ablation variants.

Mirrors GTMBridge from gtm.py exactly (same Linear shapes, same
Gumbel-softmax hard wire-through, same waveform pool + soft IQ
concat). Only the underlying GammaThetaMultiplexer is swapped for
an ablation variant.

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 14.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from experiments.benchmark_multiplexer_vs_baselines.architectures.gtm_ablations import (
    GTMGammaOnly,
    GTMNoPlasticity,
    GTMThetaOnly,
)
from track_p.multiplexer import GammaThetaConfig


class _AblationBridge(nn.Module):
    """Shared scaffold ; subclasses set ``gtm_cls``."""

    gtm_cls: type

    def __init__(
        self,
        dim: int = 16,
        code_dim: int = 16,
        *,
        seed: int | None = None,
        cfg: GammaThetaConfig | None = None,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.dim = dim
        self.code_dim = code_dim
        self.gumbel_tau = gumbel_tau
        self.cfg = cfg if cfg is not None else GammaThetaConfig()
        self.gtm = self.gtm_cls(cfg=self.cfg, seed=seed)
        self.alphabet_size = self.cfg.alphabet_size
        self.k_symbols = self.cfg.symbols_per_theta

        self.encode_logits = nn.Linear(
            dim, self.k_symbols * self.alphabet_size
        )
        self.waveform_pool_dim = 64
        self.encode_mix = nn.Linear(
            self.waveform_pool_dim + self.k_symbols * 2, code_dim
        )

        self.decode_logits = nn.Linear(
            code_dim, self.k_symbols * self.alphabet_size
        )
        self.decode_mix = nn.Linear(
            self.waveform_pool_dim + self.k_symbols * 2, dim
        )

    def _gtm_pass(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        one_hot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        indices = one_hot.argmax(dim=-1)
        carrier = self.gtm(indices)
        pooled = F.adaptive_avg_pool1d(
            carrier.unsqueeze(1), self.waveform_pool_dim
        ).squeeze(1)
        soft_iq = one_hot @ self.gtm.constellation
        soft_iq_flat = soft_iq.flatten(1)
        return pooled, soft_iq_flat

    def encode(self, x: Tensor) -> Tensor:
        b = x.size(0)
        logits = self.encode_logits(x).view(
            b, self.k_symbols, self.alphabet_size
        )
        pooled, soft_iq = self._gtm_pass(logits)
        return self.encode_mix(torch.cat([pooled, soft_iq], dim=-1))

    def decode(self, code: Tensor) -> Tensor:
        b = code.size(0)
        logits = self.decode_logits(code).view(
            b, self.k_symbols, self.alphabet_size
        )
        pooled, soft_iq = self._gtm_pass(logits)
        return self.decode_mix(torch.cat([pooled, soft_iq], dim=-1))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class GTMGammaOnlyBridge(_AblationBridge):
    gtm_cls = GTMGammaOnly


class GTMThetaOnlyBridge(_AblationBridge):
    gtm_cls = GTMThetaOnly


class GTMNoPlasticityBridge(_AblationBridge):
    gtm_cls = GTMNoPlasticity
