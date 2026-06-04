"""GTMBridge — real wire-through of GammaThetaMultiplexer.

Replaces the Task 12 linear-only stand-in. The bridge maps
[B, dim] continuous tensors through the discrete GTM channel:

    encode(x):
      1. Linear x -> logits[B, K, A]
      2. Gumbel-softmax hard one-hot (straight-through gradient)
      3. argmax -> indices[B, K] long for gtm.forward
      4. carrier[B, T] = gtm(indices)        — real PAC waveform
      5. soft_iq[B, K, 2] = one_hot @ constellation
         (parallel ST path so gradient reaches encode_logits and
          constellation through the soft side; the waveform path
          carries gradient through the constellation as well)
      6. concat(pool(carrier), flatten(soft_iq)) -> linear -> code

    decode(code): symmetric.

This keeps the GTM channel on the critical path (constellation
parameter is in the autograd graph, plasticity hook still active)
while preserving end-to-end differentiability through the discrete
quantization step. Q1 benchmark per
HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 13.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer

# Sentinel: distinguish "user did not pass anything" (use cosine
# decay default) from "user explicitly passed None" (no schedule).
_USE_DEFAULT_SCHEDULE = object()


def default_cosine_plasticity_schedule(
    total_steps: int = 800,
    floor: float = 0.1,
) -> Callable[[int], float]:
    """Cosine decay 1.0 -> ``floor`` over ``total_steps``, then clamped.

    Used as the GTMBridge default so that the GTMNoPlasticity ablation
    (which pins the schedule to constant 1.0) is meaningfully different
    from the baseline GTM. Without this, both have a no-op plasticity
    hook and the ablation is bit-identical to baseline.
    """
    span = max(int(total_steps), 1)
    amplitude = 1.0 - floor

    def _schedule(step: int) -> float:
        s = min(max(int(step), 0), span)
        return floor + amplitude * 0.5 * (1.0 + math.cos(math.pi * s / span))

    return _schedule


class GTMBridge(nn.Module):
    """GTM with discrete quantization + straight-through gradient."""

    def __init__(
        self,
        dim: int = 16,
        code_dim: int = 16,
        *,
        seed: int | None = None,
        cfg: GammaThetaConfig | None = None,
        gumbel_tau: float = 1.0,
        plasticity_schedule: Callable[[int], float] | None = _USE_DEFAULT_SCHEDULE,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.dim = dim
        self.code_dim = code_dim
        self.gumbel_tau = gumbel_tau
        self.cfg = cfg if cfg is not None else GammaThetaConfig()
        if plasticity_schedule is _USE_DEFAULT_SCHEDULE:
            plasticity_schedule = default_cosine_plasticity_schedule()
        self.gtm = GammaThetaMultiplexer(
            cfg=self.cfg,
            seed=seed,
            plasticity_schedule=plasticity_schedule,
        )
        self.alphabet_size = self.cfg.alphabet_size            # 64
        self.k_symbols = self.cfg.symbols_per_theta            # 7

        # encode side: x -> per-symbol logits over alphabet
        self.encode_logits = nn.Linear(
            dim, self.k_symbols * self.alphabet_size
        )
        # waveform pooled to a fixed length, then mixed with soft IQ
        self.waveform_pool_dim = 64
        self.encode_mix = nn.Linear(
            self.waveform_pool_dim + self.k_symbols * 2, code_dim
        )

        # decode side: symmetric
        self.decode_logits = nn.Linear(
            code_dim, self.k_symbols * self.alphabet_size
        )
        self.decode_mix = nn.Linear(
            self.waveform_pool_dim + self.k_symbols * 2, dim
        )

    # ------------------------------------------------------------ helpers

    def _gtm_pass(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """logits[B,K,A] -> (pooled_waveform[B,P], soft_iq_flat[B,K*2]).

        Gumbel-softmax hard gives one-hot with straight-through
        gradient. The hard indices feed the real GTM forward (waveform
        path; gradient flows into constellation). The soft one-hot is
        also matmul'd against the constellation directly to push
        gradient back into encode_logits / decode_logits.
        """
        one_hot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        # one_hot: [B, K, A] — argmax-equal forward, soft backward.
        indices = one_hot.argmax(dim=-1)            # [B, K] long
        carrier = self.gtm(indices)                 # [B, T]
        pooled = F.adaptive_avg_pool1d(
            carrier.unsqueeze(1), self.waveform_pool_dim
        ).squeeze(1)                                # [B, P]
        soft_iq = one_hot @ self.gtm.constellation  # [B, K, 2]
        soft_iq_flat = soft_iq.flatten(1)           # [B, K*2]
        return pooled, soft_iq_flat

    # ------------------------------------------------------------ public

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
