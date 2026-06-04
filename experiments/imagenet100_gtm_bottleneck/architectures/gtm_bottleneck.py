"""GTMBottleneck adapter for vision feature compression.

Wraps GammaThetaMultiplexer with linear input/output adapters for
high-dimensional vision features (ResNet50 = 2048-dim, ResNet18 = 512-dim).
Uses Gumbel-softmax hard quantization for end-to-end gradient flow
through gtm.constellation. Same pattern as N8 T13 GTMBridge but with
classification head instead of reconstruction.

Plan: HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md (N10-A smoke).
Pre-reg: docs/milestones/n10a-imagenet100-gtm-bottleneck-2026-05-11.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer


class GTMBottleneck(nn.Module):
    """High-dim GTM bottleneck for vision classification.

    Forward: features [B, feat_dim] -> code [B, code_dim] -> logits [B, n_classes]

    encode: features -> per-symbol logits over alphabet -> Gumbel-hard
            indices -> gtm.forward -> waveform pool -> code
    classify: code -> linear -> logits
    """

    def __init__(
        self,
        feat_dim: int = 2048,  # ResNet50 default ; 512 for ResNet18
        code_dim: int = 64,
        n_classes: int = 100,  # ImageNet-100 ; 10 for CIFAR-10 smoke
        n_symbols: int = 4,
        *,
        seed: int | None = None,
        cfg: GammaThetaConfig | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.feat_dim = feat_dim
        self.code_dim = code_dim
        self.n_classes = n_classes
        self.n_symbols = n_symbols
        self.cfg = cfg if cfg is not None else GammaThetaConfig()
        self.gtm = GammaThetaMultiplexer(cfg=self.cfg, seed=seed)
        self.alphabet_size = self.cfg.alphabet_size

        self.encode_logits = nn.Linear(feat_dim, n_symbols * self.alphabet_size)
        self.waveform_pool_dim = 64
        # Concat pooled waveform + flat soft-constellation for ST gradient
        self.encode_pool = nn.Linear(
            self.waveform_pool_dim + n_symbols * 2, code_dim
        )
        self.classify = nn.Linear(code_dim, n_classes)

    def _gumbel_path(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        soft = F.gumbel_softmax(logits, tau=1.0, hard=True)
        idx = soft.argmax(dim=-1)
        return idx, soft @ self.gtm.constellation

    def _waveform_to_pool(self, waveform: Tensor) -> Tensor:
        if waveform.dim() == 3:
            waveform = waveform.flatten(1)
        return F.adaptive_avg_pool1d(
            waveform.unsqueeze(1), self.waveform_pool_dim
        ).squeeze(1)

    def encode(self, features: Tensor) -> Tensor:
        b = features.size(0)
        logits = self.encode_logits(features).view(
            b, self.n_symbols, self.alphabet_size
        )
        idx, soft_const = self._gumbel_path(logits)
        with torch.set_grad_enabled(True):
            waveform = self.gtm(idx)
        pooled = self._waveform_to_pool(waveform)
        soft_const_flat = soft_const.flatten(1)
        return self.encode_pool(torch.cat([pooled, soft_const_flat], dim=-1))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encode(features)
        logits = self.classify(code)
        return logits, code  # return both for downstream MI estimation
