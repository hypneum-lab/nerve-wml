"""RecursiveLinkBridge — port of Yang et al. 2026 RecursiveMAS link.

A 2-layer linear projection with cosine-aligned bottleneck.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


class RecursiveLinkBridge(nn.Module):
    """2-layer linear bridge with optional cosine alignment loss hook."""

    def __init__(
        self,
        dim: int = 16,
        code_dim: int = 16,
        hidden_dim: int = 64,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.encode_layer = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.decode_layer = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encode_layer(x)

    def decode(self, code: Tensor) -> Tensor:
        return self.decode_layer(code)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def cosine_alignment(self, code_a: Tensor, code_b: Tensor) -> Tensor:
        """Cosine alignment loss — Yang 2026 §3.2 RecursiveMAS link loss."""
        return 1.0 - F.cosine_similarity(code_a, code_b, dim=-1).mean()
