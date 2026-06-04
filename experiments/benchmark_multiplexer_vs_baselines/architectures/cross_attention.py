"""CrossAttentionBridge — single-head cross-attention bridge.

Encoder produces a code via Q×K cross-attention with a learned
prototype set ; decoder uses the symmetric reverse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CrossAttentionBridge(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        code_dim: int = 16,
        n_prototypes: int = 8,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.dim = dim
        self.code_dim = code_dim
        self.encode_q = nn.Linear(dim, code_dim)
        self.decode_q = nn.Linear(code_dim, dim)
        self.prototypes_enc = nn.Parameter(torch.randn(n_prototypes, code_dim))
        self.prototypes_dec = nn.Parameter(torch.randn(n_prototypes, dim))

    def _attend(self, q: Tensor, kv: Tensor) -> Tensor:
        scores = q @ kv.t() / (q.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ kv

    def encode(self, x: Tensor) -> Tensor:
        q = self.encode_q(x)
        return self._attend(q, self.prototypes_enc)

    def decode(self, code: Tensor) -> Tensor:
        q = self.decode_q(code)
        return self._attend(q, self.prototypes_dec)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
