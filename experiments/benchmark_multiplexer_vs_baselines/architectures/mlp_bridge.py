"""MLPBridge — vanilla 2-layer MLP baseline.

Same hidden_dim and code_dim as RecursiveLink for parity ; differs
only in the absence of cosine-alignment hook. Uses ReLU instead of
Tanh to avoid trivial duplication.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class MLPBridge(nn.Module):
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
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.decode_layer = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encode_layer(x)

    def decode(self, code: Tensor) -> Tensor:
        return self.decode_layer(code)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
