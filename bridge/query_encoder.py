"""QueryEncoder — tokens → fixed embedding → VQ-quantized neuroletter code.

Real deployment would wrap a MiniLM sentence-transformer. For nerve-wml's
own CI we use a small learnable linear projection with a fixed codebook
so the encoder has no external deps. Downstream (micro-kiki) can replace
the projection with a real encoder while keeping the same API.

Plan 4d Task 3.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QueryEncoder(nn.Module):
    """Maps a [B, token_dim] tensor → [B] long tensor of neuroletter codes.

    Pipeline:
      1. Linear projection to hidden dim.
      2. Nearest-codebook lookup (hard argmax) against a fixed codebook.

    The codebook is set at construction (typically copied from a trained
    WML) — it is NOT trained by this module.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        codebook: Tensor,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        assert codebook.shape[1] == hidden_dim, (
            f"codebook dim {codebook.shape[1]} != hidden_dim {hidden_dim}"
        )

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Save global RNG, init projection from local Generator, restore.
        saved = torch.get_rng_state()
        try:
            self.projection = nn.Linear(token_dim, hidden_dim)
            with torch.no_grad():
                self.projection.weight.data = torch.randn(
                    hidden_dim, token_dim, generator=gen,
                ) * 0.1
                self.projection.bias.data.zero_()
        finally:
            torch.set_rng_state(saved)

        # Codebook is a registered buffer — not trainable, but moves with the module.
        self.register_buffer("codebook", codebook.detach().clone())

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [B, token_dim] float → codes: [B] long in [0, alphabet_size)."""
        h = self.projection(tokens)                      # [B, hidden_dim]
        dist = torch.cdist(h, self.codebook)  # type: ignore[arg-type]
        return dist.argmin(dim=-1)
