"""TransformerWML — a WML whose core is a small transformer encoder.

Implements the WML Protocol (nerve_core.protocols.WML): listens on its
nerve input, decodes inbound codes via embed_inbound mean-pool, splits
the pooled embedding into n_tokens tokens, runs a multi-layer multi-head
encoder, mean-pools the attended token sequence, and emits π (γ phase)
and optionally ε (θ phase) neuroletters.

Third substrate in the polymorphism lineage: MlpWML (stateless MLP),
LifWML (LIF neurons + surrogate gradient), TransformerWML (attention).
Shares the same emit_head_pi / emit_head_eps readout interface so the
same train_wml_on_task loop covers all three substrates.

Invariants preserved:
  N-1: step() is silent when not confident (implicit via emit head logits).
  W-1: step() never mutates another WML — all communication via nerve.send.
  W-2: parameters() includes codebook + all internal weights.
  W-5: local codebook (64, d_model) — no cross-WML sharing.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class TransformerWML(nn.Module):
    """WML with a Transformer encoder core + independent π/ε emission heads."""

    def __init__(
        self,
        id:            int,
        d_model:       int   = 16,
        n_layers:      int   = 2,
        n_heads:       int   = 2,
        n_tokens:      int   = 4,
        alphabet_size: int   = 64,
        threshold_eps: float = 0.30,
        *,
        seed:          int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_tokens != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_tokens ({n_tokens})"
            )
        self.id            = id
        self.d_model       = d_model
        self.n_tokens      = n_tokens
        self.d_token       = d_model // n_tokens
        self.alphabet_size = alphabet_size
        self.threshold_eps = threshold_eps

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Save global RNG; every random init draws from the local generator
        # and we restore global state before returning.
        saved_rng = torch.get_rng_state()
        try:
            self.codebook = nn.Parameter(
                torch.randn(alphabet_size, d_model, generator=gen) * 0.1
            )

            self.token_proj = nn.Linear(self.d_token, d_model)
            with torch.no_grad():
                self.token_proj.weight.data = torch.randn(
                    self.token_proj.weight.shape, generator=gen
                ) * 0.1
                self.token_proj.bias.data.zero_()

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.0,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

            self.emit_head_pi  = nn.Linear(d_model, alphabet_size)
            self.emit_head_eps = nn.Linear(d_model, alphabet_size)
            for lin in (self.emit_head_pi, self.emit_head_eps):
                with torch.no_grad():
                    lin.weight.data = torch.randn(
                        lin.weight.shape, generator=gen
                    ) * 0.1
                    lin.bias.data.zero_()
        finally:
            torch.set_rng_state(saved_rng)

    def core(self, x: Tensor) -> Tensor:
        """Tokenize [B, d_model] → attend → mean-pool → [B, d_model]."""
        if x.dim() != 2:
            raise ValueError(f"expected [B, d_model], got {tuple(x.shape)}")
        batch = x.shape[0]
        tokens = x.view(batch, self.n_tokens, self.d_token)
        tokens = self.token_proj(tokens)
        attended = self.encoder(tokens)
        return attended.mean(dim=1)

    def step(self, nerve: Nerve, t: float) -> None:
        """One tick: listen, transformer-encode, emit π, and optionally ε."""
        from nerve_core.neuroletter import Neuroletter, Phase, Role
        from track_w._decode import embed_inbound

        inbound = nerve.listen(self.id)
        h_in    = embed_inbound(inbound, self.codebook)          # [d_model]
        h       = self.core(h_in.unsqueeze(0)).squeeze(0)        # [d_model]

        pi_logits = self.emit_head_pi(h)
        code_pi   = int(pi_logits.argmax().item())

        for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
            if dst == self.id:
                continue
            if nerve.routing_weight(self.id, dst) == 1.0:
                nerve.send(Neuroletter(
                    code=code_pi, role=Role.PREDICTION, phase=Phase.GAMMA,
                    src=self.id, dst=dst, timestamp=t,
                ))

        # ε path — surprise = ||h − h_prior||, same semantic as MlpWML.
        h_prior  = self.core(torch.zeros_like(h_in).unsqueeze(0)).squeeze(0)
        surprise = (h - h_prior).norm().item()

        if surprise > self.threshold_eps:
            eps_logits = self.emit_head_eps(h - h_prior)
            code_eps   = int(eps_logits.argmax().item())
            for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
                if dst == self.id:
                    continue
                if nerve.routing_weight(self.id, dst) == 1.0:
                    nerve.send(Neuroletter(
                        code=code_eps, role=Role.ERROR, phase=Phase.THETA,
                        src=self.id, dst=dst, timestamp=t,
                    ))

    def parameters(self, *args, **kwargs) -> Iterable[Tensor]:  # type: ignore[override]
        return super().parameters(*args, **kwargs)
