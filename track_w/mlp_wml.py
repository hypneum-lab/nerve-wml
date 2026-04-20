"""MlpWML — a WML whose core is a 4-layer MLP.

Implements the WML protocol (nerve_core.protocols.WML): listens on its nerve
input, decodes inbound codes via an embed_inbound mean-pool, runs the MLP,
and emits π predictions (γ phase) and optionally ε errors (θ phase).

The step() method is defined in Task 6 (π) and Task 7 (ε).
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class MlpWML(nn.Module):
    """WML with a 4-layer MLP core + independent π/ε emission heads."""

    def __init__(
        self,
        id:            int,
        d_hidden:      int  = 128,
        alphabet_size: int  = 64,
        threshold_eps: float = 0.30,
        input_dim:     int | None = None,
        *,
        seed:          int | None = None,
    ) -> None:
        super().__init__()
        self.id            = id
        self.alphabet_size = alphabet_size
        self.threshold_eps = threshold_eps
        # v1.2: decouple task-input dim from internal width. When None,
        # fall back to d_hidden for backward compatibility with v1.1.
        self.input_dim = input_dim if input_dim is not None else d_hidden

        # Create a local generator for all random ops
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Local codebook (N-5 — each WML owns its vocabulary).
        init = torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
        self.codebook = nn.Parameter(init)

        # Save global RNG state to avoid mutation during nn.Linear creation.
        global_state = torch.get_rng_state()

        # Core: optional input projection (input_dim→d_hidden) if different,
        # then 4 d_hidden-wide layers with ReLU between.
        layers: list[nn.Module] = []
        if self.input_dim != d_hidden:
            layers.append(nn.Linear(self.input_dim, d_hidden))
            layers.append(nn.ReLU())
        for _ in range(4):
            lin = nn.Linear(d_hidden, d_hidden)
            layers.append(lin)
            if _ < 3:
                layers.append(nn.ReLU())
        # Seed every Linear from the local generator.
        for m in layers:
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.data = torch.randn(
                        m.weight.shape, generator=gen,
                    ) * 0.1
                    m.bias.data.zero_()

        self.core = nn.Sequential(*layers)

        # Init heads with local generator.
        self.emit_head_pi  = nn.Linear(d_hidden, alphabet_size)
        self.emit_head_eps = nn.Linear(d_hidden, alphabet_size)

        with torch.no_grad():
            self.emit_head_pi.weight.data = torch.randn(
                self.emit_head_pi.weight.shape, generator=gen
            ) * 0.1
            self.emit_head_pi.bias.data.zero_()
            self.emit_head_eps.weight.data = torch.randn(
                self.emit_head_eps.weight.shape, generator=gen
            ) * 0.1
            self.emit_head_eps.bias.data.zero_()

        # Restore global RNG state.
        torch.set_rng_state(global_state)

    def step(self, nerve: Nerve, t: float) -> None:
        """One tick: listen, MLP forward, emit π predictions, and
        optionally emit ε errors if surprise exceeds threshold.

        Surprise is measured as the L2 norm between the current hidden
        state and the model's prior (forward on zero input).
        """
        from nerve_core.neuroletter import Neuroletter, Phase, Role
        from track_w._decode import embed_inbound

        inbound = nerve.listen(self.id)
        h_in    = embed_inbound(inbound, self.codebook)
        h       = self.core(h_in.unsqueeze(0)).squeeze(0)

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

        # ε path. Surprise = L2 norm of (h − h_prior).
        # h_prior is an MLP-forward of a zero vector: the model's prior
        # expectation with no input.
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
