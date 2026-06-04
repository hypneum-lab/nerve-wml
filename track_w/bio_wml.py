"""BioWML — a WML whose core is a remote biological neural culture.

Conforms to nerve_core.protocols.WML. Like the three in-silico
substrates (MlpWML, LifWML, TransformerWML) it is an nn.Module with
a learned `codebook` (nn.Parameter) and a learned `emit_head_pi`
readout. Unlike them, the "core" computation is delegated to a
BioCultureClient: inbound codes are stimulated onto the culture and
the read-back spiking activity is decoded back to codes and pooled
into a hidden vector that the emit heads consume.

The client is injected. In tests / CI it is a MockBioCultureClient.
In production a real env-gated adapter is built via
BioWML.from_env(). See track_w/bio_clients.py and Task 0 of
docs/superpowers/plans/2026-05-19-bio-substrate-wml.md.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve
from track_w.bio_clients import BioCultureClient


class BioWML(nn.Module):
    """WML with a biological-culture core + π/ε emission heads."""

    def __init__(
        self,
        id: int,
        client: BioCultureClient,
        d_hidden: int = 16,
        alphabet_size: int = 64,
        threshold_eps: float = 0.30,
        input_dim: int | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.id = id
        self.client = client
        self.d_hidden = d_hidden
        self.alphabet_size = alphabet_size
        self.threshold_eps = threshold_eps
        self.input_dim = input_dim if input_dim is not None else d_hidden

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Local codebook (N-5 — each WML owns its vocabulary).
        init = torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
        self.codebook = nn.Parameter(init)
        # Shadow the parameter on the instance __dict__ so that
        # inspect.getattr_static (used by typing.runtime_checkable
        # in Python 3.12+) can see the attribute without going
        # through nn.Module.__getattr__. Keeps isinstance(wml, WML)
        # working with the nn.Parameter codebook.
        self.__dict__["codebook"] = self._parameters["codebook"]

        # Save global RNG so module construction never mutates it.
        saved_rng = torch.get_rng_state()
        try:
            self.input_proj = nn.Linear(self.input_dim, d_hidden)
            with torch.no_grad():
                self.input_proj.weight.data = torch.randn(
                    d_hidden, self.input_dim, generator=gen
                ) * 0.1
                self.input_proj.bias.data.zero_()

            # Decoded spike-activity is pooled into d_hidden; the
            # emit heads map that hidden vector to alphabet logits.
            self.emit_head_pi = nn.Linear(d_hidden, alphabet_size)
            self.emit_head_eps = nn.Linear(d_hidden, alphabet_size)
            for head in (self.emit_head_pi, self.emit_head_eps):
                with torch.no_grad():
                    head.weight.data = torch.randn(
                        alphabet_size, d_hidden, generator=gen
                    ) * 0.1
                    head.bias.data.zero_()
        finally:
            torch.set_rng_state(saved_rng)

        self.last_latency_ms: float | None = None

    @classmethod
    def from_env(cls, id: int, **kwargs: Any) -> BioWML:
        """Build a BioWML backed by a real env-gated adapter.

        Reads NERVE_WML_BIO_PROVIDER (default "finalspark") and
        constructs the matching adapter, which itself reads
        NERVE_WML_BIO_API_KEY and raises BioApiKeyMissing if unset.
        Callers that want offline behaviour must inject a
        MockBioCultureClient directly instead of calling this.
        """
        import os

        from track_w.bio_clients import (  # type: ignore[attr-defined]
            CL1Adapter,
            FinalSparkAdapter,
        )

        provider = os.environ.get(
            "NERVE_WML_BIO_PROVIDER", "finalspark"
        ).lower()
        client: BioCultureClient
        if provider == "cl1":
            client = CL1Adapter()
        elif provider == "finalspark":
            client = FinalSparkAdapter()
        else:
            raise ValueError(f"unknown bio provider: {provider!r}")
        return cls(id=id, client=client, **kwargs)

    def step(self, nerve: Nerve, t: float) -> None:
        """One tick: listen, stimulate the culture with inbound
        codes, decode the read-back activity into a hidden vector,
        emit π predictions (γ) and — on surprise — ε errors (θ).
        """
        from nerve_core.neuroletter import Neuroletter, Phase, Role

        inbound = nerve.listen(self.id)
        # Codes to stimulate. With no inbound, probe with a single
        # rest code so the culture/decoder still produce a hidden
        # state (the substrate is never fully silent at the API).
        codes = [n.code for n in inbound] if inbound else [0]

        # Round-trip through the biological culture (or its mock).
        frame = self.client.roundtrip(codes)
        self.last_latency_ms = frame.latency_ms
        decoded = self.client.decode_activity(frame)  # list[int]

        # Pool decoded codes into a d_hidden vector by averaging the
        # corresponding codebook rows — symmetric to embed_inbound.
        if decoded:
            rows = self.codebook[
                torch.tensor([c % self.alphabet_size for c in decoded])
            ]
            h = rows.mean(dim=0)
        else:
            h = torch.zeros(self.d_hidden)

        pi_logits = self.emit_head_pi(h)
        code_pi = int(pi_logits.argmax().item())

        for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
            if dst == self.id:
                continue
            if nerve.routing_weight(self.id, dst) == 1.0:
                nerve.send(Neuroletter(
                    code=code_pi, role=Role.PREDICTION,
                    phase=Phase.GAMMA, src=self.id, dst=dst,
                    timestamp=t,
                ))

        # ε path. Surprise = L2 norm of the hidden state itself:
        # a culture that produced strong evoked activity (large h)
        # signals a mismatch against the resting prior (h ≈ 0).
        surprise = float(h.norm().item())
        if surprise > self.threshold_eps:
            eps_logits = self.emit_head_eps(h)
            code_eps = int(eps_logits.argmax().item())
            for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
                if dst == self.id:
                    continue
                if nerve.routing_weight(self.id, dst) == 1.0:
                    nerve.send(Neuroletter(
                        code=code_eps, role=Role.ERROR,
                        phase=Phase.THETA, src=self.id, dst=dst,
                        timestamp=t,
                    ))

    def parameters(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> Iterable[Tensor]:
        return super().parameters(*args, **kwargs)
