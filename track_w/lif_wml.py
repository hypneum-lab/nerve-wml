"""LifWML — a WML whose core is a population of LIF neurons.

Dynamics: v_mem ← v_mem + (dt / tau) · (−v_mem + i_in), then spike = (v_mem > v_thr),
with reset. A pattern-match decoder compares spikes to the local codebook to
emit a code when confidence exceeds a threshold; otherwise the WML stays silent
(N-1).

step() and emission are implemented in Tasks 10 and 11.
"""
from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from nerve_core.protocols import Nerve


class LifWML(nn.Module):
    def __init__(
        self,
        id:            int,
        n_neurons:     int   = 100,
        alphabet_size: int   = 64,
        v_thr:         float = 1.0,
        tau_mem:       float = 20e-3,
        threshold_eps: float = 0.30,
        *,
        seed:          int | None = None,
    ) -> None:
        super().__init__()
        self.id            = id
        self.n_neurons     = n_neurons
        self.alphabet_size = alphabet_size
        self.v_thr         = v_thr
        self.tau_mem       = tau_mem
        self.threshold_eps = threshold_eps

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Codebook: each code is a spike pattern (binary-like target).
        init = (torch.rand(alphabet_size, n_neurons, generator=gen) > 0.7).float()
        self.codebook = nn.Parameter(init)

        # Membrane state — re-init per episode via `.reset_state()`.
        self.register_buffer("v_mem", torch.zeros(n_neurons))

        # Save global RNG, re-init input_proj from local generator, restore.
        saved_rng = torch.get_rng_state()
        try:
            self.input_proj = nn.Linear(n_neurons, n_neurons)
            with torch.no_grad():
                self.input_proj.weight.data = torch.randn(
                    n_neurons, n_neurons, generator=gen
                ) * 0.1
                self.input_proj.bias.data.zero_()
        finally:
            torch.set_rng_state(saved_rng)

    def reset_state(self) -> None:
        self.v_mem.zero_()

    def step(self, nerve: Nerve, t: float) -> None:  # pragma: no cover
        raise NotImplementedError("Task 10 defines LifWML.step()")

    def parameters(self, *args, **kwargs) -> Iterable[Tensor]:  # type: ignore[override]
        return super().parameters(*args, **kwargs)
