"""MockNerve — in-memory Nerve for Track-W.

Mirrors SimNerve's API (same Protocol) but skips γ/θ oscillators: phase gating
is driven by a simulation tick that the caller advances explicitly. Uses a
LOCAL torch.Generator so constructing a MockNerve does not mutate the global
RNG state (avoids the Plan 1 SimNerve seed footgun, now fixed).

γ priority: when both phases are active, γ delivers and θ is held. This
mirrors SimNerve.listen() so Plan 3 merge is a drop-in swap.

See spec §4.2 (Nerve) and §3 (two-tracks architecture).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.router import SparseRouter


class MockNerve:
    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def __init__(
        self,
        n_wmls:    int,
        k:         int,
        *,
        seed:      int | None = None,
        strict_n3: bool       = True,
    ) -> None:
        # Local generator — save/restore global RNG so we don't mutate it.
        old_state = torch.get_rng_state()
        try:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)

            self.n_wmls  = n_wmls
            self.router  = SparseRouter(n_wmls=n_wmls, k=k)
            # Override router init with generator so topology is deterministic.
            with torch.no_grad():
                self.router.logits.data = torch.randn(
                    n_wmls, n_wmls, generator=gen
                ) * 0.1

            self._edges: Tensor = self.router.sample_edges(
                tau=0.5, hard=True, generator=gen,
            )
        finally:
            torch.set_rng_state(old_state)

        self._strict_n3     = strict_n3
        self._queues: dict[int, list[Neuroletter]] = defaultdict(list)
        self._clock         = 0.0
        # Track W drives phase gates externally via set_phase_active.
        # Default: γ active, θ inactive — matches typical initial training.
        self._gamma_active  = True
        self._theta_active  = False

    def send(self, letter: Neuroletter) -> None:
        assert_n3_role_phase_consistent(letter, strict=self._strict_n3)
        if self._edges[letter.src, letter.dst].item() == 0:
            return
        self._queues[letter.dst].append(letter)

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]:
        pending = self._queues.get(wml_id, [])
        delivered: list[Neuroletter] = []
        held:      list[Neuroletter] = []

        for letter in pending:
            if letter.phase is Phase.GAMMA and self._gamma_active:
                delivered.append(letter)
            elif letter.phase is Phase.THETA and self._theta_active and not self._gamma_active:
                # γ priority: θ only when γ inactive (matches SimNerve).
                delivered.append(letter)
            else:
                held.append(letter)

        self._queues[wml_id] = held

        if role is not None:
            delivered = [l for l in delivered if l.role is role]
        if phase is not None:
            delivered = [l for l in delivered if l.phase is phase]
        return delivered

    def time(self) -> float:
        return self._clock

    def tick(self, dt: float) -> None:
        """Advance simulation clock. Track-W drives phase gates directly
        via set_phase_active rather than oscillator math — simpler, and
        the merge to SimNerve in Plan 3 preserves semantics."""
        self._clock += dt

    def set_phase_active(self, gamma: bool, theta: bool) -> None:
        """Test/pilot helper: directly set which phases are active.
        γ priority is still enforced in listen()."""
        self._gamma_active = gamma
        self._theta_active = theta

    def routing_weight(self, src: int, dst: int) -> float:
        return float(self._edges[src, dst].item())

    def parameters(self) -> Iterable[Tensor]:
        yield self.router.logits
