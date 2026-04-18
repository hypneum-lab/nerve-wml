"""Concrete Nerve with γ/θ oscillators and top-K sparse routing.

See spec §4.2, §3 (architecture). v0 is a functional stub: it honours the
Nerve protocol but does not yet phase-gate delivery (that's an explicit
follow-up task). This keeps unit tests deterministic and the foundation
robust; phase-gated delivery appears in Task 14 when we wire pilot P3.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role

from .oscillators import PhaseOscillator
from .router import SparseRouter


class SimNerve:
    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def __init__(
        self,
        n_wmls:      int,
        k:           int,
        *,
        seed:        int | None = 0,
        strict_n3:   bool = True,
    ) -> None:
        # Local generator — does NOT pollute global torch RNG (unlike the original
        # implementation which called torch.manual_seed(0) inside __init__).
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        self.n_wmls     = n_wmls
        self.router     = SparseRouter(n_wmls=n_wmls, k=k)
        with torch.no_grad():
            self.router.logits.data = torch.randn(
                n_wmls, n_wmls, generator=gen
            ) * 0.1
        # Pass the same generator so topology sampling is deterministic
        # regardless of global torch RNG state.
        self._edges: Tensor = self.router.sample_edges(
            tau=0.5, hard=True, generator=gen,
        )
        self.gamma_osc  = PhaseOscillator(self.GAMMA_HZ)
        self.theta_osc  = PhaseOscillator(self.THETA_HZ)
        # Initialize θ at phase 0.5 to provide temporal separation from γ.
        # γ is active in [0, 0.5), θ is active in [0, 0.5), but we offset θ to start
        # in its inactive phase [0.5, 1), creating windows where one or the other is active.
        self.theta_osc.phase = 0.5
        self._strict_n3 = strict_n3
        self._queues: dict[int, list[Neuroletter]] = defaultdict(list)
        self._clock     = 0.0

    def send(self, letter: Neuroletter) -> None:
        assert_n3_role_phase_consistent(letter, strict=self._strict_n3)
        # Enforce sparse routing — drop if edge is not active.
        if self._edges[letter.src, letter.dst].item() == 0:
            return
        self._queues[letter.dst].append(letter)

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]:
        """Phase-gated delivery with γ priority.

        γ and θ oscillators are independent, so their active windows can
        overlap. To preserve the multiplexing invariant, γ messages have
        priority: when both oscillators are active in the same tick, only
        γ messages are delivered and θ messages are held for a later tick.
        """
        pending = self._queues.get(wml_id, [])

        gamma_on = self.gamma_osc.is_active()
        theta_on = self.theta_osc.is_active()

        def is_deliverable(p: Phase) -> bool:
            if p is Phase.GAMMA:
                return gamma_on
            # θ messages only deliver when γ is NOT active (priority rule).
            return theta_on and not gamma_on

        delivered = [letter for letter in pending if is_deliverable(letter.phase)]
        held      = [letter for letter in pending if not is_deliverable(letter.phase)]
        self._queues[wml_id] = held

        if role is not None:
            delivered = [letter for letter in delivered if letter.role is role]
        if phase is not None:
            delivered = [letter for letter in delivered if letter.phase is phase]

        return delivered

    def time(self) -> float:
        return self._clock

    def tick(self, dt: float) -> None:
        self._clock += dt
        self.gamma_osc.tick(dt)
        self.theta_osc.tick(dt)

    def routing_weight(self, src: int, dst: int) -> float:
        return float(self._edges[src, dst].item())

    # Helper for Track-P debugging and tests.
    def parameters(self) -> Iterable[Tensor]:
        yield self.router.logits
