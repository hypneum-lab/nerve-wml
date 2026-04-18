"""γ (40 Hz) and θ (6 Hz) phase oscillators for the SimNerve.

See spec §7.5 (rhythmic multiplexing) and §3 (architecture).
"""
from __future__ import annotations


class PhaseOscillator:
    """A unit-period phase clock. phase in [0, 1). Active in the first half
    of each cycle (phase < 0.5), inactive otherwise. This lets SimNerve deliver
    neuroletters only during their role's phase window."""

    def __init__(self, freq_hz: float) -> None:
        assert freq_hz > 0
        self.freq_hz = freq_hz
        self.phase   = 0.0

    @property
    def period_s(self) -> float:
        return 1.0 / self.freq_hz

    def tick(self, dt: float) -> None:
        self.phase = (self.phase + dt / self.period_s) % 1.0

    def is_active(self) -> bool:
        return self.phase < 0.5
