import math

from track_p.oscillators import PhaseOscillator


def test_gamma_period_is_25ms():
    osc = PhaseOscillator(freq_hz=40.0)
    assert math.isclose(osc.period_s, 0.025, abs_tol=1e-6)


def test_phase_advances_with_tick():
    osc = PhaseOscillator(freq_hz=40.0)
    assert osc.phase == 0.0
    osc.tick(dt=0.0125)   # half period
    assert math.isclose(osc.phase, 0.5, abs_tol=1e-6)
    osc.tick(dt=0.0125)   # full period → wraps back to 0
    assert math.isclose(osc.phase, 0.0, abs_tol=1e-6)


def test_is_active_window():
    """A PhaseOscillator fires in the first half of each cycle."""
    osc = PhaseOscillator(freq_hz=40.0)
    assert osc.is_active()
    osc.tick(dt=0.020)  # into second half (phase > 0.5)
    assert not osc.is_active()
