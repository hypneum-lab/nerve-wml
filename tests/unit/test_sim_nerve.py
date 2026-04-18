import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.sim_nerve import SimNerve


def _letter(src: int, dst: int, role: Role, phase: Phase, t: float = 0.0) -> Neuroletter:
    return Neuroletter(code=3, role=role, phase=phase, src=src, dst=dst, timestamp=t)


def test_sim_nerve_round_trip():
    nerve = SimNerve(n_wmls=4, k=2)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    # Receiver sees the message on the next listen, regardless of oscillator
    # phase (no phase gating in v0 — phase filtering is additive in a later task).
    received = nerve.listen(wml_id=1)
    assert len(received) == 1
    assert received[0].code == 3


def test_sim_nerve_filter_by_role():
    nerve = SimNerve(n_wmls=4, k=2)
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA, t=0.0))
    nerve.send(_letter(2, 1, Role.ERROR,      Phase.THETA, t=0.0))
    # γ oscillator starts at phase=0 (active), θ starts at phase=0.5 (inactive),
    # so only π is delivered and ε is held.
    delivered = nerve.listen(wml_id=1)
    assert len([l for l in delivered if l.role is Role.PREDICTION]) == 1
    assert len([l for l in delivered if l.role is Role.ERROR]) == 0
    # Tick until γ is inactive AND θ is active. θ starts at phase=0.5, so
    # θ re-enters its active window when its phase wraps back below 0.5.
    # At t=95ms : γ_phase ≈ 0.80 (inactive, γ priority released),
    # θ_phase  ≈ 0.07 (active, now deliverable).
    for _ in range(95):
        nerve.tick(1e-3)
    delivered = nerve.listen(wml_id=1)
    assert len([l for l in delivered if l.role is Role.ERROR]) == 1


def test_sim_nerve_tick_advances_time():
    nerve = SimNerve(n_wmls=4, k=2)
    t0 = nerve.time()
    nerve.tick(dt=0.010)
    assert nerve.time() > t0


def test_sim_nerve_routing_weight_edge_count():
    nerve = SimNerve(n_wmls=4, k=2)
    active_edges = sum(
        1
        for i in range(4)
        for j in range(4)
        if nerve.routing_weight(i, j) == 1.0
    )
    assert active_edges == 4 * 2  # K edges per row, 4 rows
