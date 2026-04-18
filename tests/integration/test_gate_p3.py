import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from scripts.track_p_pilot import run_p3
from track_p.sim_nerve import SimNerve


def test_p3_no_phase_collisions():
    """γ letters and θ letters never share a delivery timestamp."""
    torch.manual_seed(0)
    collision_count = run_p3(n_cycles=200)
    assert collision_count == 0


def test_p3_theta_eventually_delivers():
    """Strengthen Gate P3: θ priority rule must not starve θ forever.

    Without this test, `run_p3` could pass with 0 collisions even if θ
    were always held (which would break the spec's multiplexing claim).
    """
    nerve = SimNerve(n_wmls=4, k=2)
    dt = 1e-3
    theta_delivered = 0

    for cycle in range(500):
        nerve.send(Neuroletter(3, Role.PREDICTION, Phase.GAMMA, 0, 1, nerve.time()))
        nerve.send(Neuroletter(7, Role.ERROR,      Phase.THETA, 2, 1, nerve.time()))
        nerve.tick(dt)
        for letter in nerve.listen(wml_id=1):
            if letter.phase is Phase.THETA:
                theta_delivered += 1

    assert theta_delivered > 0, (
        "θ was starved over 500 cycles — γ priority rule is too strict "
        "or θ-gate window never opens. Breaks spec multiplexing claim."
    )
