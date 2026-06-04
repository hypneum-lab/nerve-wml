from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_w.bio_clients import MockBioCultureClient
from track_w.bio_wml import BioWML
from track_w.mock_nerve import MockNerve


def _inject(nerve, dst, code):
    nerve._queues[dst].append(
        Neuroletter(code=code, role=Role.PREDICTION, phase=Phase.GAMMA,
                    src=1, dst=dst, timestamp=0.0)
    )


def test_step_runs_without_error_on_silent_input():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
    wml.step(nerve, t=0.0)  # no inbound — must not raise


def test_step_emits_a_prediction_after_stimulus():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
    _inject(nerve, dst=0, code=7)
    wml.step(nerve, t=0.0)
    out = nerve.listen(1)  # WML 1's inbox
    assert any(n.role is Role.PREDICTION for n in out)


def test_step_prediction_uses_gamma_phase():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
    _inject(nerve, dst=0, code=7)
    wml.step(nerve, t=0.0)
    out = [n for n in nerve.listen(1) if n.role is Role.PREDICTION]
    assert out and all(n.phase is Phase.GAMMA for n in out)


def test_step_emitted_code_is_in_alphabet():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
    _inject(nerve, dst=0, code=7)
    wml.step(nerve, t=0.0)
    assert all(0 <= n.code < 64 for n in nerve.listen(1))


def test_step_records_last_latency_ms():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
    _inject(nerve, dst=0, code=7)
    wml.step(nerve, t=0.0)
    assert wml.last_latency_ms is not None
    assert wml.last_latency_ms > 0.0
