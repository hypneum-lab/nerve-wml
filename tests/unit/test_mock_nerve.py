from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_w.mock_nerve import MockNerve


def _letter(src: int, dst: int, role: Role, phase: Phase, t: float = 0.0) -> Neuroletter:
    return Neuroletter(code=5, role=role, phase=phase, src=src, dst=dst, timestamp=t)


def test_mock_nerve_round_trip():
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    # MockNerve starts with γ=True, θ=False by default, so γ messages deliver.
    nerve.send(_letter(0, 1, Role.PREDICTION, Phase.GAMMA))
    received = nerve.listen(wml_id=1)
    assert len(received) == 1
    assert received[0].code == 5


def test_mock_nerve_seed_is_local():
    """Constructing a MockNerve must NOT mutate the global torch RNG."""
    import torch
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = MockNerve(n_wmls=4, k=2, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed


def test_mock_nerve_routing_weight_count():
    nerve = MockNerve(n_wmls=4, k=2, seed=0)
    active = sum(
        1
        for i in range(4)
        for j in range(4)
        if nerve.routing_weight(i, j) == 1.0
    )
    assert active == 4 * 2
