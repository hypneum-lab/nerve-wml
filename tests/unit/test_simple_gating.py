import torch

from track_p.transducer_baselines import SimpleGatingMultiplexer


def test_simple_gating_round_trips_codes_noise_free():
    torch.manual_seed(0)
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (8, 7))
    carrier = m.forward(codes)
    assert carrier.shape[0] == 8
    recovered = m.demodulate(carrier)
    # Untrained module need not be accurate, but shapes must round-trip.
    assert recovered.shape == codes.shape
    assert (recovered >= 0).all() and (recovered < 64).all()


def test_simple_gating_is_differentiable():
    torch.manual_seed(1)
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (4, 7))
    carrier = m.forward(codes)
    carrier.sum().backward()
    assert m.gate.weight.grad is not None
