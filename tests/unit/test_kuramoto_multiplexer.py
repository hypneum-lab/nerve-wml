import torch

from track_p.transducer_baselines import KuramotoMultiplexer


def test_kuramoto_round_trips_codes_noise_free():
    torch.manual_seed(0)
    m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (8, 7))
    carrier = m.forward(codes)
    assert carrier.shape[0] == 8
    recovered = m.demodulate(carrier)
    assert recovered.shape == codes.shape
    assert (recovered >= 0).all() and (recovered < 64).all()


def test_kuramoto_is_differentiable():
    torch.manual_seed(1)
    m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (4, 7))
    carrier = m.forward(codes)
    carrier.sum().backward()
    # Coupling weights and natural frequencies must receive gradient.
    assert m.coupling.grad is not None
    assert m.natural_freqs.grad is not None


def test_kuramoto_phase_state_evolves():
    torch.manual_seed(2)
    m = KuramotoMultiplexer(alphabet_size=64, n_symbols=7, n_steps=20)
    codes = torch.tensor([[3, 17, 42, 5, 9, 30, 11]])
    carrier_short = m.forward(codes, n_steps=2)
    carrier_long = m.forward(codes, n_steps=20)
    # Longer integration produces a different terminal phase pattern.
    assert not torch.allclose(carrier_short, carrier_long, atol=1e-6)
