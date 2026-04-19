"""Tests for neuromorphic.spike_encoder."""
import torch

from neuromorphic.spike_encoder import rate_encode, temporal_encode


def test_rate_encode_shape_and_binary():
    x = torch.rand(4, 16)
    spikes = rate_encode(x, n_timesteps=32, seed=0)
    assert spikes.shape == (4, 32, 16)
    # Binary tensor: only 0 and 1.
    assert ((spikes == 0) | (spikes == 1)).all()


def test_rate_encode_rate_matches_input():
    """Mean firing rate should approximate the input value."""
    x = torch.full((8, 16), 0.5)
    spikes = rate_encode(x, n_timesteps=200, seed=0)
    rate = spikes.mean()
    assert 0.4 < rate.item() < 0.6


def test_rate_encode_seed_is_local():
    x = torch.rand(4, 8)
    torch.manual_seed(42)
    expected = torch.rand(1).item()
    torch.manual_seed(42)
    _ = rate_encode(x, n_timesteps=16, seed=99)
    observed = torch.rand(1).item()
    assert expected == observed


def test_rate_encode_deterministic():
    x = torch.rand(4, 16)
    a = rate_encode(x, n_timesteps=32, seed=0)
    b = rate_encode(x, n_timesteps=32, seed=0)
    assert torch.equal(a, b)


def test_temporal_encode_shape_and_sparsity():
    x = torch.rand(4, 16)
    spikes = temporal_encode(x, n_timesteps=32)
    assert spikes.shape == (4, 32, 16)
    # At most 1 spike per (batch, dim) across timesteps.
    totals = spikes.sum(dim=-2)  # [4, 16]
    assert (totals <= 1).all()


def test_temporal_encode_high_fires_early():
    x = torch.tensor([[1.0, 0.0, 0.5]])
    spikes = temporal_encode(x, n_timesteps=10)
    # x=1.0 fires at t=0.
    assert spikes[0, 0, 0] == 1.0
    # x=0.5 fires around middle (t≈4 or 5).
    assert spikes[0, 4:6, 2].sum() == 1.0
    # x=0.0 silent.
    assert spikes[0, :, 1].sum() == 0.0
