"""Tests for the SpectrogramEncoder + MlpWML.from_spectrogram factory (issue #7).

Covers the canonical "raw waveform → carrier" path that bouba_sens (and any
future audio / bio-signal consumer) gets via ``MlpWML.from_spectrogram(...)``
instead of re-deriving the FFT pipeline.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from track_w.mlp_wml import MlpWML
from track_w.spectrogram import SpectrogramEncoder


# ---------------------------------------------------------------------------
# Shape + factory contract
# ---------------------------------------------------------------------------


def test_factory_returns_spectrogram_encoder():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360,
        window_sec=1.0,
        hop_sec=0.05,
        n_bins=128,
        target_carrier_dim=16,
    )
    assert isinstance(encoder, SpectrogramEncoder)


def test_carrier_shape_matches_target_dim():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=16, seed=0,
    )
    waveform = torch.randn(4, 360 * 3)  # 3 s of fake ECG @ 360 Hz, batch 4
    carrier = encoder(waveform)
    assert carrier.shape == (4, 16)
    assert carrier.dtype == torch.float32


def test_unbatched_waveform_is_auto_unsqueezed():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=8, seed=0,
    )
    waveform_1d = torch.randn(360 * 2)  # 2 s, no batch dim
    carrier = encoder(waveform_1d)
    assert carrier.shape == (1, 8)


def test_invalid_waveform_shape_raises():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=8, seed=0,
    )
    bad = torch.randn(2, 3, 360 * 2)  # 3-D, not allowed
    with pytest.raises(ValueError, match="expected"):
        encoder(bad)


# ---------------------------------------------------------------------------
# Determinism (issue #7 acceptance criterion)
# ---------------------------------------------------------------------------


def test_deterministic_output_given_seed_and_input():
    encoder1 = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=16, seed=42,
    )
    encoder2 = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=16, seed=42,
    )
    torch.manual_seed(0)
    waveform = torch.randn(2, 360 * 2)
    out1 = encoder1(waveform)
    out2 = encoder2(waveform)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_different_seeds_give_different_projections():
    encoder_a = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=16, seed=0,
    )
    encoder_b = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=16, seed=1,
    )
    torch.manual_seed(0)
    waveform = torch.randn(2, 360 * 2)
    out_a = encoder_a(waveform)
    out_b = encoder_b(waveform)
    assert not torch.allclose(out_a, out_b, atol=1e-3)


# ---------------------------------------------------------------------------
# STFT plumbing — sample-rate-dependent window/hop computation
# ---------------------------------------------------------------------------


def test_stft_window_hop_derived_from_sample_rate():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, window_sec=1.0, hop_sec=0.5,
        n_bins=64, target_carrier_dim=8,
    )
    assert encoder.n_fft == 360
    assert encoder.hop_length == 180


def test_audio_sample_rate_scales_correctly():
    """22050 Hz audio → larger n_fft, smaller relative bin coverage."""
    encoder = MlpWML.from_spectrogram(
        sample_rate=22050, window_sec=0.025, hop_sec=0.010,
        n_bins=128, target_carrier_dim=16,
    )
    assert encoder.n_fft == round(0.025 * 22050)  # 551
    assert encoder.hop_length == round(0.010 * 22050)  # 220


def test_n_bins_exceeding_rfft_raises():
    """rfft of length n_fft yields n_fft//2+1 bins; asking for more must fail."""
    with pytest.raises(ValueError, match="exceeds"):
        # n_fft = 360, max bins = 181
        MlpWML.from_spectrogram(
            sample_rate=360, window_sec=1.0, n_bins=500, target_carrier_dim=8,
        )


# ---------------------------------------------------------------------------
# Differentiable path (gradient flows back to the projection layer)
# ---------------------------------------------------------------------------


def test_gradient_flows_through_projection():
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, n_bins=64, target_carrier_dim=8, seed=0,
    )
    waveform = torch.randn(2, 360 * 2, requires_grad=False)
    carrier = encoder(waveform)
    loss = carrier.sum()
    loss.backward()
    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()
    assert encoder.projection.weight.grad.abs().sum() > 0.0


# ---------------------------------------------------------------------------
# Realistic ECG smoke (mimics the bouba_sens MIT-BIH consumer)
# ---------------------------------------------------------------------------


def test_realistic_ecg_smoke_360hz_3s():
    """3 s of 360 Hz ECG should produce a sensible (B, 16) carrier.

    Mirrors the bouba_sens scripts/fetch_studyforrest_sample.py default:
    sample_rate=360, window_sec=1.0, hop_sec=0.05, n_bins=128.
    """
    torch.manual_seed(0)
    encoder = MlpWML.from_spectrogram(
        sample_rate=360, window_sec=1.0, hop_sec=0.05,
        n_bins=128, target_carrier_dim=16, seed=0,
    )
    # Synthesize a fake ECG: low-freq sinusoid + noise.
    t = torch.linspace(0, 3, steps=360 * 3)
    waveform = (torch.sin(2 * torch.pi * 1.2 * t) + 0.1 * torch.randn_like(t)).unsqueeze(0)
    carrier = encoder(waveform)
    assert carrier.shape == (1, 16)
    assert torch.isfinite(carrier).all()
