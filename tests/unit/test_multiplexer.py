"""TDD spec for track_p.multiplexer.GammaThetaMultiplexer (issue #1).

Scope: DSP layer only — roundtrip, shapes, PAC, differentiability, RNG.
Invariant under the α/β architectural choice (see issue #1 Q5) — i.e. tests
do NOT assume how SimNerve consumes the carrier.

All tests are expected to FAIL until track_p/multiplexer.py is implemented.
"""
from dataclasses import FrozenInstanceError

import pytest
import torch

from nerve_core.protocols import Nerve
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer

# ---------------------------------------------------------------------------
# Configuration & defaults
# ---------------------------------------------------------------------------

def test_config_defaults_reference_nerve_constants():
    """Defaults must come from Nerve.* — no hard-coded 40.0 / 6.0 / 64."""
    cfg = GammaThetaConfig()
    assert cfg.gamma_hz == Nerve.GAMMA_HZ
    assert cfg.theta_hz == Nerve.THETA_HZ
    assert cfg.alphabet_size == Nerve.ALPHABET_SIZE


def test_config_is_frozen():
    cfg = GammaThetaConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.gamma_hz = 42.0  # type: ignore[misc]


def test_config_symbols_per_theta_matches_lisman_idiart():
    """~7±2 γ symbols per θ period (Lisman & Idiart, 1995)."""
    cfg = GammaThetaConfig()
    assert 5 <= cfg.symbols_per_theta <= 9


def test_config_nyquist_guard():
    """sample_rate must comfortably exceed 2·γ (Nyquist)."""
    cfg = GammaThetaConfig()
    assert cfg.sample_rate_hz >= 4.0 * cfg.gamma_hz


# ---------------------------------------------------------------------------
# Shapes & module structure
# ---------------------------------------------------------------------------

def test_constellation_shape_matches_alphabet():
    mux = GammaThetaMultiplexer()
    assert mux.constellation.shape == (Nerve.ALPHABET_SIZE, 2)
    assert mux.constellation.requires_grad


def test_carrier_shape_is_bin_aligned():
    """Carrier length = round(sample_rate / γ) × symbols_per_theta.

    This pinning places γ exactly on a rFFT bin and makes every symbol
    window span an integer number of γ cycles → clean I/Q orthogonality
    and a clean PAC signature on rFFT bin 1 of the γ envelope.
    """
    cfg = GammaThetaConfig(sample_rate_hz=1000.0, gamma_hz=40.0, symbols_per_theta=7)
    mux = GammaThetaMultiplexer(cfg)
    codes = torch.randint(0, cfg.alphabet_size, (4, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    expected_t = round(cfg.sample_rate_hz / cfg.gamma_hz) * cfg.symbols_per_theta
    assert carrier.shape == (4, expected_t)  # (4, 175)
    assert carrier.dtype == torch.float32


def test_time_grid_registered_as_buffer():
    """_t_grid must be a buffer (follows .to(device)), not a Parameter."""
    mux = GammaThetaMultiplexer()
    assert "_t_grid" in dict(mux.named_buffers())
    assert "_t_grid" not in dict(mux.named_parameters())


def test_forward_rejects_too_many_symbols():
    """K > symbols_per_theta must raise — encoder can't fit extra γ chunks."""
    cfg = GammaThetaConfig(symbols_per_theta=7)
    mux = GammaThetaMultiplexer(cfg)
    bad = torch.zeros((1, cfg.symbols_per_theta + 1), dtype=torch.long)
    with pytest.raises((ValueError, AssertionError)):
        mux.forward(bad)


# ---------------------------------------------------------------------------
# DSP correctness — roundtrip & spectrum
# ---------------------------------------------------------------------------

def test_forward_demodulate_roundtrip_lossless_at_zero_noise():
    """Hard demodulation on a noise-free carrier must recover exact codes."""
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (8, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    recovered = mux.demodulate(carrier, hard=True)
    assert torch.equal(recovered, codes)


def test_carrier_spectrum_dominated_by_gamma_band():
    """Dominant rFFT peak (averaged over batch) must sit at γ.

    Single-carrier spectrum can drift to symbol-rate harmonics when only 7
    random symbols happen to align coherently. Averaging over a batch of 64
    independent code sequences suppresses those random harmonics and isolates
    the deterministic γ carrier at bin 7 (= exactly 40 Hz with T=175).
    """
    cfg = GammaThetaConfig(sample_rate_hz=1000.0)
    mux = GammaThetaMultiplexer(cfg, seed=0)
    torch.manual_seed(0)
    codes = torch.randint(0, cfg.alphabet_size, (64, cfg.symbols_per_theta))
    carrier = mux.forward(codes)  # [64, n_samples]
    spec = torch.fft.rfft(carrier, dim=-1).abs().mean(dim=0)
    freqs = torch.fft.rfftfreq(carrier.shape[-1], d=1.0 / cfg.sample_rate_hz)
    peak_hz = freqs[spec.argmax()].item()
    assert abs(peak_hz - cfg.gamma_hz) < 2.0, (
        f"batch-averaged spectral peak at {peak_hz} Hz ≠ γ ({cfg.gamma_hz} Hz)"
    )


def test_phase_amplitude_coupling_detectable():
    """γ-band power envelope must show θ-rate modulation (PAC signature).

    Batch-averaged to suppress single-sequence variance; PAC is a statistical
    property of the encoder, not of one code draw.
    """
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    torch.manual_seed(0)
    codes = torch.randint(0, cfg.alphabet_size, (64, cfg.symbols_per_theta))
    carriers = mux.forward(codes)  # [64, n_samples]
    ratios = [_theta_envelope_ratio(carriers[i], cfg) for i in range(carriers.shape[0])]
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio > 2.0, (
        f"mean θ-band/low-freq-other ratio on γ envelope = {mean_ratio:.2f} "
        f"over {len(ratios)} draws — no phase-amplitude coupling detected"
    )


# ---------------------------------------------------------------------------
# Differentiability & reproducibility
# ---------------------------------------------------------------------------

def test_constellation_gradient_flows_through_carrier():
    mux = GammaThetaMultiplexer(seed=0)
    cfg = mux.cfg
    codes = torch.randint(0, cfg.alphabet_size, (2, cfg.symbols_per_theta))
    carrier = mux.forward(codes)
    carrier.pow(2).mean().backward()
    g = mux.constellation.grad
    assert g is not None
    assert g.abs().sum().item() > 0.0, "constellation received no gradient"


def test_seed_determinism_and_global_rng_isolation():
    """Same seed → same constellation. Global RNG untouched by __init__."""
    torch.manual_seed(1234)
    anchor = torch.randn(1).item()

    a = GammaThetaMultiplexer(seed=7)
    b = GammaThetaMultiplexer(seed=7)
    assert torch.equal(a.constellation, b.constellation)

    torch.manual_seed(1234)
    anchor_after = torch.randn(1).item()
    assert anchor == anchor_after, "__init__ polluted the global torch RNG"


# ---------------------------------------------------------------------------
# Helpers (pure torch.fft — no scipy dep, no torch.signal.hilbert)
# ---------------------------------------------------------------------------

def _theta_envelope_ratio(carrier: torch.Tensor, cfg: GammaThetaConfig) -> float:
    """Ratio of θ-band peak to out-of-band peak on the γ-envelope spectrum.

    Detects phase-amplitude coupling without needing a full Hilbert transform:
    1. Bandpass carrier around γ via rFFT mask.
    2. Square the result → power envelope.
    3. rFFT the zero-mean envelope.
    4. Peak in [θ_hz ± 1] vs peak elsewhere (excluding DC).
    """
    n = carrier.numel()
    spec = torch.fft.rfft(carrier)  # DSP convention X(f); lowercased per ruff N806
    freqs = torch.fft.rfftfreq(n, d=1.0 / cfg.sample_rate_hz)

    g_mask = (freqs >= cfg.gamma_hz - 10.0) & (freqs <= cfg.gamma_hz + 10.0)
    gamma_only = torch.fft.irfft(spec * g_mask.to(spec.dtype), n=n)

    power = gamma_only.pow(2)
    env_spec = torch.fft.rfft(power - power.mean()).abs()  # γ-envelope spectrum

    # θ band: around configured θ_hz. "Elsewhere" constrained to PAC-relevant
    # low-frequency range (< γ/2) to exclude 2γ harmonics — artefacts of cos²
    # in the power envelope, not PAC competitors.
    theta_band = (freqs >= cfg.theta_hz - 1.0) & (freqs <= cfg.theta_hz + 1.0)
    elsewhere = (~theta_band) & (freqs > 0.5) & (freqs < cfg.gamma_hz / 2.0)

    return (env_spec[theta_band].max() / (env_spec[elsewhere].max() + 1e-12)).item()
