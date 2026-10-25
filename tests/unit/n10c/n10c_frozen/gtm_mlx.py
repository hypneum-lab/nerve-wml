"""MLX port of the 1-channel, noise-free, role-free GammaThetaMultiplexer.

Line-by-line port of `track_p.multiplexer.GammaThetaMultiplexer.forward`
(theta_phase_offset = 0). Parity with the torch reference is gated by
`tests/unit/n10c/test_gtm_mlx.py` (atol 1e-5). No torch import here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class GTMConfigMLX:
    gamma_hz: float = 40.0
    theta_hz: float = 6.0
    sample_rate_hz: float = 1000.0
    alphabet_size: int = 16
    symbols_per_theta: int = 4


class GammaThetaMultiplexerMLX:
    def __init__(self, cfg: GTMConfigMLX, constellation: mx.array) -> None:
        self.cfg = cfg
        self.constellation = constellation  # [A, 2]
        n_per_gamma = max(1, round(cfg.sample_rate_hz / cfg.gamma_hz))
        self.n_samples = n_per_gamma * cfg.symbols_per_theta
        self._t_grid = mx.arange(self.n_samples, dtype=mx.float32) / cfg.sample_rate_hz

    def __call__(self, codes: mx.array) -> mx.array:
        """codes [B, K] int32 -> carrier [B, T] float32."""
        cfg = self.cfg
        if codes.shape[-1] > cfg.symbols_per_theta:
            raise ValueError(
                f"K={codes.shape[-1]} exceeds symbols_per_theta="
                f"{cfg.symbols_per_theta} (Lisman-Idiart capacity bound)"
            )
        raw_sym = self.constellation[codes]  # [B, K, 2]
        sym = self._normalise(raw_sym)
        return self._carrier_from_sym(sym)

    @staticmethod
    def _normalise(raw_sym: mx.array) -> mx.array:
        norm = mx.maximum(mx.sqrt(mx.sum(raw_sym * raw_sym, axis=-1, keepdims=True)), 1e-8)
        return raw_sym / norm

    def carrier_from_onehot(self, onehot: mx.array) -> mx.array:
        """Differentiable carrier from a (soft / straight-through) one-hot.

        onehot [B, K, A] -> carrier [B, T]. The carrier is linear in the
        per-slot unit symbol, so weighting the normalised constellation rows
        by the one-hot equals the hard-index carrier when the one-hot is hard,
        and passes gradient to the constellation and to the one-hot.
        """
        if onehot.shape[-2] > self.cfg.symbols_per_theta:
            raise ValueError("K exceeds symbols_per_theta")
        sym = onehot @ self._normalise(self.constellation)  # [B, K, 2]
        return self._carrier_from_sym(sym)

    def _carrier_from_sym(self, sym: mx.array) -> mx.array:
        cfg = self.cfg
        t = self._t_grid
        n_samples = self.n_samples
        k_cap = cfg.symbols_per_theta
        window_n = n_samples // k_cap
        k_active = sym.shape[1]

        two_pi_gamma_t = 2.0 * math.pi * cfg.gamma_hz * t
        gamma_i = mx.cos(two_pi_gamma_t)
        gamma_q = mx.sin(two_pi_gamma_t)

        effective_theta_hz = cfg.sample_rate_hz / n_samples
        two_pi_theta_t = 2.0 * math.pi * effective_theta_hz * t + 0.0
        theta_env = 0.55 + 0.45 * mx.cos(two_pi_theta_t)

        window_centers = (mx.arange(k_cap, dtype=t.dtype) + 0.5) * window_n
        sigma = window_n / 4.0
        idx = mx.arange(n_samples, dtype=t.dtype)
        gaussian_masks = mx.exp(
            -((mx.expand_dims(idx, 0) - mx.expand_dims(window_centers, 1)) ** 2)
            / (2.0 * sigma**2)
        )[:k_active]  # [K, T]

        i_t = mx.sum(sym[..., 0:1] * mx.expand_dims(gaussian_masks, 0), axis=1)
        q_t = mx.sum(sym[..., 1:2] * mx.expand_dims(gaussian_masks, 0), axis=1)

        carrier = (
            i_t * mx.expand_dims(gamma_i, 0) + q_t * mx.expand_dims(gamma_q, 0)
        ) * mx.expand_dims(theta_env, 0)
        return carrier.astype(mx.float32)
