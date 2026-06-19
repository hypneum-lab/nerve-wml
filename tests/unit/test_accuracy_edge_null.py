"""Unit tests for nerve_wml.methodology.accuracy_edge_null.

Tests cover:
- AccuracyEdgeCiResult: a symmetric null (sign-rate ≈ 0.5) has a CI
  that includes 0 and does not exclude zero.
- AccuracyEdgeCiResult: a constant positive edge has CI that excludes 0.
- SameSubstrateNullResult: same-substrate null is symmetric (sign-rate
  ≈ coin-flip, ci_excludes_zero=False with large n).
- sign-rate and abs_p95 computations are correct.
- Empty input raises ValueError.
- Reproducibility under fixed seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from nerve_wml.methodology import (
    AccuracyEdgeCiResult,
    SameSubstrateNullResult,
    accuracy_edge_bootstrap_ci,
    same_substrate_null_result,
)

# ---------------------------------------------------------------------------
# accuracy_edge_bootstrap_ci
# ---------------------------------------------------------------------------


def test_ci_constant_positive_edge_excludes_zero() -> None:
    """All edges positive → CI must exclude 0."""
    edges = [0.05] * 50
    result = accuracy_edge_bootstrap_ci(edges, n_resamples=1000, seed=0)
    assert isinstance(result, AccuracyEdgeCiResult)
    assert result.ci95_low > 0.0
    assert result.ci_excludes_zero is True
    assert result.sign_rate_pos == 50
    assert result.sign_rate_neg == 0


def test_ci_symmetric_null_includes_zero() -> None:
    """Edges drawn from zero-mean symmetric dist → CI must include 0."""
    rng = np.random.default_rng(42)
    edges = rng.normal(0.0, 0.05, size=200).tolist()
    result = accuracy_edge_bootstrap_ci(edges, n_resamples=2000, seed=0)
    assert isinstance(result, AccuracyEdgeCiResult)
    assert result.ci95_low < 0.0 < result.ci95_high
    assert result.ci_excludes_zero is False


def test_ci_sign_rates_sum_to_n() -> None:
    rng = np.random.default_rng(7)
    edges = rng.normal(0.01, 0.04, size=50).tolist()
    result = accuracy_edge_bootstrap_ci(edges, n_resamples=500, seed=0)
    assert result.sign_rate_pos + result.sign_rate_neg + (
        sum(1 for e in edges if e == 0.0)
    ) == result.n


def test_ci_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        accuracy_edge_bootstrap_ci([])


def test_ci_reproducible_with_seed() -> None:
    edges = list(np.random.default_rng(0).normal(0.02, 0.03, size=40))
    r1 = accuracy_edge_bootstrap_ci(edges, n_resamples=200, seed=99)
    r2 = accuracy_edge_bootstrap_ci(edges, n_resamples=200, seed=99)
    assert r1.ci95_low == r2.ci95_low
    assert r1.ci95_high == r2.ci95_high
    assert r1.mean_edge == r2.mean_edge


def test_ci_result_fields_consistent() -> None:
    edges = [0.01, 0.02, -0.005, 0.03, 0.015]
    result = accuracy_edge_bootstrap_ci(edges, n_resamples=200, seed=0)
    assert result.n == 5
    assert result.ci95_low <= result.mean_edge <= result.ci95_high
    assert result.n_resamples == 200


# ---------------------------------------------------------------------------
# same_substrate_null_result
# ---------------------------------------------------------------------------


def test_null_symmetric_distribution_does_not_exclude_zero() -> None:
    """A large symmetric null should have CI including 0."""
    rng = np.random.default_rng(13)
    null_edges = rng.normal(0.0, 0.02, size=200).tolist()
    result = same_substrate_null_result(null_edges, n_resamples=2000, seed=0)
    assert isinstance(result, SameSubstrateNullResult)
    assert result.ci_excludes_zero is False
    assert result.ci95_low < 0.0 < result.ci95_high


def test_null_sign_rate_near_half_for_symmetric() -> None:
    """Sign-rate of a zero-mean null should be near 50/50."""
    rng = np.random.default_rng(17)
    null_edges = rng.normal(0.0, 0.05, size=200).tolist()
    result = same_substrate_null_result(null_edges, n_resamples=500, seed=0)
    # Allow generous window: ±30 % around 50 %
    assert 70 <= result.sign_rate_pos + result.sign_rate_neg <= 200
    pos_frac = result.sign_rate_pos / result.n
    assert 0.3 <= pos_frac <= 0.7, f"pos_frac={pos_frac:.2f} is far from 0.5"


def test_null_abs_p95_positive() -> None:
    """abs_p95 must be non-negative for any input."""
    null_edges = [0.0] * 50
    result = same_substrate_null_result(null_edges, n_resamples=100, seed=0)
    assert result.abs_p95 == 0.0
    assert result.mean_abs_edge == 0.0


def test_null_abs_p95_correct() -> None:
    """abs_p95 should be the 95th percentile of absolute values."""
    null_edges = list(range(-10, 11))  # symmetric integers -10..10
    result = same_substrate_null_result(null_edges, n_resamples=100, seed=0)
    expected_p95 = float(np.percentile(np.abs(np.array(null_edges)), 95))
    assert abs(result.abs_p95 - expected_p95) < 1e-9


def test_null_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        same_substrate_null_result([])


def test_null_reproducible_with_seed() -> None:
    null_edges = list(np.random.default_rng(5).normal(0, 0.03, size=50))
    r1 = same_substrate_null_result(null_edges, n_resamples=200, seed=42)
    r2 = same_substrate_null_result(null_edges, n_resamples=200, seed=42)
    assert r1.ci95_low == r2.ci95_low
    assert r1.mean_abs_edge == r2.mean_abs_edge
    assert r1.abs_p95 == r2.abs_p95
