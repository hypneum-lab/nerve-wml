"""Unit tests for nerve_wml.methodology.bootstrap_ci_mi.

Light-weight: small arrays, small n_resamples, completes under 1s.
"""
from __future__ import annotations

import numpy as np
import pytest

from nerve_wml.methodology.bootstrap_ci_mi import (
    BootstrapCiResult,
    bootstrap_ci_mi,
)


def test_bootstrap_identical_codes_concentrates_on_one() -> None:
    codes = np.tile(np.arange(4, dtype=np.int64), 100)
    rng = np.random.default_rng(0)
    rng.shuffle(codes)
    result = bootstrap_ci_mi(codes, codes, n_resamples=100, seed=0)
    assert result.mi_point > 0.9
    assert result.mi_median > 0.9
    assert result.mi_ci95_low > 0.85


def test_bootstrap_independent_codes_concentrates_on_zero() -> None:
    rng = np.random.default_rng(42)
    codes_a = rng.integers(0, 4, size=1000).astype(np.int64)
    codes_b = rng.integers(0, 4, size=1000).astype(np.int64)
    result = bootstrap_ci_mi(codes_a, codes_b, n_resamples=200, seed=0)
    assert result.mi_point < 0.05
    assert result.mi_ci95_high < 0.10


def test_bootstrap_result_has_correct_quantile_ordering() -> None:
    rng = np.random.default_rng(0)
    codes_a = rng.integers(0, 6, size=300).astype(np.int64)
    codes_b = codes_a.copy()
    rng.shuffle(codes_b)  # break correlation partially
    result = bootstrap_ci_mi(codes_a, codes_b, n_resamples=100, seed=0)
    assert result.mi_ci95_low <= result.mi_p25 <= result.mi_median
    assert result.mi_median <= result.mi_p75 <= result.mi_ci95_high


def test_bootstrap_shape_mismatch_raises() -> None:
    a = np.array([0, 1, 2], dtype=np.int64)
    b = np.array([0, 1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        bootstrap_ci_mi(a, b)


def test_bootstrap_n_resamples_zero_raises() -> None:
    codes = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci_mi(codes, codes, n_resamples=0)


def test_bootstrap_reproducible_with_seed() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(0, 8, size=200).astype(np.int64)
    b = rng.integers(0, 8, size=200).astype(np.int64)
    r1 = bootstrap_ci_mi(a, b, n_resamples=50, seed=123)
    r2 = bootstrap_ci_mi(a, b, n_resamples=50, seed=123)
    assert r1.bootstrap_samples == r2.bootstrap_samples
    assert r1.mi_median == r2.mi_median


def test_bootstrap_result_dataclass() -> None:
    codes = np.zeros(50, dtype=np.int64)
    r = bootstrap_ci_mi(codes, codes, n_resamples=10, seed=0)
    assert isinstance(r, BootstrapCiResult)
    assert len(r.bootstrap_samples) == 10
    assert r.n_resamples == 10
