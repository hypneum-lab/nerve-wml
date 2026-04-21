"""Unit tests for nerve_wml.methodology.mi_null_model.

Designed to be runnable on light hardware (no training, no GPU).
Each test uses small arrays (<=2000 codes, n_shuffles<=200) so the
full test file completes in well under a second on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from nerve_wml.methodology.mi_null_model import (
    NullModelResult,
    mi_argmax_onehot,
    null_model_mi,
)


def test_mi_argmax_onehot_identical_codes_equals_one() -> None:
    codes = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    assert abs(mi_argmax_onehot(codes, codes) - 1.0) < 1e-9


def test_mi_argmax_onehot_independent_codes_is_low() -> None:
    rng = np.random.default_rng(42)
    codes_a = rng.integers(0, 4, size=2000).astype(np.int64)
    codes_b = rng.integers(0, 4, size=2000).astype(np.int64)
    mi = mi_argmax_onehot(codes_a, codes_b)
    assert 0.0 <= mi < 0.05, f"expected near-0, got {mi}"


def test_mi_argmax_onehot_shape_mismatch_raises() -> None:
    a = np.array([0, 1, 2], dtype=np.int64)
    b = np.array([0, 1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="shape"):
        mi_argmax_onehot(a, b)


def test_mi_argmax_onehot_2d_raises() -> None:
    a = np.zeros((4, 4), dtype=np.int64)
    b = np.zeros((4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="ndim"):
        mi_argmax_onehot(a, b)


def test_mi_argmax_onehot_empty_raises() -> None:
    a = np.array([], dtype=np.int64)
    b = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="empty"):
        mi_argmax_onehot(a, b)


def test_null_model_identity_is_highly_significant() -> None:
    codes = np.tile(np.arange(4, dtype=np.int64), 250)
    rng = np.random.default_rng(0)
    rng.shuffle(codes)
    result = null_model_mi(codes, codes, n_shuffles=200, seed=0)
    assert result.mi_observed > 0.9
    assert result.z_score > 5.0
    assert result.p_value < 0.01


def test_null_model_independent_codes_not_significant() -> None:
    rng = np.random.default_rng(42)
    codes_a = rng.integers(0, 4, size=500).astype(np.int64)
    codes_b = rng.integers(0, 4, size=500).astype(np.int64)
    result = null_model_mi(codes_a, codes_b, n_shuffles=200, seed=0)
    assert abs(result.z_score) < 3.0
    assert result.p_value > 0.01


def test_null_model_result_dataclass_fields() -> None:
    codes = np.zeros(100, dtype=np.int64)
    result = null_model_mi(codes, codes, n_shuffles=10, seed=0)
    assert isinstance(result, NullModelResult)
    assert len(result.null_samples) == 10
    assert result.n_shuffles == 10


def test_null_model_n_shuffles_zero_raises() -> None:
    codes = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="n_shuffles"):
        null_model_mi(codes, codes, n_shuffles=0)


def test_null_model_reproducible_with_seed() -> None:
    rng = np.random.default_rng(123)
    codes_a = rng.integers(0, 8, size=300).astype(np.int64)
    codes_b = rng.integers(0, 8, size=300).astype(np.int64)
    r1 = null_model_mi(codes_a, codes_b, n_shuffles=50, seed=7)
    r2 = null_model_mi(codes_a, codes_b, n_shuffles=50, seed=7)
    assert r1.null_samples == r2.null_samples
    assert r1.mi_observed == r2.mi_observed
