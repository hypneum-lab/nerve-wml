"""Tests for paired statistical helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from nerve_wml.methodology.paired_tests import (
    bootstrap_ci,
    mann_whitney,
    wilcoxon_paired,
)


def test_wilcoxon_identical_inputs_p_one() -> None:
    a = [0.5, 0.6, 0.7, 0.8, 0.9]
    res = wilcoxon_paired(a, a)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["median_diff"] == 0.0
    assert res["cohens_dz"] == 0.0
    assert res["n"] == 5


def test_wilcoxon_clear_shift() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.1, size=30).tolist()
    b = (np.array(a) + 0.5).tolist()
    res = wilcoxon_paired(a, b)
    assert res["p_value"] < 1e-4
    assert res["median_diff"] < 0


def test_wilcoxon_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_mann_whitney_returns_shape() -> None:
    a = [0.1, 0.2, 0.3, 0.4]
    b = [0.5, 0.6, 0.7, 0.8]
    res = mann_whitney(a, b)
    assert set(res) >= {"statistic", "p_value", "median_diff", "cohens_dz", "n"}
    assert res["p_value"] < 0.1


def test_bootstrap_ci_covers_mean() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(2.0, 0.5, size=50).tolist()
    res = bootstrap_ci(values, n_resamples=500, seed=0)
    assert res["ci95_low"] < res["mean"] < res["ci95_high"]
    assert math.isfinite(res["median"])


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([], n_resamples=10, seed=0)
