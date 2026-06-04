"""Paired statistical tests and float-list bootstrap CI.

Complements ``bootstrap_ci_mi`` (which operates on code-pair arrays)
with the per-method-score helpers required to compare runners that
emit a single scalar per seed.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats


def _cohens_dz(diffs: np.ndarray) -> float:
    if diffs.size < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs) / sd)


def wilcoxon_paired(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired observations."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("expected 1-D sequences")
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return {
            "statistic":   0.0,
            "p_value":     1.0,
            "median_diff": 0.0,
            "cohens_dz":   0.0,
            "n":           int(a.size),
        }
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic":   float(res.statistic),
        "p_value":     float(res.pvalue),
        "median_diff": float(np.median(diffs)),
        "cohens_dz":   _cohens_dz(diffs),
        "n":           int(a.size),
    }


def mann_whitney(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> dict[str, Any]:
    """Two-sided Mann-Whitney U test on independent observations."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("expected 1-D sequences")
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled_sd = float(np.sqrt(
        (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0
    )) if a.size > 1 and b.size > 1 else 0.0
    effect = (
        float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0
    )
    return {
        "statistic":   float(res.statistic),
        "p_value":     float(res.pvalue),
        "median_diff": float(np.median(a) - np.median(b)),
        "cohens_dz":   effect,
        "n":           int(a.size + b.size),
    }


def bootstrap_ci(
    values:      Sequence[float],
    *,
    n_resamples: int = 1000,
    seed:        int = 0,
) -> dict[str, float]:
    """Non-parametric bootstrap 95% CI on a list of floats."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty values")
    if arr.ndim != 1:
        raise ValueError("expected 1-D sequence")
    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(arr[idx]))
    return {
        "mean":      float(np.mean(arr)),
        "median":    float(np.median(arr)),
        "ci95_low":  float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }
