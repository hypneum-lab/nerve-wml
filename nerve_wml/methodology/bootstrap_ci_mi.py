"""Bootstrap confidence interval for mutual information on discrete codes.

Complements ``null_model_mi`` (significance test) with an uncertainty
estimate: how tight is the MI measurement when we account for sample
noise? Standard non-parametric bootstrap via resampling with
replacement.

Procedure:
  1. For i in range(n_resamples):
     - indices = rng.choice(N, size=N, replace=True)
     - MI_bootstrap[i] = estimator(codes_a[indices], codes_b[indices])
  2. Report: median, p25, p75, 95% CI from empirical quantiles.

Unlike the null-model (which destroys the joint structure), bootstrap
preserves it; we only re-sample which rows we observe, to quantify
sampling variability of the estimator itself.

Pure numpy, CPU-friendly. Runs in O(n_resamples * N) on commodity
hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nerve_wml.methodology.mi_null_model import MiEstimator, mi_argmax_onehot


@dataclass
class BootstrapCiResult:
    """Result of a bootstrap CI on mutual information.

    Attributes:
        mi_point:          MI value on the original (unresampled) data.
        mi_median:         Median MI across bootstrap resamples.
        mi_p25:            25th percentile.
        mi_p75:            75th percentile.
        mi_ci95_low:       2.5th percentile.
        mi_ci95_high:      97.5th percentile.
        n_resamples:       Number of bootstrap resamples.
        bootstrap_samples: List of per-resample MI values.
    """

    mi_point:          float
    mi_median:         float
    mi_p25:            float
    mi_p75:            float
    mi_ci95_low:       float
    mi_ci95_high:      float
    n_resamples:       int
    bootstrap_samples: list[float]


def bootstrap_ci_mi(
    codes_a:     np.ndarray,
    codes_b:     np.ndarray,
    *,
    estimator:   MiEstimator = mi_argmax_onehot,
    n_resamples: int = 1000,
    seed:        int = 0,
) -> BootstrapCiResult:
    """Bootstrap confidence interval on MI estimator.

    Args:
        codes_a:     1-D integer array of emitted codes from substrate A.
        codes_b:     1-D integer array of emitted codes from substrate B.
        estimator:   MI estimator taking two 1-D arrays -> float.
        n_resamples: Number of bootstrap resamples (default 1000).
        seed:        RNG seed.

    Returns:
        BootstrapCiResult with point estimate, IQR, 95% CI, and raw
        bootstrap samples.
    """
    if codes_a.shape != codes_b.shape:
        raise ValueError(
            f"codes_a shape {codes_a.shape} != codes_b shape {codes_b.shape}"
        )
    if codes_a.ndim != 1:
        raise ValueError(f"expected 1-D codes, got ndim={codes_a.ndim}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    n = codes_a.shape[0]
    rng = np.random.default_rng(seed)

    mi_point = estimator(codes_a, codes_b)

    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = estimator(codes_a[idx], codes_b[idx])

    return BootstrapCiResult(
        mi_point=float(mi_point),
        mi_median=float(np.median(samples)),
        mi_p25=float(np.percentile(samples, 25)),
        mi_p75=float(np.percentile(samples, 75)),
        mi_ci95_low=float(np.percentile(samples, 2.5)),
        mi_ci95_high=float(np.percentile(samples, 97.5)),
        n_resamples=n_resamples,
        bootstrap_samples=samples.tolist(),
    )
