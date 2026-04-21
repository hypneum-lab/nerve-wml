"""Null-model mutual-information significance test via label permutation.

Used to reject the null hypothesis that two code streams (e.g. codes
emitted by two substrates) share information only by chance. The
procedure is the standard permutation test used in the
information-theory / neural-decoding literature:

  1. Measure MI_obs = estimator(codes_a, codes_b).
  2. Shuffle codes_b uniformly at random; recompute
     MI_null[i] = estimator(codes_a, permuted_b). Repeat n_shuffles
     times.
  3. Under the null hypothesis of independence (conditional on the
     marginals), MI_null is distributed around 0. If MI_obs sits
     significantly above the null distribution, the shared-information
     claim is not attributable to chance.

The default estimator is the argmax-one-hot entropy-normalised MI/H(a)
used throughout the nerve-wml paper. Other estimators (Kraskov kNN,
MINE) will land in ``mi_estimators.py`` in the v1.5.3 cycle and are
passable via the ``estimator`` keyword.

This module is GPU-free and pure numpy. It runs on commodity CPU in
O(n_shuffles * N) time -- for N=5000, n_shuffles=1000 it takes under
a few seconds on a 2020-era laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MiEstimator(Protocol):
    """Callable mapping two 1-D integer arrays to a single MI value."""

    def __call__(self, codes_a: np.ndarray, codes_b: np.ndarray) -> float: ...


def mi_argmax_onehot(codes_a: np.ndarray, codes_b: np.ndarray) -> float:
    """Entropy-normalised MI / H(a) on integer codes.

    This is the baseline estimator used in the v1.2.3 scientific
    baseline of nerve-wml (see paper section Information Transmission,
    Test (1)). The output is in [0, 1], where 1 means codes_b is a
    deterministic function of codes_a and 0 means independence.
    """
    if codes_a.shape != codes_b.shape:
        raise ValueError(
            f"codes_a shape {codes_a.shape} != codes_b shape {codes_b.shape}"
        )
    if codes_a.ndim != 1:
        raise ValueError(
            f"expected 1-D code arrays, got ndim={codes_a.ndim}"
        )
    if codes_a.size == 0:
        raise ValueError("cannot compute MI on empty arrays")

    n = codes_a.shape[0]
    alphabet_a = int(codes_a.max()) + 1
    alphabet_b = int(codes_b.max()) + 1

    joint = np.zeros((alphabet_a, alphabet_b), dtype=np.float64)
    np.add.at(joint, (codes_a, codes_b), 1.0)
    joint /= n

    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    mask = joint > 0
    denom = np.where(mask, p_a[:, None] * p_b[None, :], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(mask, np.log(joint / denom), 0.0)
    mi = float(np.sum(joint * log_ratio))

    h_a = float(-np.sum([p * np.log(p) for p in p_a if p > 0]))
    return mi / h_a if h_a > 0 else 0.0


@dataclass
class NullModelResult:
    """Result of a null-model MI permutation test.

    Attributes:
        mi_observed:  MI value on the unshuffled data.
        mi_null_mean: mean MI across the n_shuffles permutations.
        mi_null_std:  standard deviation across permutations.
        z_score:      (mi_observed - mi_null_mean) / mi_null_std, or
                      +inf if std is zero and observed > mean.
        p_value:      fraction of null samples with MI >= observed;
                      one-sided.
        n_shuffles:   number of permutations performed.
        null_samples: list of per-shuffle MI values (length n_shuffles).
    """

    mi_observed:  float
    mi_null_mean: float
    mi_null_std:  float
    z_score:      float
    p_value:      float
    n_shuffles:   int
    null_samples: list[float]


def null_model_mi(
    codes_a:    np.ndarray,
    codes_b:    np.ndarray,
    *,
    estimator:  MiEstimator = mi_argmax_onehot,
    n_shuffles: int = 1000,
    seed:       int = 0,
) -> NullModelResult:
    """Run the permutation test and return observed vs null distribution.

    Args:
        codes_a:    1-D integer array of emitted codes from substrate A.
        codes_b:    1-D integer array of emitted codes from substrate B.
                    Must have the same shape as codes_a.
        estimator:  MI estimator taking two 1-D arrays and returning a
                    float. Defaults to mi_argmax_onehot.
        n_shuffles: Number of permutations (default 1000).
        seed:       RNG seed for reproducibility.

    Returns:
        NullModelResult with observed MI, null-distribution summary
        statistics (mean, std, z-score, p-value), and the raw null
        samples for downstream plotting / diagnostics.
    """
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be >= 1, got {n_shuffles}")

    rng = np.random.default_rng(seed)
    mi_observed = estimator(codes_a, codes_b)

    null_samples = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        permuted_b = rng.permutation(codes_b)
        null_samples[i] = estimator(codes_a, permuted_b)

    mean_null = float(np.mean(null_samples))
    std_null = float(np.std(null_samples))
    if std_null > 0:
        z = (mi_observed - mean_null) / std_null
    elif mi_observed > mean_null:
        z = float("inf")
    else:
        z = 0.0
    p_value = float(np.sum(null_samples >= mi_observed)) / n_shuffles

    return NullModelResult(
        mi_observed=float(mi_observed),
        mi_null_mean=mean_null,
        mi_null_std=std_null,
        z_score=float(z),
        p_value=p_value,
        n_shuffles=n_shuffles,
        null_samples=null_samples.tolist(),
    )
