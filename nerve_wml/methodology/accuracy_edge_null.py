"""Null-control and bootstrap CI for the LIF accuracy-edge claim.

Provides two primitives parallel to the MI methodology in
``mi_null_model`` and ``bootstrap_ci_mi``:

* ``accuracy_edge_bootstrap_ci`` — non-parametric bootstrap 95 % CI
  on a list of per-seed signed accuracy edges
  (edge = acc_lif - acc_mlp or any scored difference).
  Delegates to the general ``bootstrap_ci`` in ``paired_tests`` and
  wraps the result in a typed dataclass with sign-rate and zero-
  exclusion flags for convenience.

* ``same_substrate_null_result`` — builds a ``SameSubstrateNullResult``
  from a list of MLP-vs-MLP (same-substrate) edges, used to characterise
  the init-variance band against which the real LIF edge is benchmarked.
  A correct null is symmetric around 0 (sign-rate ≈ coin-flip); the LIF
  edge is directional and should exceed the null's 95th-percentile band
  at pool scale.

Both functions are pure-python/numpy and run on commodity CPU. They do
NOT train any model — they operate on lists of pre-computed scalars
produced by the committed experiment drivers (``run_w2_hard*`` family
in ``scripts/track_w_pilot.py``).

Reference: LIF spike-expressivity edge null-control experiment
(kxkm-ai x86 Linux CPU, n=50 seeds, result JSON at
``papers/paper1/figures/lif_edge_null_n50.json``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nerve_wml.methodology.paired_tests import bootstrap_ci


@dataclass
class AccuracyEdgeCiResult:
    """Bootstrap 95 % CI on a list of per-seed accuracy edges.

    Attributes:
        n:              Number of seeds / observations.
        mean_edge:      Mean of the raw edges.
        median_edge:    Median of the raw edges.
        ci95_low:       Bootstrap 2.5th percentile of the mean.
        ci95_high:      Bootstrap 97.5th percentile of the mean.
        ci_excludes_zero: True when ci95_low > 0 or ci95_high < 0.
        sign_rate_pos:  Count of seeds with edge > 0.
        sign_rate_neg:  Count of seeds with edge < 0.
        n_resamples:    Bootstrap resample count.
    """

    n: int
    mean_edge: float
    median_edge: float
    ci95_low: float
    ci95_high: float
    ci_excludes_zero: bool
    sign_rate_pos: int
    sign_rate_neg: int
    n_resamples: int


@dataclass
class SameSubstrateNullResult:
    """Summary of a same-substrate (MLP-vs-MLP) edge null distribution.

    Attributes:
        n:               Number of null draws.
        mean_edge:       Mean signed edge (should be ≈ 0 for a valid null).
        mean_abs_edge:   Mean |edge| — the init-variance band centre.
        abs_p95:         95th percentile of |edge| — the init-variance
                         band upper bound used to test the real edge.
        sign_rate_pos:   Count of null draws with edge > 0.
        sign_rate_neg:   Count of null draws with edge < 0.
        ci95_low:        Bootstrap 2.5th percentile of the mean.
        ci95_high:       Bootstrap 97.5th percentile of the mean.
        ci_excludes_zero: True when ci95_low > 0 or ci95_high < 0.
                          Should be False for a symmetric null.
    """

    n: int
    mean_edge: float
    mean_abs_edge: float
    abs_p95: float
    sign_rate_pos: int
    sign_rate_neg: int
    ci95_low: float
    ci95_high: float
    ci_excludes_zero: bool


def accuracy_edge_bootstrap_ci(
    edges: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> AccuracyEdgeCiResult:
    """Non-parametric bootstrap 95 % CI on per-seed accuracy edges.

    ``edges`` is a list of signed per-seed differences
    (e.g. ``acc_lif - acc_mlp``).  A CI that excludes zero is evidence
    that the directional edge is not attributable to sampling noise.

    Args:
        edges:       Per-seed signed accuracy differences.
        n_resamples: Bootstrap resample count (default 10 000).
        seed:        RNG seed for reproducibility.

    Returns:
        AccuracyEdgeCiResult with point estimates, CI, and sign-rate
        summary.
    """
    if len(edges) == 0:
        raise ValueError("edges must be non-empty")
    arr = np.asarray(edges, dtype=np.float64)
    ci = bootstrap_ci(arr.tolist(), n_resamples=n_resamples, seed=seed)
    return AccuracyEdgeCiResult(
        n=int(arr.size),
        mean_edge=ci["mean"],
        median_edge=ci["median"],
        ci95_low=ci["ci95_low"],
        ci95_high=ci["ci95_high"],
        ci_excludes_zero=bool(ci["ci95_low"] > 0 or ci["ci95_high"] < 0),
        sign_rate_pos=int(np.sum(arr > 0)),
        sign_rate_neg=int(np.sum(arr < 0)),
        n_resamples=n_resamples,
    )


def same_substrate_null_result(
    null_edges: Sequence[float],
    *,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> SameSubstrateNullResult:
    """Summarise a same-substrate (MLP-vs-MLP) null edge distribution.

    The null is the init-variance band: two MLP cohorts seeded
    differently but trained on the same task and evaluated on the same
    eval task.  A valid null should be symmetric around 0 (sign-rate ≈
    50 / 50) and its 95th percentile of |edge| provides a threshold
    against which the real LIF edge is compared.

    Args:
        null_edges:  Per-seed signed edges from the same-substrate null.
        n_resamples: Bootstrap resample count for the signed CI.
        seed:        RNG seed.

    Returns:
        SameSubstrateNullResult with symmetry indicators and the
        init-variance band (mean_abs_edge, abs_p95).
    """
    if len(null_edges) == 0:
        raise ValueError("null_edges must be non-empty")
    arr = np.asarray(null_edges, dtype=np.float64)
    abs_arr = np.abs(arr)
    ci = bootstrap_ci(arr.tolist(), n_resamples=n_resamples, seed=seed)
    return SameSubstrateNullResult(
        n=int(arr.size),
        mean_edge=ci["mean"],
        mean_abs_edge=float(abs_arr.mean()),
        abs_p95=float(np.percentile(abs_arr, 95)),
        sign_rate_pos=int(np.sum(arr > 0)),
        sign_rate_neg=int(np.sum(arr < 0)),
        ci95_low=ci["ci95_low"],
        ci95_high=ci["ci95_high"],
        ci_excludes_zero=bool(ci["ci95_low"] > 0 or ci["ci95_high"] < 0),
    )
