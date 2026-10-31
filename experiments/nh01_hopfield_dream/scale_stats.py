"""Bootstrap on the extended reals for the NH-01 size-sweep bound sensitivity.

Pre-registration: ``docs/milestones/2026-10-30-nh01-scale-sweep.md``
(§4.2.1). ``run.bootstrap_mean_ci`` drops non-finite values, which is
right for its historical uses but wrong for the LOW/HIGH completions,
where a bound may be +inf or -inf. This module keeps every seed:

- values are sorted by seed identifier before resampling, so the result
  does not depend on the input order;
- resampling indices are those of ``run.bootstrap_mean_ci`` (same RNG
  seed, same ``integers`` call), so on finite data the result is
  identical to it;
- a vector may hold +inf or -inf but not both (``inf - inf`` is never
  computed); a resample mean is +inf (-inf) iff the resample holds an
  infinite value; a percentile whose two neighbours are finite is
  numpy's (computed by numpy on the means with the infinities replaced
  by the largest finite float), otherwise the infinite neighbour.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from experiments.nh01_hopfield_dream import run as nh


def _percentile_ext(means: np.ndarray, q: float) -> float:
    """numpy 'linear' percentile of ``means``, extended to +/-inf neighbours.

    The two order statistics around the virtual index decide the case. If
    one of them is infinite, the infinite neighbour is returned (or the
    lower one when the index is an integer or both are equal). Otherwise
    numpy itself interpolates, on the means with +/-inf replaced by
    +/- the largest finite float: the result is numpy's, bit for bit,
    including its handling of ties and signed zeros.
    """
    s = np.sort(means)
    n = s.size
    h = (q / 100.0) * (n - 1)  # numpy: virtual index = quantile * (n - 1)
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    a, b = float(s[lo]), float(s[hi])
    if math.isinf(a) or math.isinf(b):
        if h == lo or a == b:
            return a
        return b if math.isinf(b) else a
    big = np.finfo(np.float64).max
    finite = np.where(np.isposinf(means), big, np.where(np.isneginf(means), -big, means))
    return float(np.percentile(finite, q))


def bootstrap_mean_ci_ext(
    values: Sequence[float],
    seed_ids: Sequence[int],
    n_boot: int = nh.BOOT_N,
    seed: int | Sequence[int] = nh.BOOT_SEED,
) -> tuple[float, float, float]:
    """(mean, 2.5 %, 97.5 %) of the seed-level bootstrap, extended reals."""
    x = np.asarray(values, dtype=np.float64)
    ids = np.asarray(seed_ids)
    if x.shape != ids.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("values and seed_ids must be non-empty 1-d and aligned")
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate seed id")
    if np.isnan(x).any():
        raise ValueError("NaN value")
    if np.isposinf(x).any() and np.isneginf(x).any():
        raise ValueError("+inf and -inf in one vector")
    x = x[np.argsort(ids, kind="stable")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    point = float(x.mean())
    if np.isfinite(means).all():
        return point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return point, _percentile_ext(means, 2.5), _percentile_ext(means, 97.5)
