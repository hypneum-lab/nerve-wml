"""Pre-registered inference shared by P1bis-ext and P8c-nocap (pure NumPy).

Everything here is fixed by the two pre-registrations; changing a constant or
a rule after the freeze tag is an amendment.

* Paired differences are per seed (one value per seed and cell).
* p-values: exact one-sided sign-flip permutation test on the mean of the
  paired differences (all 2**n sign patterns enumerated for n <= 20; for
  n > 20 a Monte Carlo of 2**20 patterns with a fixed generator).  The
  p-value counts patterns whose mean is >= the observed mean (ties count),
  so it is never 0; the smallest attainable value is 2**-n.
* Superiority test of H0: mu <= delta0 uses d - delta0; inferiority test of
  H0: mu >= delta1 uses delta1 - d.  Both assume symmetry of d about the
  null value (standard sign-flip assumption).
* Holm step-down within a declared family at one-sided level ALPHA_ONE_SIDED.
* Reported intervals: 95 % percentile bootstrap of the mean (10 000
  resamples, generator seeded with BOOT_SEED) and Student t 95 % interval;
  both descriptive.  Decisions use only the Holm-adjusted permutation
  p-values and the point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

ALPHA_ONE_SIDED = 0.025
N_BOOT = 10_000
BOOT_SEED = 20261104
MC_PATTERNS = 2**20
MC_SEED = 20261104
EXACT_MAX_N = 20


def _sign_matrix(n: int) -> np.ndarray:
    """All 2**n sign patterns as an int8 array of shape (2**n, n)."""
    codes = np.arange(2**n, dtype=np.int64)[:, None]
    bits = (codes >> np.arange(n, dtype=np.int64)[None, :]) & 1
    return (1 - 2 * bits).astype(np.int8)


def signflip_p_greater(d: Sequence[float], null: float = 0.0) -> float:
    """One-sided p for H1: mean(d) > null (exact for n <= 20)."""
    x = np.asarray(d, dtype=np.float64) - null
    n = x.size
    if n == 0:
        return float("nan")
    obs = x.mean()
    tol = 1e-12 * max(1.0, float(np.abs(x).max()))
    if n <= EXACT_MAX_N:
        count = 0
        total = 0
        # enumerate in chunks to bound memory (2**20 x 20 int8 = 20 MB)
        signs = _sign_matrix(n)
        for c0 in range(0, signs.shape[0], 1 << 18):
            m = (signs[c0:c0 + (1 << 18)] * x[None, :]).mean(axis=1)
            count += int(np.count_nonzero(m >= obs - tol))
            total += m.size
        return count / total
    rng = np.random.default_rng(MC_SEED)
    s = rng.integers(0, 2, size=(MC_PATTERNS, n), dtype=np.int8) * 2 - 1
    m = (s * x[None, :]).mean(axis=1)
    return (int(np.count_nonzero(m >= obs - tol)) + 1) / (MC_PATTERNS + 1)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def boot_ci(d: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (descriptive)."""
    x = np.asarray(d, dtype=np.float64)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, x.size, size=(N_BOOT, x.size))
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def t_ci(d: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Student t CI of the mean (descriptive; degenerate when sd = 0)."""
    from scipy import stats as sps  # scipy is pinned in the study environment

    x = np.asarray(d, dtype=np.float64)
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    m, s = float(x.mean()), float(x.std(ddof=1))
    if s == 0.0:
        return (m, m)
    h = float(sps.t.ppf(0.5 + level / 2, n - 1)) * s / math.sqrt(n)
    return (m - h, m + h)


def family_decisions(
    diffs: dict[str, Sequence[float]],
    sup_margin: float,
    inf_margin: float,
    min_effect: float,
    alpha: float = ALPHA_ONE_SIDED,
) -> dict[str, dict]:
    """Apply the pre-registered per-test rule inside one Holm family.

    For each test key k with paired differences d_k:
      * p_sup  = sign-flip p for H1: mean > sup_margin (Holm within family);
      * p_inf  = sign-flip p for H1: mean < inf_margin (Holm within family);
      * label  = "success"   if mean >= min_effect and Holm(p_sup) < alpha,
                 "below"     if Holm(p_inf) < alpha (effect below inf_margin),
                 "inconclusive" otherwise.
    Keys whose list is empty are labelled "not_measured" and are removed from
    the family before Holm (the family size is the number of measured tests).
    """
    keys = [k for k, v in diffs.items() if len(v) > 0]
    out: dict[str, dict] = {}
    for k, v in diffs.items():
        if len(v) == 0:
            out[k] = {"label": "not_measured", "n": 0}
    if not keys:
        return out
    p_sup = [signflip_p_greater(diffs[k], sup_margin) for k in keys]
    p_inf = [signflip_p_greater([-x for x in diffs[k]], -inf_margin) for k in keys]
    a_sup, a_inf = holm(p_sup), holm(p_inf)
    for k, ps, pi, hs, hi in zip(keys, p_sup, p_inf, a_sup, a_inf, strict=True):
        d = np.asarray(diffs[k], dtype=np.float64)
        mean = float(d.mean())
        if mean >= min_effect and hs < alpha:
            label = "success"
        elif hi < alpha:
            label = "below"
        else:
            label = "inconclusive"
        out[k] = {
            "label": label,
            "n": int(d.size),
            "mean": mean,
            "sd": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
            "p_sup": ps,
            "p_sup_holm": hs,
            "p_inf": pi,
            "p_inf_holm": hi,
            "boot95": boot_ci(d),
            "t95": t_ci(d),
        }
    return out
