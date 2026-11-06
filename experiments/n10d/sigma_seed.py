"""Extract seed spreads from the N10-D development pilot (stdlib only).

Written and committed BEFORE the pilot (PI decision 2026-10-31, item 9;
amended 2026-11-06, A4). Reads ONLY the ``nocomm`` and ``eng`` records of
condition ``asym`` on the eight development seeds 9600-9607. It computes
spreads only: no arm mean, no per-seed difference and no pairing is ever
computed or printed, so the NH-96 contrast cannot be read through it.

Input JSON: list of {"seed", "condition", "arm", "success", "n_eval"}.
``success`` is the evaluation success of the frozen policy on ``n_eval``
fresh episodes after training (protocol section 3, A8), so it is a
Binomial(n_eval, p)/n_eval draw given the seed.

- sigma_seed (nocomm, asym) calibrates the paired null (null.py):
  sigma_seed^2 = max(0, sample variance across seeds - mean p(1-p)/n_eval).
- sigma_D bound sizes k: sd(S_eng) + sd(S_nocomm), total sample sd across
  seeds (binomial term included). It bounds sd(S_eng - S_nocomm) for any
  covariance (Cauchy-Schwarz), so no pairing is read and no factor 2 is
  needed.
- k is sized for the test actually used: a one-sided sign test,
  Binomial(k, 1/2) under H0, threshold m_k, power P(Bin(k, q) >= m_k).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from statistics import NormalDist

DEV_SEEDS = tuple(range(9600, 9608))
MIN_K = 8
MAX_K = 200
MARGIN = 0.30
ALPHA = 0.05
POWER = 0.90


def _rows(records: list[dict], arm: str) -> list[dict]:
    rows = [r for r in records if r["arm"] == arm and r["condition"] == "asym"]
    seeds = sorted(r["seed"] for r in rows)
    if tuple(seeds) != DEV_SEEDS:
        raise ValueError(f"need exactly {arm}/asym seeds {DEV_SEEDS}, got {seeds}")
    return rows


def _pilot_rows(records: list[dict]) -> list[dict]:
    return _rows(records, "nocomm")


def pilot_n_eval(records: list[dict], arms: tuple[str, ...] = ("nocomm",)) -> int:
    """n_eval of the filtered pilot rows; ValueError if they disagree."""
    values = {r["n_eval"] for arm in arms for r in _rows(records, arm)}
    if len(values) != 1:
        raise ValueError(f"filtered rows have heterogeneous n_eval: {sorted(values)}")
    return values.pop()


def sigma_seed(records: list[dict]) -> float:
    rows = _pilot_rows(records)
    s = [r["success"] for r in rows]
    p = statistics.fmean(s)
    binom = statistics.fmean(p * (1 - p) / r["n_eval"] for r in rows)
    return math.sqrt(max(0.0, statistics.variance(s) - binom))


def arm_sd(records: list[dict], arm: str) -> float:
    """Total sample sd of success across the development seeds of one arm."""
    return statistics.stdev(r["success"] for r in _rows(records, arm))


def sigma_d_bound(records: list[dict]) -> float:
    """Upper bound on sd(S_eng - S_nocomm) that never pairs the two arms."""
    return arm_sd(records, "eng") + arm_sd(records, "nocomm")


def _binom_tail(k: int, m: int, q: float) -> float:
    return sum(math.comb(k, i) * q**i * (1 - q) ** (k - i) for i in range(m, k + 1))


def sign_test_threshold(k: int, alpha: float = ALPHA) -> int:
    """Smallest m with P(Bin(k, 1/2) >= m) <= alpha (ties count as failures)."""
    for m in range(k + 1):
        if _binom_tail(k, m, 0.5) <= alpha:
            return m
    raise ValueError(f"no sign-test threshold at k={k}, alpha={alpha}")


def sign_test_power(k: int, q: float, alpha: float = ALPHA) -> float:
    """P(Bin(k, q) >= m_k), q = P(D_i > 0) under the alternative."""
    return _binom_tail(k, sign_test_threshold(k, alpha), q)


def required_seeds(sigma_d: float) -> int:
    """Smallest k in [MIN_K, MAX_K] whose sign test reaches POWER.

    Alternative: D_i ~ N(MARGIN, sigma_d), so q = Phi(MARGIN / sigma_d)
    (q = 1 when sigma_d = 0). The power is not monotone in k (discrete
    threshold), hence the scan. ValueError above MAX_K: the pilot spread
    is too large for this design and needs an amendment.
    """
    q = 1.0 if sigma_d == 0 else NormalDist().cdf(MARGIN / sigma_d)
    for k in range(MIN_K, MAX_K + 1):
        if sign_test_power(k, q) >= POWER:
            return k
    raise ValueError(f"sigma_D={sigma_d:.3f} needs more than {MAX_K} seeds")


def main(argv: list[str]) -> str:
    """CLI body: n_eval comes from the filtered rows, never from recs[0]."""
    with open(argv[0]) as fh:
        recs = json.load(fh)
    n_eval = pilot_n_eval(recs, ("nocomm", "eng"))
    sg = sigma_seed(recs)
    sd = sigma_d_bound(recs)
    k = required_seeds(sd)
    return f"n_eval={n_eval} sigma_seed={sg:.4f} sigma_D={sd:.4f} k={k}"


if __name__ == "__main__":
    print(main(sys.argv[1:]))
