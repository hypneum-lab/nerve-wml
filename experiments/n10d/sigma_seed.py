"""Extract seed spreads from the N10-D development pilot (stdlib only).

L0 DRAFT, not frozen. Written and committed BEFORE the pilot (PI decision
2026-10-31, item 9; amended 2026-11-06, A4; paired sigma_D by PI decision
2026-11-06 16:55 CET, protocol section 9; reconciled with PR #100 the same
day, protocol section 8). Reads ONLY the ``nocomm`` and ``eng`` records of
condition ``asym`` on the eight development seeds 9600-9607; any other arm
name is refused (``READABLE_ARMS``).

Declared blinding breach (protocol section 4, "Pilot read-out (declared)"):
sigma_D is the sd of the per-seed differences S_eng - S_nocomm, so the
script pairs the two arms. It prints only n_eval, sigma_seed, sigma_D and
k; it never computes the mean of the differences and never prints a
per-seed score or difference.

Input JSON: list of {"seed", "condition", "arm", "success", "n_eval"}.
``success`` is the evaluation success of the frozen policy on ``n_eval``
fresh episodes after training (protocol section 3, A8), so it is a
Binomial(n_eval, p)/n_eval draw given the seed.

- sigma_seed (nocomm, asym) calibrates the paired null (null.py):
  sigma_seed^2 = max(0, sample variance across seeds - mean p(1-p)/n_eval).
  Known limit: a blind asym policy has expectation exactly 1/C, so it is 0
  up to binomial noise (estimate 0 in about 57 % of 8-seed pilots,
  P(chi2_7 < 7) = 0.571); it does not size k.
- sigma_D sizes k: sample sd (ddof = 1) of D_s = S_eng,s - S_nocomm,s
  over the development seeds, binomial term included. The two arms must
  carry the same seed identifiers and, seed by seed, the same n_eval.
  Var(D) = Var(S_eng) + Var(S_nocomm) - 2 Cov, hence the pairing.
- k is sized for the test actually used: a one-sided sign test,
  Binomial(k, 1/2) under H0, threshold m_k, power P(Bin(k, q) >= m_k),
  and never below the normal-approximation k of the paired mean.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from statistics import NormalDist

DEV_SEEDS = tuple(range(9600, 9608))
READABLE_ARMS = ("nocomm", "eng")
MIN_K = 8
MAX_K = 200
MARGIN = 0.30
ALPHA = 0.05
POWER = 0.90


def _rows(records: list[dict], arm: str) -> list[dict]:
    if arm not in READABLE_ARMS:
        raise ValueError(f"arm must be one of {READABLE_ARMS}, got {arm!r}")
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


def _paired_differences(records: list[dict]) -> list[float]:
    """Per-seed S_eng - S_nocomm; never returned by the CLI.

    ValueError if the arms do not carry the same seed identifiers or if
    the n_eval of a seed differs between the arms.
    """
    eng = {r["seed"]: r for r in _rows(records, "eng")}
    noc = {r["seed"]: r for r in _rows(records, "nocomm")}
    if eng.keys() != noc.keys():
        raise ValueError(f"unpaired seeds: eng {sorted(eng)}, nocomm {sorted(noc)}")
    diffs = []
    for s in sorted(eng):
        if eng[s]["n_eval"] != noc[s]["n_eval"]:
            raise ValueError(f"seed {s}: n_eval eng={eng[s]['n_eval']} nocomm={noc[s]['n_eval']}")
        diffs.append(eng[s]["success"] - noc[s]["success"])
    return diffs


def sigma_d(records: list[dict]) -> float:
    """Sample sd (ddof = 1) of the paired per-seed differences."""
    return statistics.stdev(_paired_differences(records))


def _binom_tail(k: int, m: int, q: float) -> float:
    """P(Binomial(k, q) >= m)."""
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


def sign_test_seeds(q: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Smallest k in [MIN_K, MAX_K] whose sign test reaches ``power`` at q.

    The power is not monotone in k (discrete threshold), hence the scan.
    ValueError above MAX_K: the design needs an amendment.
    """
    for k in range(MIN_K, MAX_K + 1):
        if sign_test_power(k, q, alpha) >= power:
            return k
    raise ValueError(f"no k <= {MAX_K} reaches power {power} at q = {q}")


def required_seeds_from_sigma_d(sigma_d: float, margin: float = MARGIN) -> int:
    """Normal-approximation k of the paired mean, sigma_D taken directly.

    k = max(MIN_K, ceil(((z_{1-alpha} + z_{power}) * sigma_D / margin)^2)),
    no factor 2 (sigma_D = 0.30 gives 9 at margin 0.30). A floor for
    ``required_seeds``, never the rule on its own.
    """
    z = NormalDist().inv_cdf(1 - ALPHA) + NormalDist().inv_cdf(POWER)
    return max(MIN_K, math.ceil((z * sigma_d / margin) ** 2))


def required_seeds(sigma_d: float) -> int:
    """k = max(sign-test k, paired-mean k), both in [MIN_K, MAX_K].

    Alternative: D_i ~ N(MARGIN, sigma_d), so q = Phi(MARGIN / sigma_d)
    (q = 1 when sigma_d = 0). ValueError above MAX_K: the pilot spread is
    too large for this design and needs an amendment.
    """
    q = 1.0 if sigma_d == 0 else NormalDist().cdf(MARGIN / sigma_d)
    try:
        k_sign = sign_test_seeds(q)
    except ValueError as exc:
        msg = f"sigma_D={sigma_d:.3f} needs more than {MAX_K} seeds"
        raise ValueError(msg) from exc
    return max(k_sign, required_seeds_from_sigma_d(sigma_d))


def main(argv: list[str] | None = None) -> str:
    """CLI body: n_eval comes from the filtered rows, never from recs[0]."""
    args = sys.argv[1:] if argv is None else argv
    with open(args[0]) as fh:
        recs = json.load(fh)
    n_eval = pilot_n_eval(recs, READABLE_ARMS)
    sg = sigma_seed(recs)
    sd = sigma_d(recs)
    k = required_seeds(sd)
    return f"n_eval={n_eval} sigma_seed={sg:.4f} sigma_D={sd:.4f} k={k}"


if __name__ == "__main__":
    print(main())
