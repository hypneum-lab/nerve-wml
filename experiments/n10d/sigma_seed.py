"""Extract sigma_seed from the N10-D development pilot (stdlib only).

L0 DRAFT, not frozen. Written and committed BEFORE the pilot (PI decision
2026-10-31, item 9); revised 2026-11-06 after the blind contradictor pass.
Reads the records of arm ``nocomm`` of condition ``asym`` only, on the
eight development seeds 9600-9607 (TBD-14 closed 2026-11-06: the pilot
never reads ``eng``). D is never computed, so the NH-96 contrast cannot be
read through this script.

Known limit: a blind asym policy has expectation exactly 1/C, so
sigma_seed(nocomm/asym) is 0 up to binomial noise (estimate 0 in about 57 %
of 8-seed pilots, P(chi2_7 < 7) = 0.571). It does not size D: it is a
consistency check (a value clearly above 0 points to a side channel,
control 4b). k comes from the planning assumption (TBD-13/14), see
``required_seeds_mc.py``.

Input JSON: list of {"seed", "condition", "arm", "success", "n_eval"}.
sigma_seed^2 = max(0, sample variance of success across seeds
                      - mean binomial variance p(1-p)/n_eval).
Also provides the sign-test helpers and, for the report only, the former
seed-count formulas (obsolete since TBD-13 was closed, protocol section 4).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from statistics import NormalDist

DEV_SEEDS = tuple(range(9600, 9608))
READABLE_ARMS = ("nocomm",)  # TBD-14 closed: the pilot reads nocomm only
MIN_K = 5  # former floor, obsolete as a rule (TBD-13 closed); report only
MARGIN = 0.30  # 0.4 * (1 - 1/C), TBD-12 closed 2026-11-06
ALPHA = 0.05
POWER = 0.90


def _rows(records: list[dict], arm: str = "nocomm") -> list[dict]:
    if arm not in READABLE_ARMS:
        raise ValueError(f"arm must be one of {READABLE_ARMS}, got {arm!r}")
    rows = [r for r in records if r["arm"] == arm and r["condition"] == "asym"]
    seeds = sorted(r["seed"] for r in rows)
    if tuple(seeds) != DEV_SEEDS:
        raise ValueError(f"need exactly {arm}/asym seeds {DEV_SEEDS}, got {seeds}")
    return rows


def sigma_seed(records: list[dict], arm: str = "nocomm") -> float:
    rows = _rows(records, arm)
    s = [r["success"] for r in rows]
    p = statistics.fmean(s)
    binom = statistics.fmean(p * (1 - p) / r["n_eval"] for r in rows)
    return math.sqrt(max(0.0, statistics.variance(s) - binom))


def pilot_n_eval(records: list[dict], arm: str = "nocomm") -> int:
    """n_eval of the filtered rows; all filtered rows must share it."""
    values = {r["n_eval"] for r in _rows(records, arm)}
    if len(values) != 1:
        raise ValueError(f"filtered {arm}/asym rows have unequal n_eval: {sorted(values)}")
    return values.pop()


def _z_sum() -> float:
    nd = NormalDist()
    return nd.inv_cdf(1 - ALPHA) + nd.inv_cdf(POWER)


def required_seeds(sigma: float, p0: float, n_eval: int) -> int:
    """OBSOLETE (TBD-13 closed): independent-arms bound, report only.

    k = max(MIN_K, ceil(((z_{1-alpha} + z_{power}) * sigma_D / margin)^2)),
    sigma_D^2 = 2 * (sigma_seed^2 + p(1-p)/n_eval) with p = 0.5. The factor 2
    assumes two INDEPENDENT arms; it does not hold for a paired D, and
    sigma_seed of nocomm alone does not determine Var(D) =
    Var(S_eng) + Var(S_nocomm) - 2 Cov. p0 is kept for the report only.
    """
    sigma_d = math.sqrt(2 * (sigma**2 + 0.25 / n_eval))
    return required_seeds_from_sigma_d(sigma_d)


def required_seeds_from_sigma_d(sigma_d: float, margin: float = MARGIN) -> int:
    """OBSOLETE k_mean (TBD-13 closed 2026-11-06), kept for the report only.

    k = max(MIN_K, ceil(((z_{1-alpha} + z_{power}) * sigma_D / margin)^2)),
    no factor 2 (sigma_D = 0.30 gives k = 9 at margin 0.30). It is the power
    of a test of D > 0 when D = margin, which is neither the CI rule nor the
    sign test of TBD-16; k is set by ``required_seeds_mc.py`` instead.
    """
    return max(MIN_K, math.ceil((_z_sum() * sigma_d / margin) ** 2))


def _binom_tail(k: int, c: int, q: float) -> float:
    """P(Binomial(k, q) >= c)."""
    return sum(math.comb(k, i) * q**i * (1 - q) ** (k - i) for i in range(c, k + 1))


def sign_test_critical(k: int, alpha: float = ALPHA) -> int | None:
    """Smallest count c of positive D_i with P(Bin(k, 1/2) >= c) <= alpha."""
    for c in range(k + 1):
        if _binom_tail(k, c, 0.5) <= alpha:
            return c
    return None


def sign_test_power(k: int, q: float, alpha: float = ALPHA) -> float:
    """Power of the one-sided sign test when P(D_i > 0) = q."""
    c = sign_test_critical(k, alpha)
    return 0.0 if c is None else _binom_tail(k, c, q)


def sign_test_seeds(q: float, alpha: float = ALPHA, power: float = POWER,
                    k_max: int = 500) -> int:
    """Smallest k >= MIN_K whose sign test reaches ``power`` at P(D_i > 0) = q."""
    for k in range(MIN_K, k_max + 1):
        if sign_test_power(k, q, alpha) >= power:
            return k
    raise ValueError(f"no k <= {k_max} reaches power {power} at q = {q}")


def main(argv: list[str] | None = None) -> str:
    args = sys.argv[1:] if argv is None else argv
    path = args[0]
    arm = args[1] if len(args) > 1 else "nocomm"
    with open(path) as fh:
        recs = json.load(fh)
    sg = sigma_seed(recs, arm)
    n_eval = pilot_n_eval(recs, arm)
    # No k printed: k does not come from the pilot (TBD-13/14).
    return f"arm={arm} sigma_seed={sg:.4f} n_eval={n_eval}"


if __name__ == "__main__":
    print(main())
