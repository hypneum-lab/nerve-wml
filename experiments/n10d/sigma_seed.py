"""Extract sigma_seed from the N10-D development pilot (stdlib only).

Written and committed BEFORE the pilot (PI decision 2026-10-31, item 9).
Reads ONLY the ``nocomm`` records of condition ``asym`` on the eight
development seeds 9600-9607. Records of any other arm are never inspected,
so the NH-96 contrast cannot be read through this script.

Input JSON: list of {"seed", "condition", "arm", "success", "n_eval"}.
sigma_seed^2 = max(0, sample variance of success across seeds
                      - mean binomial variance p(1-p)/n_eval).
Also provides the pre-written power formula for the seed count k.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from statistics import NormalDist

DEV_SEEDS = tuple(range(9600, 9608))
MIN_K = 5
MARGIN = 0.30
ALPHA = 0.05
POWER = 0.90


def sigma_seed(records: list[dict]) -> float:
    rows = [r for r in records if r["arm"] == "nocomm" and r["condition"] == "asym"]
    seeds = sorted(r["seed"] for r in rows)
    if tuple(seeds) != DEV_SEEDS:
        raise ValueError(f"need exactly nocomm/asym seeds {DEV_SEEDS}, got {seeds}")
    s = [r["success"] for r in rows]
    p = statistics.fmean(s)
    binom = statistics.fmean(p * (1 - p) / r["n_eval"] for r in rows)
    return math.sqrt(max(0.0, statistics.variance(s) - binom))


def required_seeds(sigma: float, p0: float, n_eval: int) -> int:
    """k = max(5, ceil(((z_{1-alpha} + z_{power}) * sigma_D / margin)^2)).

    sigma_D^2 = 2 * (sigma_seed^2 + p(1-p)/n_eval) with p = 0.5 (worst case
    for the binomial term, p0 is kept for the report only). One-sided
    paired test of D > 0, normal approximation, detecting D = MARGIN.
    """
    nd = NormalDist()
    sigma_d = math.sqrt(2 * (sigma**2 + 0.25 / n_eval))
    k = ((nd.inv_cdf(1 - ALPHA) + nd.inv_cdf(POWER)) * sigma_d / MARGIN) ** 2
    return max(MIN_K, math.ceil(k))


if __name__ == "__main__":
    recs = json.load(open(sys.argv[1]))
    sg = sigma_seed(recs)
    n_eval = recs[0]["n_eval"]
    print(f"sigma_seed={sg:.4f} k={required_seeds(sg, 0.25, n_eval)}")
