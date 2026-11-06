"""Seed count k by simulation of the actual decision rule (TBD-13).

L0 DRAFT, not frozen. Written 2026-11-06 from the PI decision on TBD-13
(multiple-choice in session); k raised from 18 to 20 the same day by the
sourced analysis the PI chose to follow (protocol section 9). Uses no run
data: every D_i is drawn from the planning assumption below.

Planning assumption (an inference used for planning, not a measurement):
bimodal seeds; a fraction PLAN_PI = 0.8 learns the engineered code
(D_i ~ N(0.75, 0.03)), the rest stays blind (D_i ~ N(0, 0.027), the
binomial sd of a paired difference at n_eval = 500 and p = 1/4). Then
Delta = 0.60, sigma_D = 0.75 * sqrt(pi * (1 - pi)) ~ 0.30, q = 0.90.

Rule simulated (``decision.decide_asym``, TBD-16): A iff mean(d) >= 0.10,
the lower bound of the 95 % percentile paired bootstrap CI >= margin and
the one-sided sign test rejects at alpha 0.05. Same quantile index rule as
``decision._quantile``. Global power = p ** n_backends (conjunction of
TBD-10, backends assumed independent). The CI method is open until the
freeze (TBD-16); the simulation must be rerun with the method chosen.

Command used for the protocol (macM1, numpy):
    python -m experiments.n10d.required_seeds_mc --sims 20000 --boot 2000
``--scan`` adds the former search for the smallest k meeting the target.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .decision import FAILED_FLOOR
from .sigma_seed import ALPHA, MARGIN, sign_test_critical

PLAN_PI = 0.8
PLAN_D_LEARNED = 0.75
PLAN_SD_LEARNED = 0.03
PLAN_SD_BLIND = 0.027
SENSITIVITY_PI = (0.70, 0.75, 0.80)
MC_SEED = 20261106
# PI decision 2026-11-06 ~18:00 (follow the sourced analysis): 20 per backend.
K_CONFIRM = 20


def draw_d(rng: np.random.Generator, n_sims: int, k: int, pi: float) -> np.ndarray:
    learned = rng.random((n_sims, k)) < pi
    hi = rng.normal(PLAN_D_LEARNED, PLAN_SD_LEARNED, (n_sims, k))
    lo = rng.normal(0.0, PLAN_SD_BLIND, (n_sims, k))
    return np.where(learned, hi, lo)


def prob_a(
    k: int,
    pi: float = PLAN_PI,
    n_sims: int = 20_000,
    n_boot: int = 2_000,
    margin: float = MARGIN,
    seed: int = MC_SEED,
    chunk: int = 200,
) -> float:
    """Monte-Carlo P(status A) for one backend with k seeds."""
    rng = np.random.default_rng([seed, k, int(round(pi * 1000))])
    crit = sign_test_critical(k, ALPHA)
    if crit is None:
        return 0.0
    i_lo = min(n_boot - 1, int(0.025 * n_boot))
    hits = 0
    done = 0
    while done < n_sims:
        c = min(chunk, n_sims - done)
        d = draw_d(rng, c, k, pi)
        idx = rng.integers(0, k, (c, n_boot, k))
        boot = d[np.arange(c)[:, None, None], idx].mean(axis=2)
        boot.sort(axis=1)
        lower = boot[:, i_lo]
        ok = (d.mean(axis=1) >= FAILED_FLOOR) & (lower >= margin) & ((d > 0).sum(axis=1) >= crit)
        hits += int(ok.sum())
        done += c
    return hits / n_sims


def required_seeds_sim(
    target: float = 0.90, n_backends: int = 2, k_min: int = 11, k_max: int = 30, **kw
) -> tuple[int, dict[int, float]]:
    """Smallest k with prob_a(k) ** n_backends >= target, and the scanned table."""
    table: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        table[k] = prob_a(k, **kw)
        if table[k] ** n_backends >= target:
            return k, table
    raise ValueError(f"no k <= {k_max} reaches {target} for {n_backends} backends")


def report(k: int = K_CONFIRM, n_backends: int = 2, **kw) -> dict:
    """Power at k under the plan and its pi sensitivity, type I at the margin."""
    power = {str(pi): prob_a(k, pi=pi, **kw) for pi in SENSITIVITY_PI}
    return {
        "k": k,
        "power": power,
        "global_power": {pi: p**n_backends for pi, p in power.items()},
        "type1_at_margin": prob_a(k, pi=MARGIN / PLAN_D_LEARNED, **kw),
        "p_a_under_h0": prob_a(k, pi=0.0, **kw),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K_CONFIRM)
    ap.add_argument("--sims", type=int, default=20_000)
    ap.add_argument("--boot", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=MC_SEED)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--kmin", type=int, default=11)
    ap.add_argument("--kmax", type=int, default=30)
    a = ap.parse_args(argv)
    kw = {"n_sims": a.sims, "n_boot": a.boot, "seed": a.seed}
    out = report(a.k, **kw)
    if a.scan:
        k_min_target, table = required_seeds_sim(0.90, 2, a.kmin, a.kmax, **kw)
        out["scan"] = {"k_min_target": k_min_target, "power": {str(k): p for k, p in table.items()}}
    out.update({"sims": a.sims, "boot": a.boot, "seed": a.seed})
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
