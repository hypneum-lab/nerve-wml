"""Power of the pre-registered decision rules (Monte Carlo, normal per-seed differences).

Reproduces the power tables of both pre-registrations::

    python -m experiments.hopfield_dream_conf.power

The exact sign-flip test is approximated by the one-sided paired t test (same
asymptotics; the sign-flip minimum p with n = 20 is 2**-20, far below every
Holm level used).  Two Holm bounds are reported: "best" (the test is the
last step, level 0.025) and "worst" (first step, level 0.025 / m).
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

N_SEEDS = 20
REPS = 20_000
SEED = 20261104


def power(mu, sd, min_effect, level, n=N_SEEDS, reps=REPS, inferiority_margin=None):
    rng = np.random.default_rng(SEED)
    d = rng.normal(mu, sd, size=(reps, n))
    m = d.mean(1)
    se = d.std(1, ddof=1) / np.sqrt(n)
    if inferiority_margin is None:     # success: mean >= min_effect and p(H1: mu > 0) < level
        p = sps.t.sf(m / se, n - 1)
        return float(np.mean((m >= min_effect) & (p < level)))
    p = sps.t.sf((inferiority_margin - m) / se, n - 1)   # "below": mu < margin
    return float(np.mean(p < level))


def table(title, mus, sds, min_effect, m_family, inferiority=None):
    print(f"\n## {title} (n = {N_SEEDS}, family size {m_family})")
    hdr = "mu \\ sd " + " ".join(f"{s:>13}" for s in sds)
    print(hdr)
    for mu in mus:
        cells = []
        for sd in sds:
            pb = power(mu, sd, min_effect, 0.025, inferiority_margin=inferiority)
            pw = power(mu, sd, min_effect, 0.025 / m_family, inferiority_margin=inferiority)
            cells.append(f"{pw:5.3f}/{pb:5.3f}")
        print(f"{mu:<8} " + " ".join(f"{c:>13}" for c in cells))


def main():
    print("cells: worst-Holm / best-Holm power")
    # P1bis-ext primary: pilot sd of b - d' and b - g at f = 0.20 is 0.0010-0.0041 (n = 9)
    table("P1bis-ext primary success (min effect 0.02)", [0.02, 0.025, 0.03, 0.042],
          [0.0041, 0.0082, 0.0164], 0.02, 6)
    table("P1bis-ext primary 'below' (true effect, margin 0.02)", [0.0, 0.01, 0.015],
          [0.0041, 0.0082, 0.0164], 0.02, 6, inferiority=0.02)
    # P8c-nocap primary: pilot paired sd 0.003-0.028 (matched budget, 200 cues)
    table("P8c-nocap primary success (min effect 0.10)", [0.10, 0.15, 0.20, 0.30],
          [0.03, 0.06, 0.10, 0.20, 0.30], 0.10, 12)
    table("P8c-nocap primary 'below' (margin 0.10)", [0.0, 0.03, 0.05],
          [0.03, 0.06, 0.10, 0.20, 0.30], 0.10, 12, inferiority=0.10)


if __name__ == "__main__":
    main()
