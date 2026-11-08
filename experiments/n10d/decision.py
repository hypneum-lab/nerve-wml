"""Decision rules of N10-D (stdlib only). L0 DRAFT, not frozen.

Written 2026-11-06 from the PI decisions on TBD-15 and TBD-16
(multiple-choice in session), revised the same day by the sourced
analysis the PI chose to follow (protocol section 9); nothing here has
seen a run.

asym (TBD-16, CI rule), applied only once the asym controls of protocol
section 3 have passed; d holds the per-seed paired differences
S_eng - S_nocomm:
  mean(d) < FAILED_FLOOR                         -> "C" (failed prediction)
  CI lower >= margin and the sign test rejects   -> "A"
  CI upper < margin and the sign test rejects    -> "B" (weaker than predicted)
  otherwise                                      -> "non concluant"
CI = 95 % percentile paired bootstrap of the mean of d for now; the CI
method (paired t or expanded percentile) is open until the freeze
(TBD-16, protocol section 9).

Backends (TBD-10): identical verdicts A, B or C propagate; any other pair
is "non concluant" (``combine_backends``).

sym leak check (TBD-15): gate on eng_sym only (mean over k seeds >=
1 - SYM_EPSILON); nocomm_sym is not gated, it is reported with its CI
(``bootstrap_ci`` of the nocomm_sym values). Case D fires iff the CI lower
bound of D_sym = mean(eng_sym - nocomm_sym) >= SYM_LEAK_DELTA.
"""

from __future__ import annotations

import random
import statistics

from .sigma_seed import ALPHA, MARGIN, sign_test_critical
from .spec import SYM_EPSILON, SYM_LEAK_DELTA

FAILED_FLOOR = 0.10  # case C threshold (TBD-6)
CI_LEVEL = 0.95
N_BOOT = 10_000
BOOT_SEED = 20261106
PROPAGATING = ("A", "B", "C")


def _quantile(sorted_values: list[float], q: float) -> float:
    """Index rule shared with required_seeds_mc: s[min(n - 1, int(q * n))]."""
    n = len(sorted_values)
    return sorted_values[min(n - 1, int(q * n))]


def bootstrap_ci(
    d: list[float], n_boot: int = N_BOOT, level: float = CI_LEVEL, rng_seed: int = BOOT_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of the values d."""
    if len(d) < 2:
        raise ValueError("need at least 2 seeds")
    rng = random.Random(rng_seed)
    k = len(d)
    means = sorted(statistics.fmean(rng.choices(d, k=k)) for _ in range(n_boot))
    tail = (1 - level) / 2
    return _quantile(means, tail), _quantile(means, 1 - tail)


def sign_test_rejects(d: list[float], alpha: float = ALPHA) -> bool:
    """One-sided sign test of P(D_i > 0) > 1/2; D_i = 0 counts as not positive."""
    c = sign_test_critical(len(d), alpha)
    return c is not None and sum(x > 0 for x in d) >= c


def decide_asym(
    d: list[float], margin: float = MARGIN, n_boot: int = N_BOOT, rng_seed: int = BOOT_SEED
) -> str:
    """Per-backend asym status under the CI rule (TBD-16)."""
    if statistics.fmean(d) < FAILED_FLOOR:
        return "C"
    lower, upper = bootstrap_ci(d, n_boot, rng_seed=rng_seed)
    signs = sign_test_rejects(d)
    if lower >= margin and signs:
        return "A"
    if upper < margin and signs:
        return "B"
    return "non concluant"


def combine_backends(first: str, second: str) -> str:
    """Global status (TBD-10): identical A, B or C propagate, else "non concluant"."""
    return first if first == second and first in PROPAGATING else "non concluant"


def sym_gate(eng_sym: list[float], epsilon: float = SYM_EPSILON) -> bool:
    """Gate of control 3 for sym: eng_sym only, mean over seeds >= 1 - epsilon."""
    return statistics.fmean(eng_sym) >= 1 - epsilon


def sym_leak_status(
    eng_sym: list[float],
    nocomm_sym: list[float],
    delta: float = SYM_LEAK_DELTA,
    n_boot: int = N_BOOT,
    rng_seed: int = BOOT_SEED,
) -> str:
    """ "not measurable", "D" (audit required) or "no leak signal"."""
    if len(eng_sym) != len(nocomm_sym):
        raise ValueError("eng_sym and nocomm_sym must be paired by seed")
    if not sym_gate(eng_sym):
        return "not measurable"
    d_sym = [e - n for e, n in zip(eng_sym, nocomm_sym, strict=True)]
    lower, _ = bootstrap_ci(d_sym, n_boot, rng_seed=rng_seed)
    return "D" if lower >= delta else "no leak signal"
