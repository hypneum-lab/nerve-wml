"""Pre-data calibrated nulls for NH-96 (stdlib only, seconds on one core).

L0 DRAFT, not frozen (amended 2026-11-06 after the blind contradictor pass,
reconciled with PR #100 the same day, protocol section 8).

Statistic: D = mean over k seeds of (S_eng - S_nocomm), S = team success
of the frozen policy evaluated on ``n_eval`` fresh episodes (protocol
section 3, A8), so S | p ~ Binomial(n_eval, p) / n_eval.

H0 (asym, the only condition in H1): the message carries no usable
information, so neither arm can exceed the exact blind ceiling B0 = 1/C.
Per seed, one latent effect u ~ N(0, sigma_seed) is shared by both arms
(paired null, PI decision 2026-11-01, brainstorm PR #102 item 2); the latent
success is p = clip(B0 + u, 0, B0): it is bounded ABOVE by B0 because no
blind policy beats 1/C (A6). Both arms are then evaluated independently.
``sigma_seed`` is measured on the nocomm/asym development seeds before
the freeze, never on registered seeds. ``paired=False`` keeps the former
independent-arm draw as a sensitivity check only, never for a decision.

``simulate_level_null`` gives the null of a single arm mean (a level, not
a difference); controls 2 and 4 and the sym gate use its quantiles (A9),
never the P99 of D. On sym, B0 = 1.0 and sigma_seed = 0: every draw is
exactly 1.0. Same pipeline: N >= 1000 draws, empirical p = (hits+1)/(N+1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .spec import TaskSpec, blind_ceiling


@dataclass(frozen=True)
class NullSummary:
    median: float
    p95: float
    p99: float
    n_draws: int


def _latent(rng: random.Random, p0: float, sigma: float) -> float:
    return min(p0, max(0.0, p0 + rng.gauss(0.0, sigma)))


def _evaluate(rng: random.Random, p: float, n_eval: int) -> float:
    return sum(rng.random() < p for _ in range(n_eval)) / n_eval


def simulate_null(
    spec: TaskSpec,
    k_seeds: int,
    n_eval: int,
    sigma_seed: float,
    n_draws: int = 10_000,
    rng_seed: int = 20261101,
    paired: bool = True,
) -> list[float]:
    """Draws of D under H0; ``paired=False`` is the sensitivity variant."""
    rng = random.Random(rng_seed)
    p0 = blind_ceiling(spec)
    out = []
    for _ in range(n_draws):
        total = 0.0
        for _ in range(k_seeds):
            p_eng = _latent(rng, p0, sigma_seed)
            p_noc = p_eng if paired else _latent(rng, p0, sigma_seed)
            total += _evaluate(rng, p_eng, n_eval) - _evaluate(rng, p_noc, n_eval)
        out.append(total / k_seeds)
    return out


def simulate_level_null(
    spec: TaskSpec,
    k_seeds: int,
    n_eval: int,
    sigma_seed: float,
    n_draws: int = 10_000,
    rng_seed: int = 20261102,
) -> list[float]:
    """Null of one arm's mean success over k seeds (blind arm at B0)."""
    rng = random.Random(rng_seed)
    p0 = blind_ceiling(spec)
    return [
        sum(_evaluate(rng, _latent(rng, p0, sigma_seed), n_eval) for _ in range(k_seeds)) / k_seeds
        for _ in range(n_draws)
    ]


def summarize(draws: list[float]) -> NullSummary:
    s = sorted(draws)
    n = len(s)
    return NullSummary(s[n // 2], s[int(0.95 * n)], s[int(0.99 * n)], n)


def empirical_p(observed: float, draws: list[float]) -> float:
    return (sum(d >= observed for d in draws) + 1) / (len(draws) + 1)


if __name__ == "__main__":
    for c in (2, 4, 8):
        for sig in (0.0, 0.02, 0.05):
            spec = TaskSpec("asym", 16, c)
            r = summarize(simulate_null(spec, 8, 500, sig))
            ind = summarize(simulate_null(spec, 8, 500, sig, paired=False))
            lv = summarize(simulate_level_null(spec, 8, 500, sig))
            print(
                f"asym C={c} sigma={sig:.2f} D: median={r.median:+.3f} "
                f"P95={r.p95:.3f} P99={r.p99:.3f} indep P99={ind.p99:.3f} "
                f"| level P99={lv.p99:.3f} (B0={blind_ceiling(spec):.3f}) "
                f"N={r.n_draws}"
            )
    spec = TaskSpec("sym", 16, 4)
    r = summarize(simulate_null(spec, 8, 500, 0.0))
    lowest = min(simulate_level_null(spec, 8, 500, 0.0))
    print(
        f"sym C=4 sigma=0.00 D: median={r.median:+.3f} P95={r.p95:.3f} "
        f"P99={r.p99:.3f} | level min={lowest:.3f} (B0=1.000) N={r.n_draws}"
    )
