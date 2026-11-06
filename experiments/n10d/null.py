"""Pre-data calibrated nulls for NH-96 (stdlib only, seconds on one core).

L0 DRAFT, not frozen (revised 2026-11-06 after the blind contradictor pass).

Statistic: D = mean over k seeds of (S_eng - S_nocomm), S = team success of
the frozen policy on n_eval independent evaluation episodes.
Under H0 (the message carries no usable information) both arms of a seed
are blind policies with the same expected success, the blind ceiling B0.

Paired null (default, PI decision 2026-11-01, brainstorm PR #102 item 2):
one latent p = clip(B0 + N(0, sigma_seed)) per seed, shared by both arms,
then one Binomial(n_eval, p) per arm. The former independent arm-by-seed
draw (``paired=False``) is kept for sensitivity only, never for a decision.

Single-arm null (``simulate_arm_null``): mean over k seeds of S_arm - B0,
used for thresholds on ONE arm (controls 2-4 of the protocol), never the
null of D. Same pipeline: N >= 1000 draws, empirical p = (hits+1)/(N+1).
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
    return min(1.0, max(0.0, p0 + rng.gauss(0.0, sigma)))


def _binomial_rate(rng: random.Random, p: float, n_eval: int) -> float:
    return sum(rng.random() < p for _ in range(n_eval)) / n_eval


def _success(rng: random.Random, p0: float, sigma: float, n_eval: int) -> float:
    return _binomial_rate(rng, _latent(rng, p0, sigma), n_eval)


def simulate_null(
    spec: TaskSpec, k_seeds: int, n_eval: int, sigma_seed: float,
    n_draws: int = 10_000, rng_seed: int = 20261101, paired: bool = True,
) -> list[float]:
    """Draws of D under H0. ``paired=False`` reproduces the pre-2026-11-06 null."""
    rng = random.Random(rng_seed)
    p0 = blind_ceiling(spec)
    out = []
    for _ in range(n_draws):
        total = 0.0
        for _ in range(k_seeds):
            if paired:
                p = _latent(rng, p0, sigma_seed)
                total += _binomial_rate(rng, p, n_eval) - _binomial_rate(rng, p, n_eval)
            else:
                total += (_success(rng, p0, sigma_seed, n_eval)
                          - _success(rng, p0, sigma_seed, n_eval))
        out.append(total / k_seeds)
    return out


def simulate_arm_null(
    spec: TaskSpec, k_seeds: int, n_eval: int, sigma_seed: float,
    n_draws: int = 10_000, rng_seed: int = 20261101,
) -> list[float]:
    """Draws of (mean over k seeds of S_arm) - B0 for one blind arm under H0."""
    rng = random.Random(rng_seed)
    p0 = blind_ceiling(spec)
    return [
        sum(_success(rng, p0, sigma_seed, n_eval) for _ in range(k_seeds)) / k_seeds - p0
        for _ in range(n_draws)
    ]


def summarize(draws: list[float]) -> NullSummary:
    s = sorted(draws)
    n = len(s)
    return NullSummary(s[n // 2], s[int(0.95 * n)], s[int(0.99 * n)], n)


def empirical_p(observed: float, draws: list[float]) -> float:
    return (sum(d >= observed for d in draws) + 1) / (len(draws) + 1)


if __name__ == "__main__":
    for cond, cs in (("asym", (2, 4, 8)), ("sym", (4,))):
        for c in cs:
            for sig in (0.0, 0.02, 0.05):
                for paired in (True, False):
                    r = summarize(simulate_null(TaskSpec(cond, 16, c), 8, 500, sig,
                                                paired=paired))
                    a = summarize(simulate_arm_null(TaskSpec(cond, 16, c), 8, 500, sig))
                    print(f"{cond} C={c} sigma={sig:.2f} paired={paired} "
                          f"median={r.median:+.4f} P95={r.p95:.4f} P99={r.p99:.4f} "
                          f"arm_P99={a.p99:.4f} N={r.n_draws}")
