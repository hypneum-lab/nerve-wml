"""Pre-data calibrated null for NH-96 (stdlib only, seconds on one core).

Statistic: D = mean over k seeds of (S_eng - S_nocomm), S = team success.
Under H0 (the message carries no usable information) both arms sit at the
blind ceiling; spread comes from binomial evaluation noise and from an
arm-by-seed random effect of sd ``sigma_seed`` (TBD: measured on nocomm
development seeds before the freeze, never on registered seeds).
Same pipeline: D is computed as in the protocol, N >= 1000 draws,
empirical p = (k+1)/(N+1).
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


def _success(rng: random.Random, p0: float, sigma: float, n_eval: int) -> float:
    p = min(1.0, max(0.0, p0 + rng.gauss(0.0, sigma)))
    return sum(rng.random() < p for _ in range(n_eval)) / n_eval


def simulate_null(
    spec: TaskSpec, k_seeds: int, n_eval: int, sigma_seed: float,
    n_draws: int = 10_000, rng_seed: int = 20261101,
) -> list[float]:
    rng = random.Random(rng_seed)
    p0 = blind_ceiling(spec)
    out = []
    for _ in range(n_draws):
        d = sum(
            _success(rng, p0, sigma_seed, n_eval) - _success(rng, p0, sigma_seed, n_eval)
            for _ in range(k_seeds)
        ) / k_seeds
        out.append(d)
    return out


def summarize(draws: list[float]) -> NullSummary:
    s = sorted(draws)
    n = len(s)
    return NullSummary(s[n // 2], s[int(0.95 * n)], s[int(0.99 * n)], n)


def empirical_p(observed: float, draws: list[float]) -> float:
    return (sum(d >= observed for d in draws) + 1) / (len(draws) + 1)


if __name__ == "__main__":
    for c in (2, 4, 8):
        for sig in (0.0, 0.02, 0.05):
            r = summarize(simulate_null(TaskSpec("asym", 16, c), 8, 500, sig))
            print(f"C={c} sigma={sig:.2f} median={r.median:+.3f} "
                  f"P95={r.p95:.3f} P99={r.p99:.3f} N={r.n_draws}")
