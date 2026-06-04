"""Equal-tuning protocol: every tunable method gets the SAME budget.

Budget = ``budget`` trials drawn deterministically from a fixed grid
per method; each trial runs ``len(seeds_per_trial)`` seeds; best
trial is selected by mean MI across seeds. The non-tunable Procrustes
arm is reported once at the same multi-seed budget for parity.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from nerve_wml.methodology.paired_tests import bootstrap_ci
from scripts.multi_seed import run_multi_seed
from scripts.transducer_baselines_pilot import (
    _build_task,
    _mi_entropy_bits,
    _train_learned,
    run_transducer_benchmark,
)

LEARNED_GRID: tuple[dict[str, Any], ...] = tuple(
    {"steps": s, "lr": lr}
    for s, lr in itertools.product((500, 2000, 8000), (1e-3, 3e-3, 1e-2))
)[:8]

VEC2VEC_GRID: tuple[dict[str, Any], ...] = tuple(
    {"lambda_cycle": lc, "steps": s}
    for lc, s in itertools.product(
        (1.0, 10.0, 50.0, 100.0), (500, 2000)
    )
)[:8]

RELREP_GRID: tuple[dict[str, Any], ...] = tuple(
    {"n_anchors": k} for k in (4, 8, 16, 32, 48, 64, 96, 128)
)[:8]


def _learned_runner(
    *, seed: int, steps: int, lr: float,
) -> dict[str, dict[str, float]]:
    src_codes, dst_codes, *_ = _build_task(seed)
    learned = _train_learned(src_codes, dst_codes, steps, lr=lr)
    pred = learned.forward(src_codes, hard=True)
    row = _mi_entropy_bits(pred, dst_codes)
    return {"learned": {"mi_bits": float(row["mi_bits"])}}


def _vec2vec_runner(
    *, seed: int, lambda_cycle: float, steps: int,
) -> dict[str, dict[str, float]]:
    res = run_transducer_benchmark(
        steps=steps, seed=seed, lambda_cycle=lambda_cycle,
    )
    return {"vec2vec": {"mi_bits": float(res["vec2vec"]["mi_bits"])}}


def _relrep_runner(
    *, seed: int, n_anchors: int,
) -> dict[str, dict[str, float]]:
    res = run_transducer_benchmark(
        steps=500, seed=seed, n_anchors=n_anchors,
    )
    return {"relrep": {"mi_bits": float(res["relative_rep"]["mi_bits"])}}


def _procrustes_runner(*, seed: int) -> dict[str, dict[str, float]]:
    res = run_transducer_benchmark(steps=500, seed=seed)
    return {"procrustes": {"mi_bits": float(res["procrustes"]["mi_bits"])}}


def _eval_trial(
    runner: Any,
    method: str,
    params: dict[str, Any],
    seeds_per_trial: Sequence[int],
) -> dict[str, Any]:
    agg = run_multi_seed(runner, seeds=seeds_per_trial, **params)
    values = agg[method]["mi_bits"]["values"]
    ci = bootstrap_ci(values, n_resamples=500, seed=0)
    return {
        "params":       params,
        "mi_values":    values,
        "mi_mean":      ci["mean"],
        "mi_ci95_low":  ci["ci95_low"],
        "mi_ci95_high": ci["ci95_high"],
    }


def _best_of(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return max(trials, key=lambda t: t["mi_mean"])


def run_equal_tuning(
    *,
    budget: int = 8,
    seeds_per_trial: tuple[int, ...] = (0, 1, 2),
) -> dict[str, dict[str, Any]]:
    """Best-of-budget MI per method under an equal trial budget."""
    out: dict[str, dict[str, Any]] = {}

    learned_trials = [
        _eval_trial(_learned_runner, "learned", p, seeds_per_trial)
        for p in LEARNED_GRID[:budget]
    ]
    out["learned"] = {"trials": learned_trials, "best": _best_of(learned_trials)}

    vec2vec_trials = [
        _eval_trial(_vec2vec_runner, "vec2vec", p, seeds_per_trial)
        for p in VEC2VEC_GRID[:budget]
    ]
    out["vec2vec"] = {"trials": vec2vec_trials, "best": _best_of(vec2vec_trials)}

    relrep_trials = [
        _eval_trial(_relrep_runner, "relrep", p, seeds_per_trial)
        for p in RELREP_GRID[:budget]
    ]
    out["relrep"] = {"trials": relrep_trials, "best": _best_of(relrep_trials)}

    proc_trial = _eval_trial(
        _procrustes_runner, "procrustes", {}, seeds_per_trial,
    )
    out["procrustes"] = {
        "trials": [proc_trial],
        "best":   proc_trial,
        "note":   "closed-form, non-tunable; reported at parity budget",
    }
    return out


def main() -> None:  # pragma: no cover
    out = run_equal_tuning()
    for method, payload in out.items():
        best = payload["best"]
        print(
            f"{method:<12}  best mi={best['mi_mean']:.4f}  "
            f"CI=[{best['mi_ci95_low']:.4f}, {best['mi_ci95_high']:.4f}]  "
            f"params={best['params']}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
