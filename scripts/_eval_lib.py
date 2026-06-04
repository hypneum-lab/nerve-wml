"""Shared helpers for extended evaluations.

Factored from extended_eval.py and extended_eval_hp.py so that the v3
massive-seed driver and any future driver can reuse them. Keep the API
stable: change here breaks downstream drivers.
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any

from nerve_wml.methodology.paired_tests import bootstrap_ci, wilcoxon_paired

# Public types
RunnerKwargs = dict[str, Any]
PerMethodMetrics = dict[str, dict[str, float]]
Runner = Callable[..., PerMethodMetrics]


def _runner_call(args: tuple) -> PerMethodMetrics:
    """Top-level for multiprocessing pickling: (runner, seed, kwargs)."""
    runner, seed, kwargs = args
    return runner(seed=seed, **kwargs)


def run_seeds_parallel(
    runner: Runner,
    seeds: Sequence[int],
    *,
    workers: int | None = None,
    **runner_kwargs: Any,
) -> list[PerMethodMetrics]:
    """Run `runner(seed=s, **runner_kwargs)` across `seeds`, in parallel.

    `workers` defaults to `min(len(seeds), cpu_count())`. Returns the per-seed
    list in seed order (deterministic mapping).
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if workers is None:
        workers = min(len(seeds), mp.cpu_count())
    if workers <= 1:
        return [runner(seed=int(s), **runner_kwargs) for s in seeds]
    args = [(runner, int(s), runner_kwargs) for s in seeds]
    with mp.Pool(processes=workers) as pool:
        return pool.map(_runner_call, args)


def aggregate(
    per_seed: list[PerMethodMetrics],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-seed runs into per-method-per-metric stats with CI.

    Returns {method: {metric: {"values": [...], "mean": .., "std": ..,
                                 "ci95_low": .., "ci95_high": ..}}}.
    """
    if not per_seed:
        raise ValueError("per_seed must be non-empty")
    methods = set(per_seed[0])
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        out[method] = {}
        metrics = set(per_seed[0][method])
        for metric in metrics:
            values = [float(run[method][metric]) for run in per_seed]
            n = len(values)
            mean = sum(values) / n
            std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5 if n > 1 else 0.0
            ci = bootstrap_ci(values, n_resamples=2000, seed=0)
            out[method][metric] = {
                "values": values,
                "mean": mean,
                "std": std,
                "ci95_low": ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
                "n": n,
            }
    return out


def paired_test_vs_reference(
    aggregated: dict[str, dict[str, dict[str, Any]]],
    *,
    reference_method: str,
    metric: str = "mi_bits",
    other_methods: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    """For each other method, paired-Wilcoxon `metric` vs the reference method."""
    if reference_method not in aggregated:
        raise ValueError(f"reference {reference_method!r} not in aggregated")
    ref_values = aggregated[reference_method][metric]["values"]
    others = other_methods or [m for m in aggregated if m != reference_method]
    return {
        m: wilcoxon_paired(ref_values, aggregated[m][metric]["values"])
        for m in others
    }


def format_metric_table(
    aggregated: dict[str, dict[str, dict[str, Any]]],
    *,
    metric: str = "mi_bits",
) -> str:
    """Markdown table of one metric across methods."""
    lines = [
        "| method | mean ± std | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for method, metrics in aggregated.items():
        s = metrics[metric]
        lines.append(
            f"| {method} | {s['mean']:.4f} ± {s['std']:.4f} | "
            f"[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}] | {s['n']} |"
        )
    return "\n".join(lines)
