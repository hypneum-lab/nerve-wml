"""Multi-seed wrapper for Plan A pilot runners.

Calls a runner callable across multiple seeds and aggregates each
leaf float metric into ``{"values": [...], "mean": float, "std": float}``,
preserving the runner's outer ``{method: {metric: ...}}`` shape.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

Runner = Callable[..., Mapping[str, Mapping[str, float]]]


def _aggregate_leaves(
    per_seed: list[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not per_seed:
        raise ValueError("at least one seed required")
    first = per_seed[0]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for method, metrics in first.items():
        out[method] = {}
        for metric in metrics:
            values = [float(run[method][metric]) for run in per_seed]
            mean = sum(values) / len(values)
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(var)
            else:
                std = 0.0
            out[method][metric] = {
                "values": values,
                "mean":   mean,
                "std":    std,
            }
    return out


def run_multi_seed(
    runner:  Runner,
    *,
    seeds:   Sequence[int],
    **kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Call ``runner(seed=s, **kwargs)`` for each seed and aggregate.

    Args:
        runner: A callable returning ``{method: {metric: float}}``.
        seeds:  Sequence of integer seeds.
        kwargs: Forwarded to every runner call.

    Returns:
        ``{method: {metric: {"values": [...], "mean": .., "std": ..}}}``.
    """
    per_seed = [runner(seed=int(s), **kwargs) for s in seeds]
    return _aggregate_leaves(per_seed)


def summarize(
    aggregated: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    """Pretty-print the aggregated table."""
    lines = ["method                  metric              mean       std"]
    lines.append("-" * 60)
    for method, metrics in aggregated.items():
        for metric, stats in metrics.items():
            lines.append(
                f"{method:<22}  {metric:<18}  "
                f"{stats['mean']:>8.4f}  {stats['std']:>8.4f}"
            )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    raise SystemExit(
        "multi_seed is a library; call run_multi_seed from a pilot."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
