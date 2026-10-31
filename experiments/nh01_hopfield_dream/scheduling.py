"""Deadline scheduling and completeness checks for the NH-01 size sweep.

Pre-registration: ``docs/milestones/2026-10-30-nh01-scale-sweep.md``
(§7.5). The scheduler starts at most ``max_in_flight`` tasks at a time,
in the fixed submission order, and starts no task once the deadline has
passed on a monotonic clock; tasks not started by then are cancelled
(never run), running tasks finish. Clock, start and wait functions are
injected so that the rule is unit-tested with a simulated clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScheduleLog:
    started: list[tuple[Any, float]] = field(default_factory=list)
    finished: list[tuple[Any, float, Any]] = field(default_factory=list)
    cancelled: list[Any] = field(default_factory=list)
    deadline_hit_at: float | None = None


def run_with_deadline(
    tasks: Sequence[Any],
    start: Callable[[Any], Hashable],
    wait_any: Callable[[set], Iterable[tuple[Hashable, Any]]],
    max_in_flight: int,
    deadline_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> ScheduleLog:
    """Run ``tasks`` in order with at most ``max_in_flight`` running.

    ``start(task)`` launches a task and returns a handle ; ``wait_any(handles)``
    blocks until at least one running handle is done and returns the done
    ``(handle, result)`` pairs. Times in the log are ``clock() - t0``.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    t0 = clock()
    log = ScheduleLog()
    pending = list(tasks)
    running: dict[Hashable, Any] = {}
    while pending or running:
        while pending and len(running) < max_in_flight:
            now = clock() - t0
            if now >= deadline_s:
                if log.deadline_hit_at is None:
                    log.deadline_hit_at = now
                log.cancelled.extend(pending)
                pending.clear()
                break
            task = pending.pop(0)
            running[start(task)] = task
            log.started.append((task, now))
        if not running:
            break
        for handle, result in wait_any(set(running)):
            task = running.pop(handle)
            log.finished.append((task, clock() - t0, result))
    return log


def expected_cell_keys(
    n: int,
    seed: int,
    alpha_grid: Sequence[float],
    t_grid: Sequence[float],
    conditions: Sequence[tuple[float, float]],
    condition_arms: Sequence[Any] | None = None,
) -> set[tuple]:
    """Keys (N, seed, alpha, arm, t, temp, flips) a complete unit must hold."""
    arms = [("hebb", 0.0)] + [(a, float(t)) for t in t_grid[1:] for a in ("dream", "shrink")]
    if condition_arms is None:
        condition_arms = (None,) * len(conditions)
    keys = set()
    for (temp, flips), allowed in zip(conditions, condition_arms, strict=True):
        for alpha in alpha_grid:
            for arm, t in arms:
                if allowed is None or (arm, t) in allowed:
                    keys.add((n, seed, alpha, arm, t, temp, flips))
    return keys


def unit_status(records: Iterable[dict], expected: set[tuple]) -> dict:
    """Completeness of one unit: missing, duplicated and unexpected keys."""
    seen: dict[tuple, int] = {}
    for r in records:
        k = (r["N"], r["seed"], r["alpha"], r["arm"], float(r["t"]), r["temp"], r["flips"])
        seen[k] = seen.get(k, 0) + 1
    missing = sorted(expected - set(seen))
    duplicated = sorted(k for k, c in seen.items() if c > 1)
    unexpected = sorted(set(seen) - expected)
    return {
        "complete": not (missing or duplicated or unexpected),
        "n_expected": len(expected),
        "missing": missing,
        "duplicated": duplicated,
        "unexpected": unexpected,
    }
