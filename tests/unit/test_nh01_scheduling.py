"""Deadline scheduler and completeness checks, with a simulated clock."""

from __future__ import annotations

import pytest

from experiments.nh01_hopfield_dream import run as nh
from experiments.nh01_hopfield_dream import scheduling as sc


class FakePool:
    """Tasks are durations; time advances only when waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.ends: dict[int, float] = {}
        self.next = 0

    def clock(self) -> float:
        return self.now

    def start(self, duration: float) -> int:
        h = self.next
        self.next += 1
        self.ends[h] = self.now + duration
        return h

    def wait_any(self, handles: set) -> list[tuple[int, str]]:
        t = min(self.ends[h] for h in handles)
        self.now = t
        done = [h for h in handles if self.ends[h] <= t]
        return [(h, "ok") for h in done]


def test_no_start_after_deadline_and_queue_cancelled() -> None:
    pool = FakePool()
    tasks = [10.0] * 10  # 10 tasks of 10 s, 2 slots, deadline 25 s
    log = sc.run_with_deadline(tasks, pool.start, pool.wait_any, 2, 25.0, pool.clock)
    starts = [t for _, t in log.started]
    assert starts == [0.0, 0.0, 10.0, 10.0, 20.0, 20.0]
    assert all(t < 25.0 for t in starts)
    assert len(log.cancelled) == 4
    assert len(log.finished) == 6
    # overshoot bounded by the residual duration of the tasks running at the deadline
    assert max(t for _, t, _ in log.finished) == 30.0
    assert log.deadline_hit_at == 30.0


def test_in_flight_bound_and_order() -> None:
    pool = FakePool()
    tasks = [5.0, 1.0, 1.0, 1.0]
    log = sc.run_with_deadline(tasks, pool.start, pool.wait_any, 2, 1e9, pool.clock)
    assert [task for task, _ in log.started] == tasks
    assert log.cancelled == [] and log.deadline_hit_at is None
    # never more than 2 running: the 3rd task starts when the 2nd ends (t = 1)
    assert [t for _, t in log.started] == [0.0, 0.0, 1.0, 2.0]


def test_deadline_zero_starts_nothing() -> None:
    pool = FakePool()
    log = sc.run_with_deadline([1.0, 1.0], pool.start, pool.wait_any, 4, 0.0, pool.clock)
    assert log.started == [] and log.cancelled == [1.0, 1.0]


def test_completeness_detects_missing_duplicated_unexpected() -> None:
    cfg = {
        "alpha_grid": (0.1, 0.2),
        "t_grid": (0.0, 10.0),
        "conditions": ((0.0, 0.1), (0.1, 0.1)),
        "max_probes": 4,
        "spectral": False,
        "condition_arms": (None, {("hebb", 0.0), ("shrink", 10.0)}),
    }
    recs = nh.evaluate_task((16, 0, cfg))
    exp = sc.expected_cell_keys(16, 0, cfg["alpha_grid"], cfg["t_grid"],
                                cfg["conditions"], cfg["condition_arms"])
    assert len(exp) == 2 * 3 + 2 * 2
    assert sc.unit_status(recs, exp)["complete"]
    assert not sc.unit_status(recs[1:], exp)["complete"]
    assert sc.unit_status(recs[1:], exp)["missing"]
    assert sc.unit_status(recs + recs[:1], exp)["duplicated"]
    other = dict(recs[0], alpha=0.3)
    assert sc.unit_status(recs + [other], exp)["unexpected"]


def test_p_max_sets_the_pattern_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes: list[tuple] = []
    real = nh.stream

    class Spy:
        def __init__(self, g, ids):
            self.g, self.ids = g, ids

        def random(self, size=None):
            if self.ids[:1] == (1,):
                shapes.append(size if isinstance(size, tuple) else (size,))
            return self.g.random(size)

        def __getattr__(self, name):
            return getattr(self.g, name)

    monkeypatch.setattr(nh, "stream", lambda seed, n, *ids: Spy(real(seed, n, *ids), ids))
    base = {"alpha_grid": (0.1,), "t_grid": (0.0, 10.0), "conditions": ((0.0, 0.1),),
            "max_probes": 2, "spectral": False}
    nh.evaluate_task((20, 0, dict(base, p_max=15)))
    assert shapes == [(15, 20)]
    shapes.clear()
    nh.evaluate_task((20, 0, base))
    assert shapes == [(2, 20)]
    shapes.clear()
    nh.evaluate_task((20, 0, dict(base, ancestor_m=0.3, p_max=15)))
    assert (15, 20) in shapes


def test_load_above_p_max_is_rejected() -> None:
    cfg = {"alpha_grid": (0.5,), "t_grid": (0.0, 10.0), "conditions": ((0.0, 0.1),),
           "max_probes": 2, "spectral": False, "p_max": 3}
    with pytest.raises(ValueError, match="p_max"):
        nh.evaluate_task((20, 0, cfg))
