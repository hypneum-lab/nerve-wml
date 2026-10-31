"""Implementation checks for the NH-01 size-sweep options of run.py.

No empirical number: tiny N, the tests only check that the size-sweep
configuration keys keep the cells of earlier configurations bit-identical
(per-load split, appended conditions with restricted arms) and that every
temperature of the sweep has a retrieval threshold.
"""

from __future__ import annotations

import json

from experiments.nh01_hopfield_dream import run as nh

N = 24
SEED = 2
BASE = {
    "alpha_grid": (0.1, 0.2, 0.3),
    "t_grid": (0.0, 1.0, 10.0),
    "conditions": ((0.0, 0.1), (0.2, 0.1)),
    "max_probes": 4,
    "spectral": False,
}
KEY = ("N", "seed", "alpha", "arm", "t", "temp", "flips")


def _canon(recs: list[dict]) -> dict[tuple, str]:
    out = {}
    for r in recs:
        k = tuple(r[f] for f in KEY)
        assert k not in out
        out[k] = json.dumps({f: v for f, v in r.items() if f != "task_seconds"}, sort_keys=True)
    return out


def test_thresholds_cover_sweep_temperatures() -> None:
    assert nh.RETRIEVED_THRESH[0.0] == 0.9
    for temp in (0.1, 0.2, 0.3):
        assert nh.RETRIEVED_THRESH[temp] == 0.8


def test_defaults_unchanged_by_new_keys() -> None:
    full = nh.evaluate_task((N, SEED, BASE))
    explicit = nh.evaluate_task((N, SEED, dict(
        BASE,
        alpha_indices=(0, 1, 2),
        p_max=round(0.3 * N),
        condition_arms=(None, None),
    )))
    assert _canon(full) == _canon(explicit)


def test_per_load_split_is_bit_identical() -> None:
    full = _canon(nh.evaluate_task((N, SEED, BASE)))
    p_max = round(BASE["alpha_grid"][-1] * N)
    split: dict[tuple, str] = {}
    for ai, alpha in enumerate(BASE["alpha_grid"]):
        cfg = dict(BASE, alpha_grid=(alpha,), alpha_indices=(ai,), p_max=p_max)
        split.update(_canon(nh.evaluate_task((N, SEED, cfg))))
    assert split == full


def test_per_load_split_correlated_is_bit_identical() -> None:
    cfg0 = dict(BASE, ancestor_m=0.3)
    full = _canon(nh.evaluate_task((N, SEED, cfg0)))
    p_max = round(BASE["alpha_grid"][-1] * N)
    split: dict[tuple, str] = {}
    for ai, alpha in enumerate(BASE["alpha_grid"]):
        cfg = dict(cfg0, alpha_grid=(alpha,), alpha_indices=(ai,), p_max=p_max)
        split.update(_canon(nh.evaluate_task((N, SEED, cfg))))
    assert split == full


def test_naive_split_differs() -> None:
    """Without the global index the streams change: the option is needed."""
    full = _canon(nh.evaluate_task((N, SEED, BASE)))
    naive = _canon(nh.evaluate_task((N, SEED, dict(BASE, alpha_grid=(0.3,)))))
    assert any(full[k] != v for k, v in naive.items())


def test_appended_restricted_conditions_keep_old_cells() -> None:
    old = _canon(nh.evaluate_task((N, SEED, BASE)))
    three = {("hebb", 0.0), ("dream", 10.0), ("shrink", 10.0)}
    cfg = dict(
        BASE,
        conditions=BASE["conditions"] + ((0.1, 0.1), (0.3, 0.1)),
        condition_arms=(None, None, three, three),
    )
    new = _canon(nh.evaluate_task((N, SEED, cfg)))
    for k, v in old.items():
        assert new[k] == v
    extra = [k for k in new if k not in old]
    assert len(extra) == len(BASE["alpha_grid"]) * 3 * 2
    assert {(k[3], k[4]) for k in extra} == three
    assert {k[5] for k in extra} == {0.1, 0.3}
