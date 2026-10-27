"""Implementation checks for the NH-01 correlated-pattern run (no empirical number)."""

from __future__ import annotations

import numpy as np

from experiments.nh01_hopfield_dream import correlated as cr
from experiments.nh01_hopfield_dream import run as nh


def test_correlated_patterns_overlap_statistics() -> None:
    rng = np.random.default_rng(0)
    xi = nh.correlated_patterns(200, 2000, 0.3, rng)
    assert xi.shape == (200, 2000) and set(np.unique(xi)) == {-1.0, 1.0}
    q = xi @ xi.T / 2000
    off = q[~np.eye(200, dtype=bool)]
    assert abs(off.mean() - 0.09) < 0.01  # pairwise overlap m**2


def test_m_zero_is_uncorrelated() -> None:
    xi = nh.correlated_patterns(200, 2000, 0.0, np.random.default_rng(1))
    off = (xi @ xi.T / 2000)[~np.eye(200, dtype=bool)]
    assert abs(off.mean()) < 0.005


def test_records_carry_level_and_pilot_records_do_not() -> None:
    cfg = dict(cr.cfg_for(0.3), alpha_grid=(0.1,), t_grid=(0.0, 10.0),
               conditions=((0.0, 0.1),), max_probes=4)
    recs = nh.evaluate_task((32, 0, cfg))
    assert recs and all(r["ancestor_m"] == 0.3 for r in recs)
    pilot = nh.evaluate_task((16, 0))
    assert all("ancestor_m" not in r for r in pilot)


def test_c4_trend_labels() -> None:
    key = f"T0_f{cr.cf.FLIPS:g}_N{cr.N}"
    tk = f"{cr.cf.T_PRIMARY:g}"

    def lv(vals: list[float]) -> dict:
        d = {"ratio_per_seed": vals, "ratio_mean_ci": nh.bootstrap_mean_ci(np.array(vals))}
        return {"conditions": {key: {"dream": {tk: d}}}}

    up = {"0": lv([3.0] * 5), "0.15": lv([3.5] * 5), "0.3": lv([4.0, 4.1, 3.9, 4.2, 4.0])}
    assert cr.c4_trend(up)["label"] == "gain_increases"
    down = {"0": lv([3.0] * 5), "0.15": lv([2.5] * 5), "0.3": lv([2.0, 2.1, 1.9, 2.2, 2.0])}
    assert cr.c4_trend(down)["label"] == "gain_decreases"
    flat = {"0": lv([3.0, 3.2, 2.8, 3.1, 2.9]), "0.15": lv([3.0] * 5),
            "0.3": lv([3.1, 2.9, 3.0, 2.8, 3.2])}
    assert cr.c4_trend(flat)["label"] == "inconclusive"
