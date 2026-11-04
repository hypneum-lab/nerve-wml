"""Checks of the pre-registered inference for P1bis-ext and P8c-nocap.

No empirical number: synthetic inputs only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.hopfield_dream_conf import analyse, stats


def test_signflip_all_positive_is_two_to_minus_n():
    assert stats.signflip_p_greater([0.1] * 5) == pytest.approx(1 / 32)
    assert stats.signflip_p_greater([0.3] * 20) == pytest.approx(2.0**-20)


def test_signflip_symmetric_sample_is_not_significant():
    p = stats.signflip_p_greater([-1.0, 1.0, -2.0, 2.0])
    assert p > 0.5


def test_signflip_null_shift():
    d = [0.11, 0.12, 0.13, 0.14, 0.15]
    assert stats.signflip_p_greater(d, 0.10) == pytest.approx(1 / 32)
    assert stats.signflip_p_greater(d, 0.20) == pytest.approx(1.0)


def test_holm_known_values():
    adj = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_family_rule_labels():
    rng = np.random.default_rng(0)
    diffs = {
        "big": list(0.3 + 0.01 * rng.standard_normal(20)),
        "zero": list(0.0 + 0.01 * rng.standard_normal(20)),
        "noisy": list(0.1 + 0.5 * rng.standard_normal(20)),
        "missing": [],
    }
    out = stats.family_decisions(diffs, 0.0, 0.10, 0.10)
    assert out["big"]["label"] == "success"
    assert out["zero"]["label"] == "below"
    assert out["noisy"]["label"] == "inconclusive"
    assert out["missing"]["label"] == "not_measured"


def test_success_requires_point_estimate_threshold():
    d = [0.05] * 20      # significant > 0 but below the 0.10 minimum effect
    out = stats.family_decisions({"k": d}, 0.0, 0.10, 0.10)
    assert out["k"]["label"] == "below"


def _p8c_seed(kernel_a, kernel_b, grid_a, grid_b, match_a=None, match_b=None):
    tab = {"hebb": None, "kernel": {}, "grid": {}, "matched": {}}
    for t, a, b in zip(analyse.P8_T, kernel_a, kernel_b, strict=True):
        tab["kernel"][analyse._tkey(t)] = {"rec_q0.1_A": a, "rec_q0.1_B": b}
    for r, a, b in zip(analyse.P8_R, grid_a, grid_b, strict=True):
        tab["grid"][round(r, 6)] = {"rec_q0.1_A": a, "rec_q0.1_B": b}
    if match_a is not None:
        for t, a, b in zip(analyse.P8_T, match_a, match_b, strict=True):
            tab["matched"][analyse._tkey(t)] = {"rec_q0.1_A": a, "rec_q0.1_B": b}
    return tab


def test_p8c_cross_fitted_selection_and_ties():
    ka = [0, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]       # tie from t = 2 -> pick t = 2
    kb = [0, 0, 0.4, 0.9, 1.0, 1.0, 1.0, 0.8]
    ga = [0, 0, 0, 0, 0.2, 0.9, 0.9, 0.5]            # tie r = 0.7 / 0.8 -> pick 0.7
    gb = [0, 0, 0, 0, 0.3, 0.95, 1.0, 0.4]
    best, match = analyse.p8c_contrasts(_p8c_seed(ka, kb, ga, gb), 0.1)
    assert best["t_hat"] == 2.0 and best["r_hat"] == 0.7
    assert best["delta"] == pytest.approx(0.9 - 0.95)
    assert match is None


def test_p8c_matched_contrast():
    ka = [0, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]
    kb = [0, 0, 0.4, 0.9, 1.0, 1.0, 1.0, 0.8]
    ma = [0, 0, 0.0, 0.1, 0.9, 1.0, 1.0, 1.0]        # largest A-gap at t = 2 (0.9)
    mb = [0, 0, 0.0, 0.2, 0.9, 1.0, 1.0, 1.0]
    best, match = analyse.p8c_contrasts(_p8c_seed(ka, kb, ka, kb, ma, mb), 0.1)
    assert match["t_tilde"] == 2.0
    assert match["delta"] == pytest.approx(0.9 - 0.2)


def test_p8c_incomplete_grid_is_not_measured():
    ka = kb = [1.0] * 8
    tab = _p8c_seed(ka, kb, [0.5] * 8, [0.5] * 8)
    del tab["grid"][0.9]
    best, _ = analyse.p8c_contrasts(tab, 0.1)
    assert best is None


def test_p8c_t_grid_includes_infinity_last():
    assert analyse.P8_T[-1] == "inf" and analyse._tkey(math.inf) == "inf"


def _fake_p8c_rows(kernel_rec, stoch_rec, hebb_rec=0.0):
    """Synthetic rows: constant recoveries for every cue level and set."""
    rows = []
    qs = analyse.P8_Q_PRIMARY + analyse.P8_Q_SECONDARY
    for n in analyse.P8_NS:
        for a in analyse.P8_ALPHAS:
            for s in range(32001, 32021):
                def rec(v):
                    return {f"rec_q{q:.1f}_{x}": v for q in qs for x in "AB"}
                common = {"N": n, "alpha": a, "seed": s}
                rows.append(dict(common, arm="hebb", t=0.0, **rec(hebb_rec)))
                for t in analyse.P8_T:
                    rows.append(dict(common, arm="kernel", t=t, **rec(kernel_rec)))
                    rows.append(dict(common, arm="stoch", kind="matched", t=t, r=0.5, gap=0.0,
                                     **rec(stoch_rec)))
                for r in analyse.P8_R:
                    rows.append(dict(common, arm="stoch", kind="grid", t=None, r=r, gap=0.0,
                                     **rec(stoch_rec)))
    return rows


def _write(tmp_path, rows):
    import json
    d = tmp_path / "n4096"
    d.mkdir()
    with open(d / "cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path)


def test_p8c_end_to_end_labels(tmp_path):
    res = analyse.analyse_p8c(_write(tmp_path, _fake_p8c_rows(1.0, 0.5)))
    assert res["gates"]["pass"]
    assert res["overall_primary"].startswith("supported_in_cells:")


def test_p8c_end_to_end_equal_arms_not_supported(tmp_path):
    res = analyse.analyse_p8c(_write(tmp_path, _fake_p8c_rows(1.0, 1.0)))
    assert res["overall_primary"] == "not_supported"


def test_p8c_gate_fails_when_kernel_does_not_recover(tmp_path):
    res = analyse.analyse_p8c(_write(tmp_path, _fake_p8c_rows(0.0, 0.0)))
    assert res["overall_primary"] == "void_gate_failed"
