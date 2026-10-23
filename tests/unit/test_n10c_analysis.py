"""Tests for the frozen N10-C estimation analysis (numpy/scipy only)."""

import numpy as np
import pytest
from scipy import stats

from experiments.procgen_multiagent_gtm_comm.analysis import (
    fieller_ci,
    one_sample_ci,
    pilot_category,
    primary_report,
    welch_ci,
)

T4 = 2.7764451052  # t(0.975, df=4)


def test_welch_ci_matches_scipy():
    x = np.array([0.62, 0.70, 0.55, 0.81, 0.66])
    y = np.array([0.40, 0.52, 0.47, 0.35, 0.58])
    for level in (0.95, 1 - 0.05 / 3, 1 - 0.05 / 9):
        d, lo, hi = welch_ci(x, y, level)
        ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval(level)
        assert d == pytest.approx(x.mean() - y.mean())
        assert lo == pytest.approx(ci.low, abs=1e-9)
        assert hi == pytest.approx(ci.high, abs=1e-9)


def test_welch_ci_hand_computed():
    # v1 = 2.5/5 = 0.5, v2 = 0 -> df = 0.25 / (0.25 / 4) = 4
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.zeros(5)
    d, lo, hi = welch_ci(x, y, 0.95)
    half = T4 * np.sqrt(0.5)
    assert d == pytest.approx(3.0)
    assert lo == pytest.approx(3.0 - half, abs=1e-6)
    assert hi == pytest.approx(3.0 + half, abs=1e-6)
    assert half == pytest.approx(1.96322, abs=1e-4)


def test_zero_variance_returns_none():
    assert welch_ci(np.full(5, 0.3), np.full(5, 0.3), 0.95) is None
    assert welch_ci(np.zeros(5), np.zeros(5), 0.95) is None
    assert one_sample_ci(np.full(5, 0.2), 0.95) is None


def test_one_sample_ci_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # s^2 = 2.5, s/sqrt5 = sqrt(.5)
    m, lo, hi = one_sample_ci(x, 0.95)
    half = T4 * np.sqrt(0.5)
    assert m == pytest.approx(3.0)
    assert lo == pytest.approx(3.0 - half, abs=1e-6)
    assert hi == pytest.approx(3.0 + half, abs=1e-6)


def test_pilot_category_precedence():
    d = 0.10
    ok = (0.2, 0.15, 0.25)
    # any unavailable interval wins over everything else
    assert pilot_category([ok, None, (0.0, -0.1, 0.05)], d) == "pilot-interval-unavailable"
    # U < delta for one baseline -> excludes
    assert pilot_category([ok, ok, (0.0, -0.1, 0.05)], d) == "pilot-excludes-SESOI"
    # all L >= delta -> positive (L == delta counts)
    assert pilot_category([ok, (0.3, 0.10, 0.5), ok], d) == "pilot-positive"
    # otherwise inconclusive
    assert pilot_category([ok, ok, (0.1, 0.05, 0.15)], d) == "pilot-inconclusive"


def test_fieller_hand_computed():
    nocomm = np.zeros(5)
    eng = np.array([0.8, 0.9, 1.0, 1.1, 1.2])  # mean 1, v = 0.005
    arm = np.array([0.4, 0.45, 0.5, 0.55, 0.6])  # mean .5, v = 0.00125
    # Cov = v_NOCOMM = 0; a=.5, b=1; t^2 = 7.70865 (df 4 both)
    # A = 1 - t^2*.005 = .961457 ; C = .25 - t^2*.00125 = .240364
    # disc = .25 - A*C = .0189 ; roots = (.5 -+ sqrt(disc)) / A
    rho, lo, hi = fieller_ci(arm, nocomm, eng)
    assert rho == pytest.approx(0.5)
    assert lo == pytest.approx(0.37706, abs=2e-3)
    assert hi == pytest.approx(0.66306, abs=2e-3)


def test_fieller_endpoints_satisfy_inequality_boundary():
    rng = np.random.default_rng(3)
    nocomm = rng.normal(0.1, 0.02, 5)
    eng = rng.normal(0.8, 0.03, 5)
    arm = rng.normal(0.5, 0.03, 5)
    rho, lo, hi = fieller_ci(arm, nocomm, eng)
    a = arm.mean() - nocomm.mean()
    b = eng.mean() - nocomm.mean()
    vn, va, ve = (np.var(z, ddof=1) / 5 for z in (nocomm, arm, eng))
    var_a, var_b = va + vn, ve + vn
    # smaller Welch df of the two differences
    df_a = (va + vn) ** 2 / (va**2 / 4 + vn**2 / 4)
    df_b = (ve + vn) ** 2 / (ve**2 / 4 + vn**2 / 4)
    t = stats.t.ppf(0.975, min(df_a, df_b))
    for r in (lo, hi):
        lhs = (a - r * b) ** 2
        rhs = t**2 * (var_a - 2 * r * vn + r**2 * var_b)
        assert lhs == pytest.approx(rhs, rel=1e-9)
    assert lo < rho < hi


def test_fieller_none_when_denominator_crosses_zero():
    nocomm = np.array([0.40, 0.50, 0.45, 0.55, 0.50])
    eng = np.array([0.45, 0.55, 0.50, 0.60, 0.40])  # b ~ 0.0, wide
    arm = np.array([0.5, 0.6, 0.55, 0.65, 0.5])
    assert fieller_ci(arm, nocomm, eng) is None


def test_fieller_none_when_variances_degenerate():
    z = np.zeros(5)
    assert fieller_ci(np.full(5, 0.5), z, np.ones(5)) is None


# ---------------------------------------------------------------- report
def _cell(r, t, bw, s):
    t = np.asarray(t, float)
    return {
        "reached_threshold": t < 50e6,
        "R": np.array(r, float),
        "time_to_half": np.array(t, float),
        "bandwidth_bits": np.array(bw, float),
        "S": np.array(s, float),
    }


def _fixture(eng_closed_s=(0.30, 0.34, 0.28, 0.32, 0.36)):
    cap = 50e6
    res = {}
    base = [-0.02, 0.01, 0.0, 0.02, -0.01]  # zero-mean jitter
    j = np.array(base)

    def mk(r, t, bw, s):
        return _cell(r + j, np.minimum(t + 1e6 * j * 10, cap), bw + j, s + j * 0.1)

    for cond in ("registered", "closed"):
        res[("nocomm", cond)] = _cell(0.10 + j, [cap] * 5, [0.0] * 5, [0.0] * 5)
        res[("eng", cond)] = mk(0.80, np.full(5, 20e6), np.full(5, 6.0), np.full(5, 0.3))
        res[("gtm", cond)] = mk(0.60, np.full(5, 30e6), np.full(5, 3.0), np.full(5, 0.2))
        res[("dial", cond)] = mk(0.30, np.full(5, 45e6), np.full(5, 5.0), np.full(5, 0.1))
    res[("eng", "closed")]["S"] = np.array(eng_closed_s, float)
    for arm, r, t, bw in (("mlp", 0.20, 48e6, 4.0), ("atoc", 0.40, 40e6, 7.0)):
        res[(arm, "registered")] = mk(r, np.full(5, t), np.full(5, bw), np.full(5, 0.05))
    return res


def test_primary_report_end_to_end():
    rep = primary_report(_fixture())
    p = rep["primary"]
    assert p["level"] == pytest.approx(1 - 0.05 / 3)
    assert p["delta"] == 0.10 and p["range"] == 1.0
    assert set(p["contrasts"]) == {"mlp", "atoc", "dial"}
    j = np.array([-0.02, 0.01, 0.0, 0.02, -0.01])
    # theta_(gtm,dial) = 0.60 - 0.30 with identical jitter -> zero diff variance
    # not both-zero (each has variance), so interval available
    c = p["contrasts"]["dial"]
    assert c["estimate"] == pytest.approx(0.30)
    ref = welch_ci(0.60 + j, 0.30 + j, 1 - 0.05 / 3)
    assert (c["low"], c["high"]) == pytest.approx(ref[1:])
    assert p["category"] == "pilot-positive" or p["category"] in (
        "pilot-inconclusive",
        "pilot-excludes-SESOI",
    )
    # GTM beats atoc by only .2 ; all L >= .10 given tiny variance -> positive
    assert p["category"] == "pilot-positive"

    s = rep["secondary"]
    assert s["level"] == pytest.approx(1 - 0.05 / 9)
    assert len(s["contrasts"]) == 9
    keys = {(c["baseline"], c["metric"]) for c in s["contrasts"]}
    assert keys == {(b, m) for b in ("mlp", "atoc", "dial")
                    for m in ("R", "time_to_half", "bandwidth_bits")}
    by = {(c["baseline"], c["metric"]): c for c in s["contrasts"]}
    # sample efficiency = comparator - GTM : dial 45M - 30M (jitter shifts equal)
    assert by[("dial", "time_to_half")]["estimate"] == pytest.approx(15e6, rel=1e-6)
    # bandwidth signed: GTM - comparator = 3 - 5
    assert by[("dial", "bandwidth_bits")]["estimate"] == pytest.approx(-2.0)
    assert by[("atoc", "R")]["estimate"] == pytest.approx(0.20)
    counts = s["counts"]
    assert counts["above_0"] + counts["below_0"] + counts["includes_0"] + counts["unavailable"] == 9
    assert not any("favour" in k or "label" in k for k in counts)
    assert "favours_gtm" not in by[("dial", "bandwidth_bits")]

    ni = rep["normalized_improvement"]
    assert ni["registered"]["eng"] == {"estimate": 1.0, "fixed": True}
    assert ni["registered"]["nocomm"] == {"estimate": 0.0, "fixed": True}
    g = ni["registered"]["gtm"]
    assert g["estimate"] == pytest.approx((0.60 - 0.10) / (0.80 - 0.10))
    assert g["low"] is not None and g["low"] < g["estimate"] < g["high"]
    assert set(ni["closed"]) == {"eng", "nocomm", "gtm", "dial"}


def test_ni_unavailable_when_denominator_not_above_zero():
    res = _fixture()
    res[("eng", "closed")]["R"] = res[("nocomm", "closed")]["R"] + np.array(
        [0.3, -0.3, 0.2, -0.1, 0.0]
    )
    rep = primary_report(res)
    g = rep["normalized_improvement"]["closed"]["gtm"]
    assert g["low"] is None and g["high"] is None
    assert g["estimate"] == pytest.approx(
        (res[("gtm", "closed")]["R"].mean() - res[("nocomm", "closed")]["R"].mean())
        / (res[("eng", "closed")]["R"].mean() - res[("nocomm", "closed")]["R"].mean())
    )


def test_positive_control_sensitive_allows_s_interpretation():
    rep = primary_report(_fixture())
    sd = rep["S"]
    assert sd["positive_control"] == "sensitive"
    assert sd["interpretable"] is True
    assert sd["level"] == 0.95
    assert sd["intervals"][("gtm", "registered")]["low"] is not None


def test_positive_control_insensitive_blocks_s_interpretation():
    rep = primary_report(_fixture(eng_closed_s=(-0.30, 0.34, -0.28, 0.32, 0.10)))
    assert rep["S"]["positive_control"] == "insensitive"
    assert rep["S"]["interpretable"] is False


def test_positive_control_uninterpretable_when_interval_unavailable():
    rep = primary_report(_fixture(eng_closed_s=(0.3,) * 5))
    assert rep["S"]["positive_control"] == "uninterpretable"
    assert rep["S"]["interpretable"] is False
    assert rep["S"]["intervals"][("eng", "closed")]["low"] is None
    assert rep["S"]["intervals"][("eng", "closed")]["mean"] == pytest.approx(0.3)


def test_nocomm_s_is_deterministic_check_without_interval():
    rep = primary_report(_fixture())
    assert rep["S"]["nocomm_zero_check"] == {"registered": True, "closed": True}
    assert ("nocomm", "registered") not in rep["S"]["intervals"]
    res = _fixture()
    res[("nocomm", "closed")]["S"] = np.array([0, 0, 1e-9, 0, 0.0])
    assert primary_report(res)["S"]["nocomm_zero_check"]["closed"] is False


def test_primary_unavailable_when_both_zero_variance():
    res = _fixture()
    res[("gtm", "registered")]["R"] = np.zeros(5)
    res[("dial", "registered")]["R"] = np.zeros(5)
    p = primary_report(res)["primary"]
    assert p["category"] == "pilot-interval-unavailable"
    assert p["contrasts"]["dial"]["low"] is None
    assert p["contrasts"]["dial"]["estimate"] == 0.0


# ------------------------------------------------- fix round 1
@pytest.mark.parametrize("v", [0.1, 0.3, 0.7])
def test_constant_decimals_are_unavailable(v):
    c = np.full(5, v)
    assert one_sample_ci(c, 0.95) is None
    assert welch_ci(c, np.full(5, v), 0.95) is None
    assert welch_ci(c, np.full(5, 0.9 * v), 0.95) is None
    # only one arm constant -> still available
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert welch_ci(x, c, 0.95) is not None
    assert welch_ci(c, x, 0.95) is not None


@pytest.mark.parametrize("v", [0.1, 0.3, 0.7])
def test_constant_s_positive_control_uninterpretable(v):
    rep = primary_report(_fixture(eng_closed_s=(v,) * 5))
    assert rep["S"]["positive_control"] == "uninterpretable"
    assert rep["S"]["intervals"][("eng", "closed")]["low"] is None


def test_constant_decimal_primary_unavailable():
    res = _fixture()
    res[("gtm", "registered")]["R"] = np.full(5, 0.7)
    res[("dial", "registered")]["R"] = np.full(5, 0.3)
    p = primary_report(res)["primary"]
    assert p["contrasts"]["dial"]["low"] is None
    assert p["category"] == "pilot-interval-unavailable"


def test_threshold_counts_mixture():
    res = _fixture()
    cap = 50e6
    res[("gtm", "registered")]["time_to_half"] = np.array([10e6, 20e6, cap, cap, cap])
    res[("gtm", "registered")]["reached_threshold"] = np.array(
        [True, True, True, False, False]
    )
    tc = primary_report(res)["threshold_counts"]["gtm"]
    assert tc == {
        "reached_before_final_eval": 2,
        "reached_at_final_eval": 1,
        "never_reached": 2,
    }


def test_missing_reached_threshold_rejected():
    res = _fixture()
    del res[("gtm", "registered")]["reached_threshold"]
    with pytest.raises(ValueError, match="reached_threshold"):
        primary_report(res)


def test_reached_threshold_inconsistent_rejected():
    res = _fixture()
    res[("gtm", "registered")]["reached_threshold"] = np.zeros(5, bool)  # t < cap
    with pytest.raises(ValueError, match="cap"):
        primary_report(res)


@pytest.mark.parametrize(
    "bad",
    [np.ones(4), np.ones(6), np.ones((5, 1)), np.array([0.1, np.nan, 0.2, 0.3, 0.4])],
)
def test_malformed_cell_rejected(bad):
    res = _fixture()
    res[("gtm", "registered")]["R"] = bad
    with pytest.raises(ValueError):
        primary_report(res)


def test_reached_threshold_length_rejected():
    res = _fixture()
    res[("gtm", "registered")]["reached_threshold"] = np.ones(4, bool)
    with pytest.raises(ValueError):
        primary_report(res)
