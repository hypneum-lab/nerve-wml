"""Tests for the frozen N10-C estimation analysis (numpy/scipy only)."""

import numpy as np
import pytest
from scipy import stats

from experiments.procgen_multiagent_gtm_comm.analysis import (
    cross_play_report,
    fieller_ci,
    one_sample_ci,
    pilot_category,
    primary_report,
    welch_ci,
)

T5 = 2.5705818356  # t(0.975, df=5), scipy.stats.t.ppf(0.975, 5)


def test_welch_ci_matches_scipy():
    x = np.array([0.62, 0.70, 0.55, 0.81, 0.66, 0.73])
    y = np.array([0.40, 0.52, 0.47, 0.35, 0.58, 0.44])
    for level in (0.95, 1 - 0.05 / 3, 1 - 0.05 / 9):
        d, lo, hi = welch_ci(x, y, level)
        ci = stats.ttest_ind(x, y, equal_var=False).confidence_interval(level)
        assert d == pytest.approx(x.mean() - y.mean())
        assert lo == pytest.approx(ci.low, abs=1e-9)
        assert hi == pytest.approx(ci.high, abs=1e-9)


def test_welch_df_equal_variances_is_ten():
    # equal variances and n = 6 each -> Welch df = 10, t(0.975, 10) = 2.2281388520
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d, lo, hi = welch_ci(x + 10, x, 0.95)
    half = 2.2281388520 * np.sqrt(2 * 3.5 / 6)
    assert d == pytest.approx(10.0)
    assert hi - d == pytest.approx(half, abs=1e-6)
    assert stats.t.ppf(0.975, 10) == pytest.approx(2.2281388520, abs=1e-9)


def test_welch_ci_hand_computed():
    # s^2 = 3.5, v1 = 3.5/6, v2 = 0 -> df = v1^2 / (v1^2 / 5) = 5
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.zeros(6)
    d, lo, hi = welch_ci(x, y, 0.95)
    half = T5 * np.sqrt(3.5 / 6)
    assert d == pytest.approx(3.5)
    assert lo == pytest.approx(3.5 - half, abs=1e-6)
    assert hi == pytest.approx(3.5 + half, abs=1e-6)
    assert half == pytest.approx(1.96331, abs=1e-4)


def test_zero_variance_returns_none():
    assert welch_ci(np.full(6, 0.3), np.full(6, 0.3), 0.95) is None
    assert welch_ci(np.zeros(6), np.zeros(6), 0.95) is None
    assert one_sample_ci(np.full(6, 0.2), 0.95) is None


def test_one_sample_ci_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # s^2 = 3.5, s/sqrt6
    m, lo, hi = one_sample_ci(x, 0.95)
    half = T5 * np.sqrt(3.5 / 6)  # df = 5
    assert m == pytest.approx(3.5)
    assert lo == pytest.approx(3.5 - half, abs=1e-6)
    assert hi == pytest.approx(3.5 + half, abs=1e-6)


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
    nocomm = np.zeros(6)
    eng = np.array([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])  # mean 1, s2 = 0.056
    arm = np.array([0.35, 0.40, 0.45, 0.55, 0.60, 0.65])  # mean .5, s2 = 0.014
    # v_eng = .056/6, v_arm = .014/6, Cov = v_NOCOMM = 0, a=.5, b=1;
    # df = 5 for both (zero-variance NOCOMM), t^2 = 2.5705818356^2
    # qa = 1 - t^2 v_eng ; qc = .25 - t^2 v_arm ; roots = (.5 -+ sqrt(.25 - qa qc)) / qa
    t2 = 2.5705818356**2
    qa = 1 - t2 * 0.056 / 6
    qc = 0.25 - t2 * 0.014 / 6
    r = np.sqrt(0.25 - qa * qc)
    rho, lo, hi = fieller_ci(arm, nocomm, eng)
    assert rho == pytest.approx(0.5)
    assert lo == pytest.approx((0.5 - r) / qa, abs=1e-9)
    assert hi == pytest.approx((0.5 + r) / qa, abs=1e-9)
    # checked against scipy t(0.975, 5) in a separate evaluation
    assert lo == pytest.approx(0.34863, abs=1e-4)
    assert hi == pytest.approx(0.71710, abs=1e-4)


def test_fieller_endpoints_satisfy_inequality_boundary():
    rng = np.random.default_rng(3)
    nocomm = rng.normal(0.1, 0.02, 6)
    eng = rng.normal(0.8, 0.03, 6)
    arm = rng.normal(0.5, 0.03, 6)
    rho, lo, hi = fieller_ci(arm, nocomm, eng)
    a = arm.mean() - nocomm.mean()
    b = eng.mean() - nocomm.mean()
    vn, va, ve = (np.var(z, ddof=1) / 6 for z in (nocomm, arm, eng))
    var_a, var_b = va + vn, ve + vn
    # smaller Welch df of the two differences
    df_a = (va + vn) ** 2 / (va**2 / 5 + vn**2 / 5)
    df_b = (ve + vn) ** 2 / (ve**2 / 5 + vn**2 / 5)
    t = stats.t.ppf(0.975, min(df_a, df_b))
    for r in (lo, hi):
        lhs = (a - r * b) ** 2
        rhs = t**2 * (var_a - 2 * r * vn + r**2 * var_b)
        assert lhs == pytest.approx(rhs, rel=1e-9)
    assert lo < rho < hi


def test_fieller_none_when_denominator_crosses_zero():
    nocomm = np.array([0.40, 0.50, 0.45, 0.55, 0.50, 0.45])
    eng = np.array([0.45, 0.55, 0.50, 0.60, 0.40, 0.50])  # b ~ 0.0, wide
    arm = np.array([0.5, 0.6, 0.55, 0.65, 0.5, 0.55])
    assert fieller_ci(arm, nocomm, eng) is None


def test_fieller_none_when_variances_degenerate():
    z = np.zeros(6)
    assert fieller_ci(np.full(6, 0.5), z, np.ones(6)) is None


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


def _fixture(eng_closed_s=(0.30, 0.34, 0.28, 0.32, 0.36, 0.30)):
    cap = 50e6
    res = {}
    base = [-0.02, 0.01, 0.0, 0.02, -0.01, 0.0]  # zero-mean jitter
    j = np.array(base)

    def mk(r, t, bw, s):
        return _cell(r + j, np.minimum(t + 1e6 * j * 10, cap), bw + j, s + j * 0.1)

    for cond in ("registered", "closed"):
        res[("nocomm", cond)] = _cell(0.10 + j, [cap] * 6, [0.0] * 6, [0.0] * 6)
        res[("eng", cond)] = mk(0.80, np.full(6, 20e6), np.full(6, 6.0), np.full(6, 0.3))
        res[("gtm", cond)] = mk(0.60, np.full(6, 30e6), np.full(6, 3.0), np.full(6, 0.2))
        res[("dial", cond)] = mk(0.30, np.full(6, 45e6), np.full(6, 5.0), np.full(6, 0.1))
    res[("eng", "closed")]["S"] = np.array(eng_closed_s, float)
    for arm, r, t, bw in (("mlp", 0.20, 48e6, 4.0), ("atoc", 0.40, 40e6, 7.0)):
        res[(arm, "registered")] = mk(r, np.full(6, t), np.full(6, bw), np.full(6, 0.05))
    return res


def test_primary_report_end_to_end():
    rep = primary_report(_fixture())
    p = rep["primary"]
    assert p["level"] == pytest.approx(1 - 0.05 / 3)
    assert p["delta"] == 0.10 and p["range"] == 1.0
    assert set(p["contrasts"]) == {"mlp", "atoc", "dial"}
    j = np.array([-0.02, 0.01, 0.0, 0.02, -0.01, 0.0])
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
        [0.3, -0.3, 0.2, -0.1, 0.0, 0.1]
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
    rep = primary_report(_fixture(eng_closed_s=(-0.30, 0.34, -0.28, 0.32, 0.10, -0.05)))
    assert rep["S"]["positive_control"] == "insensitive"
    assert rep["S"]["interpretable"] is False


def test_positive_control_uninterpretable_when_interval_unavailable():
    rep = primary_report(_fixture(eng_closed_s=(0.3,) * 6))
    assert rep["S"]["positive_control"] == "uninterpretable"
    assert rep["S"]["interpretable"] is False
    assert rep["S"]["intervals"][("eng", "closed")]["low"] is None
    assert rep["S"]["intervals"][("eng", "closed")]["mean"] == pytest.approx(0.3)


def test_nocomm_s_is_deterministic_check_without_interval():
    rep = primary_report(_fixture())
    assert rep["S"]["nocomm_zero_check"] == {"registered": True, "closed": True}
    assert ("nocomm", "registered") not in rep["S"]["intervals"]
    res = _fixture()
    res[("nocomm", "closed")]["S"] = np.array([0, 0, 1e-9, 0, 0.0, 0.0])
    assert primary_report(res)["S"]["nocomm_zero_check"]["closed"] is False


def test_primary_unavailable_when_both_zero_variance():
    res = _fixture()
    res[("gtm", "registered")]["R"] = np.zeros(6)
    res[("dial", "registered")]["R"] = np.zeros(6)
    p = primary_report(res)["primary"]
    assert p["category"] == "pilot-interval-unavailable"
    assert p["contrasts"]["dial"]["low"] is None
    assert p["contrasts"]["dial"]["estimate"] == 0.0


# ------------------------------------------------- fix round 1
@pytest.mark.parametrize("v", [0.1, 0.3, 0.7])
def test_constant_decimals_are_unavailable(v):
    c = np.full(6, v)
    assert one_sample_ci(c, 0.95) is None
    assert welch_ci(c, np.full(6, v), 0.95) is None
    assert welch_ci(c, np.full(6, 0.9 * v), 0.95) is None
    # only one arm constant -> still available
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert welch_ci(x, c, 0.95) is not None
    assert welch_ci(c, x, 0.95) is not None


@pytest.mark.parametrize("v", [0.1, 0.3, 0.7])
def test_constant_s_positive_control_uninterpretable(v):
    rep = primary_report(_fixture(eng_closed_s=(v,) * 6))
    assert rep["S"]["positive_control"] == "uninterpretable"
    assert rep["S"]["intervals"][("eng", "closed")]["low"] is None


def test_constant_decimal_primary_unavailable():
    res = _fixture()
    res[("gtm", "registered")]["R"] = np.full(6, 0.7)
    res[("dial", "registered")]["R"] = np.full(6, 0.3)
    p = primary_report(res)["primary"]
    assert p["contrasts"]["dial"]["low"] is None
    assert p["category"] == "pilot-interval-unavailable"


def test_threshold_counts_mixture():
    res = _fixture()
    cap = 50e6
    res[("gtm", "registered")]["time_to_half"] = np.array([10e6, 20e6, cap, cap, cap, cap])
    res[("gtm", "registered")]["reached_threshold"] = np.array(
        [True, True, True, False, False, False]
    )
    tc = primary_report(res)["threshold_counts"]["gtm"]
    assert tc == {
        "reached_before_final_eval": 2,
        "reached_at_final_eval": 1,
        "never_reached": 3,
    }


def test_missing_reached_threshold_rejected():
    res = _fixture()
    del res[("gtm", "registered")]["reached_threshold"]
    with pytest.raises(ValueError, match="reached_threshold"):
        primary_report(res)


def test_reached_threshold_inconsistent_rejected():
    res = _fixture()
    res[("gtm", "registered")]["reached_threshold"] = np.zeros(6, bool)  # t < cap
    with pytest.raises(ValueError, match="cap"):
        primary_report(res)


@pytest.mark.parametrize(
    "bad",
    [np.ones(5), np.ones(7), np.ones((6, 1)), np.array([0.1, np.nan, 0.2, 0.3, 0.4, 0.5])],
)
def test_malformed_cell_rejected(bad):
    res = _fixture()
    res[("gtm", "registered")]["R"] = bad
    with pytest.raises(ValueError):
        primary_report(res)


def test_reached_threshold_length_rejected():
    res = _fixture()
    res[("gtm", "registered")]["reached_threshold"] = np.ones(5, bool)
    with pytest.raises(ValueError):
        primary_report(res)


def test_cross_play_report_is_descriptive_only():
    r_int = [0.9, 0.8, 0.7, 0.85, 0.95, 0.75]
    r_cross = [0.2, 0.3, 0.1, 0.25, 0.15, 0.2]
    out = cross_play_report({
        ("gtm", "registered"): {"R_intact": r_int, "R_cross": r_cross},
        ("nocomm", "closed"): {"R_intact": r_int, "R_cross": r_int},
    })
    g = out[("gtm", "registered")]
    assert g["R_cross"] == r_cross
    assert g["mean_cross"] == pytest.approx(np.mean(r_cross))
    assert g["mean_intact"] == pytest.approx(np.mean(r_int))
    assert g["gap"] == pytest.approx(np.mean(r_int) - np.mean(r_cross))
    # no interval, p-value, label or comparison of any kind
    assert set(g) == {"R_cross", "mean_cross", "mean_intact", "gap"}
    assert out[("nocomm", "closed")]["gap"] == pytest.approx(0.0)


def test_cross_play_report_validates_input():
    ok = {"R_intact": [0.5, 0.6], "R_cross": [0.1, 0.2]}
    with pytest.raises(ValueError):
        cross_play_report({("gtm", "registered"): {"R_intact": [0.5, 0.6]}})
    with pytest.raises(ValueError):
        cross_play_report({("gtm", "registered"): {**ok, "R_cross": [0.1]}})
    with pytest.raises(ValueError):
        cross_play_report({("gtm", "registered"): {**ok, "R_cross": [0.1, np.nan]}})
    with pytest.raises(ValueError):
        cross_play_report({("gtm", "registered"): {"R_intact": [0.5], "R_cross": [0.1]}})
