"""Implementation checks for the NH-01 confirmatory run (amended pre-reg).

No empirical number: the tests check the configurable path of the pilot
runner (default = pilot output), the unimodality rule and the amended
verdict on synthetic condition entries.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.nh01_hopfield_dream import confirmatory as cf
from experiments.nh01_hopfield_dream import run as nh

SMALL_CFG = {
    "alpha_grid": (0.1, 0.2),
    "t_grid": (0.0, 1.0, 10.0),
    "conditions": ((0.0, 0.1), (0.2, 0.1)),
    "max_probes": 8,
    "spectral": True,
}


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "task_seconds"}


def test_cfg_path_counts_and_spectral_flag() -> None:
    recs = nh.evaluate_task((32, 0, SMALL_CFG))
    n_arms = 1 + 2 * (len(SMALL_CFG["t_grid"]) - 1)
    assert len(recs) == len(SMALL_CFG["alpha_grid"]) * n_arms * len(SMALL_CFG["conditions"])
    recs_nospec = nh.evaluate_task((32, 0, dict(SMALL_CFG, spectral=False)))
    for a, b in zip(recs, recs_nospec, strict=True):
        a2, b2 = _strip(a), _strip(b)
        c_spec_a, c_spec_b = a2.pop("c_spec"), b2.pop("c_spec")
        if a["arm"] == "hebb":
            assert c_spec_a == 1.0 and c_spec_b == 1.0
        else:
            assert math.isfinite(c_spec_a) and math.isnan(c_spec_b)
        assert a2 == b2


def test_cfg_path_default_equals_pilot_constants() -> None:
    """(n, seed) and (n, seed, cfg with the pilot constants) give identical records."""
    cfg = {
        "alpha_grid": nh.ALPHA_GRID, "t_grid": nh.T_GRID, "conditions": nh.CONDITIONS,
        "max_probes": nh.MAX_PROBES, "spectral": True,
    }
    # restrict to a tiny grid for speed by monkeypatching nothing: compare on N = 16,
    # where the full pilot grid is cheap (P <= 15)
    a = nh.evaluate_task((16, 3))
    b = nh.evaluate_task((16, 3, cfg))
    assert [_strip(r) for r in a] == [_strip(r) for r in b]


def test_shrink_invariance_holds_at_n1000_scale_tolerance() -> None:
    """The tie tolerance keeps the sign rule scale-invariant at N = 1000 too."""
    rng = np.random.default_rng(5)
    xi = np.where(rng.random((100, 1000)) < 0.5, -1.0, 1.0)
    j0 = nh.offdiag(nh.hebbian_kernel(xi))
    cues = nh.make_cues(xi, np.arange(4), 0.10, np.random.default_rng(1))
    s_a, sw_a, _ = nh.run_zero_temperature(j0, cues, np.random.default_rng(2), max_sweeps=5)
    s_b, sw_b, _ = nh.run_zero_temperature(0.613 * j0, cues, np.random.default_rng(2), max_sweeps=5)
    assert sw_a == sw_b and np.array_equal(s_a, s_b)


def test_unimodality_rule() -> None:
    assert cf.unimodality([0.1, 0.3, 0.6, 0.5, 0.4], 0.02) == (2, True)
    assert cf.unimodality([0.1, 0.3, 0.6, 0.59, 0.6], 0.02) == (2, True)  # within tol
    assert cf.unimodality([0.1, 0.3, 0.6, 0.5, 0.55], 0.02) == (2, False)  # rises again
    assert cf.unimodality([0.1, 0.05, 0.6, 0.5, 0.4], 0.02) == (2, False)  # dips before
    assert cf.unimodality([0.1, 0.2, 0.3, 0.4, 0.5], 0.02) == (4, True)  # edge maximum


def _entry(hebb: list[float], dream: dict[float, list[float]], shrink_ratio: float) -> dict:
    hebb_a = np.array(hebb)
    e: dict = {
        "hebb": {"alpha_c_per_seed": hebb},
        "dream": {}, "shrink": {},
        "spurious": {
            f"{a:g}": {"relative_reduction_dream": 1.0} for a in cf.SPURIOUS_ALPHAS
        },
    }
    for t in cf.T_GRID[1:]:
        vals = np.array(dream[t])
        e["dream"][f"{t:g}"] = {
            "alpha_c_per_seed": vals.tolist(),
            "median": float(np.median(vals)),
            "ratio_mean_ci": nh.bootstrap_mean_ci(vals / hebb_a, n_boot=200),
            "diff_mean_ci": nh.bootstrap_mean_ci(vals - hebb_a, n_boot=200),
        }
        e["shrink"][f"{t:g}"] = {
            "ratio_per_seed": [shrink_ratio] * len(hebb),
            "ratio_mean_ci": (shrink_ratio, shrink_ratio, shrink_ratio),
        }
    return e


def test_verdict_confirmed_case() -> None:
    hebb = [0.15 + 0.01 * i for i in range(5)]
    peak = {1.0: 0.4, 3.0: 0.55, 5.0: 0.58, 7.0: 0.6, 10.0: 0.6, 15.0: 0.59, 20.0: 0.57,
            30.0: 0.55, 100.0: 0.5}
    dream = {t: [v + 0.01 * i for i in range(5)] for t, v in peak.items()}
    conds = {
        f"T0_f{cf.FLIPS:g}_N{cf.N_CONF}": _entry(hebb, dream, 1.0),
        f"T{cf.T_CONTROL:g}_f{cf.FLIPS:g}_N{cf.N_CONF}": _entry(hebb, dream, 0.98),
    }
    v = cf.verdict(conds)
    assert v["C1_gain"]["label"] == "confirmed"
    assert v["C2_unimodal"]["label"] == "holds" and v["C2_unimodal"]["t_star"] == 7.0
    assert v["C2_unimodal"]["finite_optimum"] is True
    assert v["C3_shrink_T_control"]["label"] == "holds"
    assert v["overall"] == "confirme"


def test_verdict_refuted_by_shrink_and_void_by_t0() -> None:
    hebb = [0.15] * 5
    dream = {t: [0.3 + 0.001 * t] * 5 for t in cf.T_GRID[1:]}  # still rising at t = 100
    conds = {
        f"T0_f{cf.FLIPS:g}_N{cf.N_CONF}": _entry(hebb, dream, 1.0),
        f"T{cf.T_CONTROL:g}_f{cf.FLIPS:g}_N{cf.N_CONF}": _entry(hebb, dream, 2.5),
    }
    v = cf.verdict(conds)
    assert v["C1_gain"]["label"] == "inconclusive"
    assert v["C3_shrink_T_control"]["label"] == "refutes"
    assert v["C2_unimodal"]["label"] == "inconclusive"  # maximum at the grid edge
    assert v["overall"] == "refute"
    conds[f"T0_f{cf.FLIPS:g}_N{cf.N_CONF}"] = _entry(hebb, dream, 0.9)
    assert cf.verdict(conds)["overall"] == "void_implementation_failure"
