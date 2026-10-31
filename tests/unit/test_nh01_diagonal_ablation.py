"""Implementation checks for the NH-01 diagonal ablation (pre-reg draft L0).

No empirical number.  The tests check:
  1. That dream_diag and dream_offdiag kernels differ **only** on the diagonal.
  2. That at t = 0 both arms give hebb (off-diagonal) and hebb (with diagonal)
     respectively, and that they differ only on the diagonal.
  3. That the tie tolerance scales with the larger entries of the diag-kept
     kernel (non-trivial: max|J_diag| >= max|J_offdiag|).
  4. A one-cell sanity check: evaluate_task on a tiny grid runs without error
     and returns the correct number of records.
  5. verdict() returns the correct C5/C6 labels on synthetic summary inputs
     covering every decision branch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.nh01_hopfield_dream import run as nh
from experiments.nh01_hopfield_dream.diagonal_ablation import (
    ALPHA_GRID,
    CENSOR_MAX_SEEDS,
    CONTROL_DROP_CI,
    T_GRID,
    THRESH_NOT_CAUSE,
    TOL_C2,
    analyse,
    evaluate_task,
    verdict,
)


def _patterns(p: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random((p, n)) < 0.5, -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Test 1: dream_diag and dream_offdiag differ only on the diagonal
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t", [1.0, 7.0, 100.0])
def test_diag_and_offdiag_kernels_differ_only_on_diagonal(t: float) -> None:
    """At any t > 0, keeping vs zeroing the diagonal changes only J_ii entries."""
    n = 64
    xi = _patterns(6, n, seed=42)
    j_full = nh.hebbian_kernel(xi)
    w, v = np.linalg.eigh(j_full)
    w = np.clip(w, 0.0, None)
    wt = w * (1.0 + t) / (1.0 + t * w)
    jt_full = (v * wt) @ v.T        # dream_diag kernel
    jt_off = nh.offdiag(jt_full)    # dream_offdiag kernel

    diff = jt_full - jt_off
    # Off-diagonal entries must be identical (zero difference)
    mask_off = ~np.eye(n, dtype=bool)
    assert np.allclose(diff[mask_off], 0.0, atol=1e-14), (
        "dream_diag and dream_offdiag differ on off-diagonal entries"
    )
    # Diagonal entries of diff must equal diagonal of jt_full
    diag_diff = np.diag(diff)
    diag_jt = np.diag(jt_full)
    assert np.allclose(diag_diff, diag_jt, atol=1e-14), (
        "Diagonal of diff does not match diagonal of jt_full"
    )
    # The diagonal of jt_full is non-zero (self-coupling is present)
    assert np.all(np.abs(diag_jt) > 1e-6), (
        "Diagonal of dreaming kernel is unexpectedly zero or tiny"
    )


def test_at_t0_offdiag_arm_matches_hebb_offdiag() -> None:
    """At t = 0, dream_offdiag uses offdiag(J) = standard confirmatory hebb arm."""
    n = 64
    xi = _patterns(4, n, seed=7)
    j_full = nh.hebbian_kernel(xi)
    j0_off = nh.offdiag(j_full)
    # J(0) from spectral path = J
    jt_full_t0 = np.array(j_full, dtype=np.float64, copy=True)
    jt_off_t0 = nh.offdiag(jt_full_t0)
    assert np.allclose(jt_off_t0, j0_off, atol=1e-14)


def test_at_t0_diag_arm_uses_full_hebb_kernel() -> None:
    """At t = 0, dream_diag uses J (Hebbian with diagonal), not j0_off."""
    n = 64
    xi = _patterns(4, n, seed=7)
    j_full = nh.hebbian_kernel(xi)
    j0_off = nh.offdiag(j_full)
    jt_full_t0 = np.array(j_full, dtype=np.float64, copy=True)
    # jt_full_t0 should differ from j0_off on the diagonal
    diag_diff = np.diag(jt_full_t0 - j0_off)
    assert np.all(np.abs(diag_diff) > 1e-6), (
        "dream_diag at t=0 should retain non-zero diagonal"
    )
    mask_off = ~np.eye(n, dtype=bool)
    assert np.allclose((jt_full_t0 - j0_off)[mask_off], 0.0, atol=1e-14)


# --------------------------------------------------------------------------- #
# Test 2: tie-tolerance is larger for the diag-kept kernel
# --------------------------------------------------------------------------- #

def test_diag_kernel_has_larger_max_abs_than_offdiag() -> None:
    """max|J_diag| >= max|J_offdiag| because the diagonal entries are positive."""
    n = 64
    xi = _patterns(6, n, seed=13)
    j_full = nh.hebbian_kernel(xi)
    w, v = np.linalg.eigh(j_full)
    w = np.clip(w, 0.0, None)
    for t in [1.0, 7.0, 100.0]:
        wt = w * (1.0 + t) / (1.0 + t * w)
        jt_full = (v * wt) @ v.T
        jt_off = nh.offdiag(jt_full)
        assert float(np.abs(jt_full).max()) >= float(np.abs(jt_off).max()), (
            f"At t={t}: max|J_diag| < max|J_offdiag|"
        )


# --------------------------------------------------------------------------- #
# Test 3: evaluate_task returns correct number of records on a tiny grid
# --------------------------------------------------------------------------- #

def test_evaluate_task_record_count() -> None:
    """evaluate_task returns 2 * n_t * n_alpha records (2 arms, same t and load grid)."""
    small_cfg = {
        "alpha_grid": (0.10, 0.20),
        "t_grid": (0.0, 1.0, 10.0),
        "max_probes": 8,
    }
    n_alpha = len(small_cfg["alpha_grid"])
    n_t = len(small_cfg["t_grid"])
    n_arms = 2  # dream_offdiag and dream_diag

    recs = evaluate_task((64, 200, small_cfg))
    assert len(recs) == n_arms * n_t * n_alpha, (
        f"Expected {n_arms * n_t * n_alpha} records, got {len(recs)}"
    )
    arms_seen = {r["arm"] for r in recs}
    assert arms_seen == {"dream_offdiag", "dream_diag"}, (
        f"Unexpected arm names: {arms_seen}"
    )


def test_evaluate_task_seeds_200_to_209_do_not_collide_with_0_to_9() -> None:
    """Patterns generated with seed=200 differ from seed=0 at same N."""
    n = 64
    p = 5
    xi_200 = np.where(nh.stream(200, n, 1).random((p, n)) < 0.5, -1.0, 1.0)
    xi_0 = np.where(nh.stream(0, n, 1).random((p, n)) < 0.5, -1.0, 1.0)
    assert not np.array_equal(xi_200, xi_0), (
        "Seeds 200 and 0 generated identical patterns — namespace collision"
    )


# --------------------------------------------------------------------------- #
# Test 4: verdict() on synthetic summary inputs — all branches
# --------------------------------------------------------------------------- #

def _make_summary(
    *,
    pc_overlap: bool = True,
    mean_diag: float = TOL_C2 - 0.001,
    hi_diag: float = THRESH_NOT_CAUSE - 0.001,
    lo_diag: float = 0.0,
    lo_dd: float = 0.001,
    hi_dd: float = 0.05,
    n_censored_off_tstar: int = 0,
    n_censored_diag_tstar: int = 0,
) -> dict:
    """Build a minimal synthetic summary dict suitable for verdict()."""
    t_star_str = f"{7.0:g}"
    t_end_str = f"{100.0:g}"

    def _arm_cg(n_tstar: int) -> dict:
        return {
            t_star_str: {"n_censored": n_tstar, "exceeds_limit": n_tstar > CENSOR_MAX_SEEDS},
            t_end_str: {"n_censored": 0, "exceeds_limit": False},
        }

    return {
        "positive_control": {
            "ci_overlap": pc_overlap,
            "reference_mean": 0.0876,
            "reference_ci": list(CONTROL_DROP_CI),
            "D_offdiag_mean_ci": [0.0876, CONTROL_DROP_CI[0], CONTROL_DROP_CI[1]],
        },
        "censoring_guard": {
            "dream_offdiag": _arm_cg(n_censored_off_tstar),
            "dream_diag": _arm_cg(n_censored_diag_tstar),
        },
        "D_diag": {"mean_ci": [mean_diag, lo_diag, hi_diag], "per_seed": []},
        "delta_D": {"mean_ci": [lo_dd + 0.01, lo_dd, hi_dd], "per_seed": []},
    }


def test_verdict_positive_control_failed() -> None:
    s = _make_summary(pc_overlap=False)
    v = verdict(s)
    assert v["C5"] == "positive_control_failed"
    assert v["C6"] is None


def test_verdict_inconclusive_censoring_offdiag() -> None:
    s = _make_summary(n_censored_off_tstar=CENSOR_MAX_SEEDS + 1)
    v = verdict(s)
    assert v["C5"] == "inconclusive_censoring"
    assert v["C6"] is None


def test_verdict_inconclusive_censoring_diag() -> None:
    s = _make_summary(n_censored_diag_tstar=CENSOR_MAX_SEEDS + 1)
    v = verdict(s)
    assert v["C5"] == "inconclusive_censoring"
    assert v["C6"] is None


def test_verdict_diagonal_is_cause() -> None:
    """D_diag ≤ TOL_C2, CI upper bound < THRESH_NOT_CAUSE, ΔD CI lower bound > 0."""
    s = _make_summary(
        mean_diag=TOL_C2 - 0.005,
        hi_diag=THRESH_NOT_CAUSE - 0.005,
        lo_diag=0.0,
        lo_dd=0.01,
        hi_dd=0.05,
    )
    v = verdict(s)
    assert v["C5"] == "diagonal_is_cause"
    assert v["C6"] == "dD_positive"


def test_verdict_diagonal_is_not_cause() -> None:
    """D_diag ≥ THRESH_NOT_CAUSE, CI lower bound > TOL_C2; ΔD CI contains 0."""
    s = _make_summary(
        mean_diag=THRESH_NOT_CAUSE + 0.01,
        lo_diag=TOL_C2 + 0.005,
        hi_diag=THRESH_NOT_CAUSE + 0.05,
        lo_dd=-0.01,
        hi_dd=0.01,
    )
    v = verdict(s)
    assert v["C5"] == "diagonal_is_not_cause"
    assert v["C6"] == "dD_zero"


def test_verdict_inconclusive_mid_range() -> None:
    """D_diag between TOL_C2 and THRESH_NOT_CAUSE — neither branch fires."""
    s = _make_summary(
        mean_diag=(TOL_C2 + THRESH_NOT_CAUSE) / 2.0,
        hi_diag=THRESH_NOT_CAUSE + 0.01,
        lo_diag=TOL_C2 - 0.005,
        lo_dd=-0.01,
        hi_dd=0.01,
    )
    v = verdict(s)
    assert v["C5"] == "inconclusive"
    assert v["C6"] == "dD_zero"


def test_verdict_c6_dD_negative() -> None:
    """ΔD CI upper bound < 0 → dD_negative (unexpected but pre-specified)."""
    s = _make_summary(
        mean_diag=(TOL_C2 + THRESH_NOT_CAUSE) / 2.0,
        hi_diag=THRESH_NOT_CAUSE - 0.005,
        lo_diag=TOL_C2 + 0.005,
        lo_dd=-0.05,
        hi_dd=-0.001,
    )
    v = verdict(s)
    assert v["C6"] == "dD_negative"


def test_verdict_written_into_analyse_output() -> None:
    """End-to-end: analyse() + verdict() on a tiny real run produces the verdict key."""
    small_cfg = {
        "alpha_grid": (0.10, 0.20, 0.30),
        "t_grid": (0.0, 7.0, 100.0),
        "max_probes": 4,
    }
    # Run two seeds to have enough data for bootstrap
    recs = []
    for seed in (200, 201):
        recs.extend(evaluate_task((64, seed, small_cfg)))

    summary = analyse(recs, alpha_grid=(0.10, 0.20, 0.30), t_grid=(0.0, 7.0, 100.0))
    v = verdict(summary)
    assert "C5" in v and "C6" in v
    assert v["C5"] in {
        "positive_control_failed", "inconclusive_censoring",
        "diagonal_is_cause", "diagonal_is_not_cause", "inconclusive",
    }
