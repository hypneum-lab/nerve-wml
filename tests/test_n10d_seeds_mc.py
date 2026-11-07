"""Small reproducible checks of the k Monte-Carlo (planning assumption only)."""

from experiments.n10d.required_seeds_mc import (
    K_CONFIRM,
    SENSITIVITY_PI,
    prob_a,
    report,
    required_seeds_sim,
)

K = K_CONFIRM


def test_k_confirm_is_twenty_with_pi_sensitivity():
    # PI decision 2026-11-06 ~18:00: follow the sourced analysis, k = 20.
    assert K_CONFIRM == 20
    assert SENSITIVITY_PI == (0.70, 0.75, 0.80)


def test_mc_is_reproducible_and_null_never_gives_a():
    p1 = prob_a(K, n_sims=300, n_boot=300)
    assert p1 == prob_a(K, n_sims=300, n_boot=300)
    assert prob_a(K, pi=0.0, n_sims=300, n_boot=300) == 0.0
    assert p1 > 0.85


def test_report_power_falls_with_pi():
    out = report(K, n_sims=300, n_boot=300)
    p = [out["power"][str(pi)] for pi in SENSITIVITY_PI]
    assert p[0] < p[2] and out["p_a_under_h0"] == 0.0
    assert out["global_power"]["0.8"] == out["power"]["0.8"] ** 2


def test_required_seeds_sim_returns_first_k_meeting_target():
    k, table = required_seeds_sim(0.90, 2, k_min=15, k_max=30, n_sims=300, n_boot=300)
    assert table[k] ** 2 >= 0.90
    assert all(p**2 < 0.90 for kk, p in table.items() if kk < k)
