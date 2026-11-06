"""Small reproducible checks of the k Monte-Carlo (planning assumption only)."""
from experiments.n10d.required_seeds_mc import prob_a, required_seeds_sim

K = 19


def test_mc_is_reproducible_and_null_never_gives_a():
    p1 = prob_a(K, n_sims=300, n_boot=300)
    assert p1 == prob_a(K, n_sims=300, n_boot=300)
    assert prob_a(K, pi=0.0, n_sims=300, n_boot=300) == 0.0
    assert p1 > 0.85


def test_required_seeds_sim_returns_first_k_meeting_target():
    k, table = required_seeds_sim(0.90, 2, k_min=15, k_max=30, n_sims=300, n_boot=300)
    assert table[k] ** 2 >= 0.90
    assert all(p ** 2 < 0.90 for kk, p in table.items() if kk < k)
