"""Light tests for the N10-D L0 skeleton (no JAX, torch or EGG)."""
import random

import pytest

from experiments.n10d.backends import make_egg_game, make_jaxmarl_game
from experiments.n10d.interfaces import episode_derangement
from experiments.n10d.null import empirical_p, simulate_arm_null, simulate_null, summarize
from experiments.n10d.spec import TaskSpec, blind_ceiling


def test_blind_ceiling_exact():
    assert blind_ceiling(TaskSpec("asym", 16, 4)) == 0.25
    assert blind_ceiling(TaskSpec("sym", 16, 4)) == 1.0


def test_bad_spec_rejected():
    with pytest.raises(ValueError):
        TaskSpec("asym", 4, 8)


def test_null_is_centred_and_p_is_valid():
    d = simulate_null(TaskSpec("asym", 16, 4), 4, 200, 0.02, n_draws=1000)
    s = summarize(d)
    assert abs(s.median) < 0.02 and s.p99 > s.p95 > s.median
    assert 0 < empirical_p(0.30, d) <= 1


def test_backends_are_stubs():
    for f in (make_jaxmarl_game, make_egg_game):
        with pytest.raises(NotImplementedError):
            f(TaskSpec("asym"))


def test_sym_null_is_degenerate_at_zero_sigma():
    # Revision 2026-11-06: makes the degenerate sym null visible (P99 = 0 at sigma = 0).
    spec = TaskSpec("sym", 16, 4)
    for paired in (True, False):
        d = simulate_null(spec, 8, 50, 0.0, n_draws=500, paired=paired)
        assert summarize(d).p99 == 0.0
        assert empirical_p(0.01, d) == 1 / 501


def test_paired_null_is_narrower_than_independent_arms():
    spec = TaskSpec("asym", 16, 4)
    paired = summarize(simulate_null(spec, 4, 100, 0.05, n_draws=1000))
    indep = summarize(simulate_null(spec, 4, 100, 0.05, n_draws=1000, paired=False))
    assert paired.p99 < indep.p99


def test_single_arm_null():
    a = summarize(simulate_arm_null(TaskSpec("asym", 16, 4), 4, 100, 0.0, n_draws=1000))
    assert abs(a.median) < 0.03 and a.p99 > 0
    s = simulate_arm_null(TaskSpec("sym", 16, 4), 4, 50, 0.0, n_draws=200)
    assert max(s) == 0.0 and min(s) == 0.0


def test_episode_derangement_has_no_fixed_point():
    rng = random.Random(0)
    for n in (2, 3, 5, 16, 64):
        for _ in range(50):
            perm = episode_derangement(n, rng)
            assert sorted(perm) == list(range(n))
            assert all(i != p for i, p in enumerate(perm))
    with pytest.raises(ValueError):
        episode_derangement(1, rng)
