"""Light tests for the N10-D L0 skeleton (no JAX, torch or EGG)."""
import random

import pytest

from experiments.n10d.backends import make_egg_game, make_jaxmarl_game
from experiments.n10d.interfaces import derange_episodes
from experiments.n10d.null import (
    _latent,
    empirical_p,
    simulate_level_null,
    simulate_null,
    summarize,
)
from experiments.n10d.spec import TaskSpec, blind_ceiling, oracle_contrast


def test_blind_ceiling_exact():
    assert blind_ceiling(TaskSpec("asym", 16, 4)) == 0.25
    assert blind_ceiling(TaskSpec("sym", 16, 4)) == 1.0


def test_oracle_contrast_is_analytical():
    assert oracle_contrast(TaskSpec("asym", 16, 4)) == 0.75
    assert oracle_contrast(TaskSpec("sym", 16, 4)) == 0.0


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
    # sym: B0 = 1.0 and sigma_seed = 0, so every arm scores exactly 1.0.
    sym = TaskSpec("sym", 16, 4)
    d = simulate_null(sym, 8, 500, 0.0, n_draws=200)
    assert summarize(d).p99 == 0.0 and set(d) == {0.0}
    assert set(simulate_level_null(sym, 8, 500, 0.0, n_draws=200)) == {1.0}


def test_blind_latent_never_exceeds_ceiling():
    rng = random.Random(0)
    assert max(_latent(rng, 0.25, 0.05) for _ in range(10_000)) <= 0.25


def test_paired_null_is_narrower_than_independent():
    spec = TaskSpec("asym", 16, 4)
    paired = summarize(simulate_null(spec, 8, 100, 0.05, n_draws=1000))
    indep = summarize(simulate_null(spec, 8, 100, 0.05, n_draws=1000, paired=False))
    assert paired.p99 < indep.p99


def test_msg_shuffle_deranges_episodes_not_targets():
    rng = random.Random(7)
    perms = [derange_episodes(50, rng) for _ in range(100)]
    for perm in perms:
        assert sorted(perm) == list(range(50))
        assert all(i != j for i, j in enumerate(perm))
    assert len({tuple(p) for p in perms}) == 100  # fresh per batch, not fixed
    # Same target in two episodes must be able to receive different messages:
    # the shuffled message is not a function of the target identity.
    targets = [e % 4 for e in range(50)]
    perm = perms[0]
    by_target = {}
    for e, src in enumerate(perm):
        by_target.setdefault(targets[e], set()).add(targets[src])
    assert any(len(srcs) > 1 for srcs in by_target.values())
    with pytest.raises(ValueError):
        derange_episodes(1, rng)
