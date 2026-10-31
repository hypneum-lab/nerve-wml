"""Light tests for the N10-D L0 skeleton (no JAX, torch or EGG)."""
import pytest

from experiments.n10d.backends import make_egg_game, make_jaxmarl_game
from experiments.n10d.null import empirical_p, simulate_null, summarize
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
