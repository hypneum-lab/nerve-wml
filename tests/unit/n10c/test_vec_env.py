"""VecCoopEnv: numpy-only, procgen-free (scripted envs); runs in both envs."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.procgen_multiagent_gtm_comm.harness.codes import (
    run_seed_sequence,
)
from experiments.procgen_multiagent_gtm_comm.harness.vec_env import (
    VecCoopEnv,
)

N = 3


class _Scripted:
    """Coin for agent 0 at step 0 and for agent 1 at step 1 -> success."""

    def __init__(self, rand_seed: int, coin_steps: set[int]) -> None:
        self.rand_seed = rand_seed
        self.coin_steps = coin_steps
        self.t = -1

    def act(self, actions: np.ndarray) -> None:
        self.t += 1

    def observe(self):
        coin = self.t in self.coin_steps
        img = np.full((1, 64, 64, 3), (self.t + 1) % 256, dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([coin])

    def close(self) -> None:
        pass


def _vec(team_summary: bool, n: int = N):
    created: list[_Scripted] = []

    def factory(rand_seed: int) -> _Scripted:
        i = len(created) % 2
        env = _Scripted(rand_seed, {0} if i == 0 else {1})
        created.append(env)
        return env

    vec = VecCoopEnv(
        n,
        run_seed_sequence(0, "gtm", "registered"),
        team_summary,
        env_factory=factory,
    )
    return vec, created


def _act(n: int = N) -> np.ndarray:
    return np.zeros((n, 2), dtype=np.int32)


def test_vec_env_autoreset_returns_new_episode_obs() -> None:
    vec, _ = _vec(True)
    first = vec.reset()
    assert first.obs.shape == (N, 2, 64, 64, 3)
    assert first.obs.dtype == np.uint8
    assert first.summary.shape == (N, 2, 2)
    assert first.summary.dtype == np.float32
    s0 = vec.step(_act())
    assert not s0.done.any()
    assert s0.finished.tolist() == [[True, False]] * N
    s1 = vec.step(_act())
    assert s1.done.all()
    assert s1.team_reward.dtype == np.float32
    assert np.all(s1.team_reward == 1.0)
    # New episode's first observation: frozen frames of the old episode
    # were 1/2; fresh envs show the initial frame (step index -1 -> 0).
    assert np.all(s1.obs == 0)
    assert np.all(s1.summary == 0.0)
    assert np.all(s1.window[0] == False)  # noqa: E712
    assert np.all(s1.window[1] == 0)
    # the next episode is playable
    assert not vec.step(_act()).done.any()


def test_vec_env_pairs_share_level_seed() -> None:
    vec, created = _vec(True)
    seeds = [e.rand_seed for e in created]
    for k in range(N):
        assert seeds[2 * k] == seeds[2 * k + 1]
    assert len({seeds[2 * k] for k in range(N)}) == N


def test_vec_env_is_deterministic_per_seed_sequence() -> None:
    _, a = _vec(True)
    _, b = _vec(True)
    assert [e.rand_seed for e in a] == [e.rand_seed for e in b]


def test_closed_condition_summary_is_zero() -> None:
    vec, _ = _vec(False)
    assert np.all(vec.reset().summary == 0.0)
    step = vec.step(_act())
    assert np.all(step.summary == 0.0)
    # window view is still computed
    assert step.window[0].tolist() == [[True, False]] * N
    assert step.window[1].tolist() == [[20, 0]] * N
    assert step.window[1].dtype == np.int32


def test_open_summary_goes_to_partner() -> None:
    vec, _ = _vec(True)
    vec.reset()
    step = vec.step(_act())
    assert np.all(step.summary[:, 0] == 0.0)
    assert np.all(step.summary[:, 1, 0] == 1.0)


def test_reset_outputs_are_not_aliased_to_later_steps() -> None:
    vec, _ = _vec(True)
    first = vec.reset()
    kept = first.obs.copy()
    win = vec.window()
    kept_win = (win[0].copy(), win[1].copy())
    stepped = vec.step(_act())  # agent 1 frame changes (0 -> 1)
    assert not np.array_equal(stepped.obs, kept)
    assert np.array_equal(first.obs, kept)
    assert np.array_equal(win[0], kept_win[0])
    assert np.array_equal(win[1], kept_win[1])


def test_autoreset_only_touches_completed_pairs() -> None:
    created: list[_Scripted] = []

    def factory(rand_seed: int) -> _Scripted:
        n = len(created)
        if n >= 4:  # post-reset episodes: no coins
            coins: set[int] = set()
        else:
            pair, agent = divmod(n, 2)
            coins = {0} if agent == 0 else {1 if pair == 0 else 5}
        env = _Scripted(rand_seed, coins)
        created.append(env)
        return env

    vec = VecCoopEnv(
        2, run_seed_sequence(0, "gtm", "registered"), True, env_factory=factory
    )
    vec.step(_act(2))
    s = vec.step(_act(2))
    assert s.done.tolist() == [True, False]
    assert np.all(s.obs[0] == 0)  # pair 0: fresh episode
    assert np.all(s.obs[1, 1] == 2)  # pair 1 agent 1 still playing
    assert np.all(s.obs[1, 0] == 0)  # pair 1 agent 0 frozen
    assert s.window[0].tolist() == [[False, False], [True, False]]
    assert s.window[1].tolist() == [[0, 0], [19, 0]]
    assert np.all(s.summary[0] == 0.0)
    assert s.summary[1, 1, 0] == 1.0


def _rand_vec(n_workers: int, n_pairs: int = 6, ts: bool = True,
              factory=None):
    from tests.unit.n10c.scripted_env import rand_factory

    return VecCoopEnv(
        n_pairs, run_seed_sequence(5, "gtm", "registered"), ts,
        env_factory=factory or rand_factory, n_workers=n_workers,
    )


def _trace(n_workers: int) -> list:
    rng = np.random.default_rng(0)
    vec = _rand_vec(n_workers)
    try:
        first = vec.reset()
        out = [first.obs, first.summary, *vec.window()]
        for _ in range(50):
            a = rng.integers(0, 15, (6, 2)).astype(np.int32)
            s = vec.step(a)
            out += [s.obs, s.summary, s.team_reward, s.done, s.finished,
                    *s.window, *vec.window()]
        return out
    finally:
        vec.close()


def test_vec_env_identical_across_worker_counts() -> None:
    ref = _trace(1)
    again = _trace(1)  # repeatable in-process: the complete trace
    assert len(again) == len(ref)
    assert all(np.array_equal(a, b) for a, b in zip(ref, again, strict=True))
    # ref layout: 4 reset arrays, then 9 per step
    steps = [ref[4 + 9 * i: 4 + 9 * (i + 1)] for i in range(50)]
    done = np.stack([x[3] for x in steps])
    finished = np.stack([x[4] for x in steps])
    assert done.any()  # auto-resets happen
    assert (finished & ~done[..., None]).any()  # frozen agents occur
    for n_workers in (2, 4):
        got = _trace(n_workers)
        assert len(got) == len(ref)
        for a, b in zip(ref, got, strict=True):
            assert a.dtype == b.dtype and np.array_equal(a, b)


def test_vec_env_workers_closed_on_error() -> None:
    from tests.unit.n10c.scripted_env import failing_factory

    vec = _rand_vec(2, factory=failing_factory)
    procs = list(vec._procs)
    assert procs and all(p.is_alive() for p in procs)
    vec.reset()
    with pytest.raises(RuntimeError, match="scripted env failure"):
        for _ in range(6):
            vec.step(np.zeros((6, 2), dtype=np.int32))
    assert not any(p.is_alive() for p in procs)
    vec.close()  # idempotent


def test_vec_env_workers_closed_on_close() -> None:
    vec = _rand_vec(3)
    procs = list(vec._procs)
    vec.reset()
    vec.close()
    assert not any(p.is_alive() for p in procs)


def _shm_gone(name: str) -> bool:
    from multiprocessing import shared_memory

    try:
        shared_memory.SharedMemory(name=name).close()
    except FileNotFoundError:
        return True
    return False


class _Tracking:
    """Closure factory recording envs; fails on the ``fail_on``-th call."""

    def __init__(self, fail_on: int | None = None, fail_t: int | None = None,
                 fail_obs_on: int | None = None):
        self.fail_obs_on = fail_obs_on
        self.made: list[_Scripted] = []
        self.closed: list[_Scripted] = []
        self.fail_on, self.fail_t = fail_on, fail_t

    def __call__(self, rand_seed: int):
        if self.fail_on is not None and len(self.made) == self.fail_on:
            raise RuntimeError("ctor failure")
        outer = self
        t_fail = self.fail_t
        obs_fail = self.fail_obs_on == len(self.made)

        class E(_Scripted):
            def act(self, a):
                super().act(a)
                if t_fail is not None and self.t == t_fail:
                    raise RuntimeError("step failure")

            def observe(self):
                if obs_fail and self.t == -1:
                    raise RuntimeError("observe failure")
                return super().observe()

            def close(self):
                outer.closed.append(self)

        env = E(rand_seed, set())
        self.made.append(env)
        return env


def test_inprocess_ctor_failure_closes_earlier_envs() -> None:
    f = _Tracking(fail_on=5)
    with pytest.raises(RuntimeError, match="ctor failure"):
        VecCoopEnv(4, run_seed_sequence(0, "gtm", "registered"), True,
                   env_factory=f)
    assert len(f.made) == 5
    assert {id(e) for e in f.made} <= {id(e) for e in f.closed}


def test_inprocess_step_failure_closes_envs() -> None:
    f = _Tracking(fail_t=1)
    vec = VecCoopEnv(2, run_seed_sequence(0, "gtm", "registered"), True,
                     env_factory=f)
    vec.reset()
    with pytest.raises(RuntimeError, match="step failure"):
        for _ in range(3):
            vec.step(_act(2))
    assert {id(e) for e in f.made} <= {id(e) for e in f.closed}


def test_inprocess_reset_failure_closes_envs() -> None:
    f = _Tracking()
    vec = VecCoopEnv(2, run_seed_sequence(0, "gtm", "registered"), True,
                     env_factory=f)
    f.fail_on = len(f.made) + 1  # second env of the reset fails
    with pytest.raises(RuntimeError, match="ctor failure"):
        vec.reset()
    assert {id(e) for e in f.made} <= {id(e) for e in f.closed}


@pytest.mark.parametrize("which", [0, 1, 3])
def test_initial_observe_failure_closes_every_env(which) -> None:
    f = _Tracking(fail_obs_on=which)
    with pytest.raises(RuntimeError, match="observe failure"):
        VecCoopEnv(2, run_seed_sequence(0, "gtm", "registered"), True,
                   env_factory=f)
    assert {id(e) for e in f.made} <= {id(e) for e in f.closed}


def test_shm_unlinked_after_close_and_failures(monkeypatch) -> None:
    from multiprocessing import shared_memory

    from experiments.procgen_multiagent_gtm_comm.harness import vec_env
    from tests.unit.n10c.scripted_env import (
        failing_factory,
        init_failing_factory,
    )

    names: list[str] = []

    class Rec(shared_memory.SharedMemory):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if k.get("create"):
                names.append(self.name)

    class NS:
        SharedMemory = Rec

    monkeypatch.setattr(vec_env, "shared_memory", NS)
    vec = _rand_vec(2)
    vec.reset()
    vec.close()
    with pytest.raises(RuntimeError, match="scripted init failure"):
        _rand_vec(2, factory=init_failing_factory)
    vec = _rand_vec(2, factory=failing_factory)
    vec.reset()
    with pytest.raises(RuntimeError, match="scripted env failure"):
        for _ in range(6):
            vec.step(np.zeros((6, 2), dtype=np.int32))
    assert len(names) == 3
    assert all(_shm_gone(n) for n in names)


def test_n_pairs_validated() -> None:
    with pytest.raises(ValueError):
        VecCoopEnv(0, run_seed_sequence(0, "gtm", "registered"), True)
