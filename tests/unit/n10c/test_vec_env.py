"""VecCoopEnv: numpy-only, procgen-free (scripted envs); runs in both envs."""
from __future__ import annotations

import numpy as np

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
