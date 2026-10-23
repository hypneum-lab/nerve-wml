"""Declared "team summary" side channel of ProcgenCoopEnv (N10-C).

The procgen boundary is replaced by scripted envs so each agent's coin
timing is controlled exactly.
"""
from __future__ import annotations

import numpy as np

from experiments.procgen_multiagent_gtm_comm.coop_env import (
    TEAM_SUMMARY_DIM,
    ProcgenCoopEnv,
)


class _ScriptedEnv:
    """Gives reward 10 (coin) on the listed step indices, else 0."""

    def __init__(self, coin_steps: set[int]) -> None:
        self.coin_steps = coin_steps
        self.t = -1

    def act(self, actions: np.ndarray) -> None:
        self.t += 1

    def observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = 10.0 if self.t in self.coin_steps else 0.0
        img = np.zeros((1, 64, 64, 3), dtype=np.uint8)
        return np.array([r]), img, np.array([False])

    def close(self) -> None:
        pass


def _env(team_summary: bool, coins_a: set[int], coins_b: set[int]):
    env = ProcgenCoopEnv(seed=0, delta_t=4, team_summary=team_summary)
    env._envs = [_ScriptedEnv(coins_a), _ScriptedEnv(coins_b)]
    return env


def test_closed_channel_returns_no_summary() -> None:
    env = _env(False, {0}, set())
    result = env.step([0, 0])
    assert result.summary_per_agent is None
    assert env.team_summaries() is None


def test_open_channel_is_zero_before_any_coin() -> None:
    env = _env(True, set(), set())
    result = env.step([0, 0])
    assert result.summary_per_agent is not None
    for vec in result.summary_per_agent:
        assert vec.shape == (TEAM_SUMMARY_DIM,)
        assert np.all(vec == 0.0)


def test_partner_sees_first_coin_and_window_shrinks() -> None:
    env = _env(True, {0}, set())
    first = env.step([0, 0]).summary_per_agent
    # Agent 0 hit the coin: only its partner (agent 1) is informed.
    assert np.all(first[0] == 0.0)
    assert first[1][0] == 1.0
    second = env.step([0, 0]).summary_per_agent
    assert second[1][0] == 1.0
    assert second[1][1] < first[1][1]


def test_summary_clears_when_partner_completes_team_coin() -> None:
    env = _env(True, {0}, {2})
    env.step([0, 0])
    env.step([0, 0])
    result = env.step([0, 0])
    assert result.team_reward == 1.0
    for vec in result.summary_per_agent:
        assert np.all(vec == 0.0)


def test_summary_clears_when_window_expires() -> None:
    env = _env(True, {0}, set())
    summaries = [env.step([0, 0]).summary_per_agent for _ in range(7)]
    assert summaries[0][1][0] == 1.0
    assert np.all(summaries[-1][1] == 0.0)


def test_channel_does_not_change_reward_or_images() -> None:
    open_env = _env(True, {0}, {2})
    closed_env = _env(False, {0}, {2})
    for _ in range(4):
        a = open_env.step([0, 0])
        b = closed_env.step([0, 0])
        assert a.team_reward == b.team_reward
        assert a.individual_rewards == b.individual_rewards
        for x, y in zip(a.obs_per_agent, b.obs_per_agent, strict=True):
            assert np.array_equal(x, y)
