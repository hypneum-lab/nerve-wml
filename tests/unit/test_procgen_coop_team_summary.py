"""Declared "team summary" side channel of ProcgenCoopEnv (N10-C).

The procgen boundary is replaced by scripted envs (via ``env_factory``)
so each agent's coin timing and episode end are controlled exactly.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.procgen_multiagent_gtm_comm.coop_env import (
    TEAM_SUMMARY_DIM,
    ProcgenCoopEnv,
)

DELTA_T = 4


class _ScriptedEnv:
    """gym3-like env with scripted coins (reward 10) and deaths.

    ``coin_steps`` / ``death_steps`` are 0-based step indices. Like
    procgen CoinRun, a coin or a death ends the env's own episode
    (``first`` is True). Observations are ``{"rgb": [1, 64, 64, 3]}``
    as in the real procgen gym3 API, filled with the step index.
    """

    def __init__(self, coin_steps: set[int], death_steps: set[int]) -> None:
        self.coin_steps = coin_steps
        self.death_steps = death_steps
        self.t = -1
        self.rand_seed: int | None = None

    def act(self, actions: np.ndarray) -> None:
        self.t += 1

    def observe(self) -> tuple[np.ndarray, dict, np.ndarray]:
        coin = self.t in self.coin_steps
        end = coin or self.t in self.death_steps
        img = np.full((1, 64, 64, 3), (self.t + 1) % 256, dtype=np.uint8)
        return np.array([10.0 if coin else 0.0]), {"rgb": img}, np.array([end])

    def close(self) -> None:
        pass


def _env(
    team_summary: bool,
    coins: tuple[set[int], set[int]],
    deaths: tuple[set[int], set[int]] = (set(), set()),
) -> tuple[ProcgenCoopEnv, list[_ScriptedEnv]]:
    created: list[_ScriptedEnv] = []

    def factory(rand_seed: int) -> _ScriptedEnv:
        i = len(created) % 2
        env = _ScriptedEnv(coins[i], deaths[i])
        env.rand_seed = rand_seed
        created.append(env)
        return env

    env = ProcgenCoopEnv(
        seed=0, delta_t=DELTA_T, team_summary=team_summary, env_factory=factory
    )
    return env, created


def test_closed_channel_returns_no_summary() -> None:
    env, _ = _env(False, ({0}, set()))
    result = env.step([0, 0])
    assert result.summary_per_agent is None
    assert env.team_summaries() is None


def test_open_channel_is_zero_before_any_coin() -> None:
    env, _ = _env(True, (set(), set()))
    result = env.step([0, 0])
    assert result.summary_per_agent is not None
    for vec in result.summary_per_agent:
        assert vec.shape == (TEAM_SUMMARY_DIM,)
        assert np.all(vec == 0.0)


def test_agents_share_the_level_seed() -> None:
    env, created = _env(True, (set(), set()))
    assert created[0].rand_seed == created[1].rand_seed
    env.reset()
    assert created[2].rand_seed == created[3].rand_seed
    assert created[2].rand_seed != created[0].rand_seed


def test_partner_sees_first_coin_and_window_shrinks() -> None:
    env, _ = _env(True, ({0}, set()))
    first = env.step([0, 0]).summary_per_agent
    # Agent 0 hit the coin: only its partner (agent 1) is informed.
    assert np.all(first[0] == 0.0)
    assert first[1][0] == 1.0
    assert first[1][1] == 1.0
    second = env.step([0, 0]).summary_per_agent
    assert second[1][0] == 1.0
    assert second[1][1] < first[1][1]


def test_fraction_positive_exactly_while_partner_can_score() -> None:
    # For every partner coin step k, the summary seen before step k is
    # > 0 iff a coin at step k still earns the team reward.
    for k in range(1, DELTA_T + 3):
        env, _ = _env(True, ({0}, {k}))
        rewarded = False
        for _ in range(k):
            result = env.step([0, 0])
            summary = result.summary_per_agent
            if result.done:
                break
        else:
            rewarded = env.step([0, 0]).team_reward == 1.0
        assert (summary[1][1] > 0.0) == rewarded, k


def test_first_coin_does_not_end_episode_partner_can_act() -> None:
    env, _ = _env(True, ({0}, {2}))
    r0 = env.step([0, 0])
    # The coin ends agent 0's own procgen episode but not the team's.
    assert not r0.done
    assert r0.info["finished"] == [True, False]
    assert r0.summary_per_agent[1][0] == 1.0
    assert not env.step([0, 0]).done
    r2 = env.step([0, 0])
    assert r2.team_reward == 1.0
    assert r2.done


def test_finished_agent_is_frozen() -> None:
    env, _ = _env(True, ({0}, set()))
    obs0 = env.reset()[0].copy()
    r0 = env.step([0, 0])
    # Agent 0's frame stays at its last pre-termination frame.
    assert np.array_equal(r0.obs_per_agent[0], obs0)
    r1 = env.step([0, 0])
    assert np.array_equal(r1.obs_per_agent[0], obs0)
    assert r1.individual_rewards[0] == 0.0


def test_window_expiry_ends_episode_and_clears_summary() -> None:
    env, _ = _env(True, ({0}, set()))
    results = [env.step([0, 0]) for _ in range(DELTA_T + 1)]
    assert all(not r.done for r in results[:-1])
    assert results[-1].done
    assert np.all(results[-1].summary_per_agent[1] == 0.0)


def test_death_without_coin_ends_episode() -> None:
    env, _ = _env(True, ({0}, set()), deaths=(set(), {1}))
    assert not env.step([0, 0]).done
    result = env.step([0, 0])
    assert result.done
    assert result.team_reward == 0.0


def test_success_is_the_only_reward() -> None:
    env, _ = _env(False, ({0}, {DELTA_T}))
    rewards = []
    while True:
        result = env.step([0, 0])
        rewards.append(result.team_reward)
        if result.done:
            break
    assert rewards == [0.0] * DELTA_T + [1.0]


def test_summary_is_reset_with_the_episode() -> None:
    env, _ = _env(True, ({0}, set()))
    result = env.step([0, 0])
    assert result.summary_per_agent[1][0] == 1.0
    env.reset()
    for vec in env.team_summaries():
        assert np.all(vec == 0.0)


def test_channel_does_not_change_reward_or_images() -> None:
    open_env, _ = _env(True, ({0}, {2}))
    closed_env, _ = _env(False, ({0}, {2}))
    for _ in range(3):
        a = open_env.step([0, 0])
        b = closed_env.step([0, 0])
        assert a.team_reward == b.team_reward
        assert a.done == b.done
        assert a.individual_rewards == b.individual_rewards
        for x, y in zip(a.obs_per_agent, b.obs_per_agent, strict=True):
            assert np.array_equal(x, y)


def test_stub_is_rejected_unless_allowed(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_procgen(name, *args, **kwargs):
        if name == "procgen":
            raise ImportError("procgen hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_procgen)
    with pytest.raises(RuntimeError, match="procgen"):
        ProcgenCoopEnv(seed=0)
    env = ProcgenCoopEnv(seed=0, allow_stub=True)
    assert env.using_stub


def test_window_view_matches_summary() -> None:
    env, _ = _env(True, ({0}, set()))
    assert env.window_view(0) == (False, 0)
    for _ in range(DELTA_T):
        env.step([0, 0])
        summary = env.team_summaries()[1]
        assert env.window_view(0) == (
            bool(summary[0] == 1.0),
            round(float(summary[1]) * DELTA_T),
        )
        assert env.window_view(0)[0]
        assert env.window_view(1) == (False, 0)
    assert env.window_view(0) == (True, 1)
    env.step([0, 0])  # window expires
    assert env.window_view(0) == (False, 0)


def test_window_view_available_when_closed() -> None:
    env, _ = _env(False, ({0}, set()))
    assert env.team_summaries() is None
    env.step([0, 0])
    assert env.window_view(0) == (True, DELTA_T)
    assert env.window_view(1) == (False, 0)
    env.reset()
    assert env.window_view(0) == (False, 0)
