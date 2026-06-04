"""ProcgenCoopEnv — 2-agent cooperative variant wrapper.

Wraps two single-agent procgen instances (e.g. CoinRun) into a Python
multi-agent environment with a shared team reward signal.

For CoinRun-coop: team reward = +1 if both agents reach a coin within
DELTA_T timesteps of each other ; -0.1 step penalty otherwise. Each
agent observes its own raw image + a low-dim "team summary" channel
(e.g. partner agent's distance-to-coin estimate).

Plan: HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md (N10-C kickoff).
Pre-reg: docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md.
Arch sketch: HYPNEUM-PLANS/2026-05-11-n10c-architecture-sketch.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CoopStepResult:
    obs_per_agent: list[np.ndarray]
    team_reward: float
    individual_rewards: list[float]
    done: bool
    info: dict


class ProcgenCoopEnv:
    """2-agent cooperative procgen wrapper.

    Parameters
    ----------
    env_name : str
        One of "coinrun", "caveflyer", "maze" (this kickoff covers
        coinrun only ; others raise NotImplementedError).
    n_agents : int
        Number of cooperative agents (default 2).
    delta_t : int
        Time-window for "both reached coin" reward (CoinRun-coop only).
    seed : int | None
        Seed for both env instances and for the random policy in smoke.
    """

    def __init__(
        self,
        env_name: str = "coinrun",
        n_agents: int = 2,
        delta_t: int = 20,
        seed: int | None = None,
    ) -> None:
        if env_name != "coinrun":
            raise NotImplementedError(
                f"env_name={env_name!r} not supported in N10-C kickoff. "
                "Only 'coinrun' is implemented ; caveflyer + maze are "
                "future milestones."
            )
        if n_agents != 2:
            raise NotImplementedError(
                f"n_agents={n_agents} not supported in N10-C kickoff. "
                "Only n_agents=2 (CoinRun-coop) is implemented."
            )
        self.env_name = env_name
        self.n_agents = n_agents
        self.delta_t = delta_t
        self.seed = seed
        self._envs: list[Any] = []  # underlying procgen envs
        self._steps_since_first_coin: int | None = None
        self._first_coin_agent: int | None = None
        self._reset_envs()

    def _reset_envs(self) -> None:
        try:
            from procgen import ProcgenGym3Env
            self._envs = [
                ProcgenGym3Env(
                    num=1,
                    env_name=self.env_name,
                    rand_seed=(self.seed or 0) + i,
                    distribution_mode="easy",
                )
                for i in range(self.n_agents)
            ]
        except (ImportError, Exception):
            # Fallback STUB if procgen unavailable on this platform.
            # TODO: install procgen in real N10-C sprint (needs cp310 env).
            self._envs = [
                _StubProcgenEnv(seed=(self.seed or 0) + i)
                for i in range(self.n_agents)
            ]
        self._steps_since_first_coin = None
        self._first_coin_agent = None

    def reset(self) -> list[np.ndarray]:
        self._reset_envs()
        return self._observe()

    def step(self, actions: list[int]) -> CoopStepResult:
        assert len(actions) == self.n_agents, "actions per-agent count mismatch"
        individual_rewards: list[float] = []
        coin_flags: list[bool] = [False] * self.n_agents
        any_done = False
        for i, env in enumerate(self._envs):
            env.act(np.array([actions[i]], dtype=np.int32))
            r, _, first = env.observe()
            individual_rewards.append(float(r[0]))
            # CoinRun: reward = 10 when coin reached
            if individual_rewards[-1] > 0.5:
                coin_flags[i] = True
            if first[0]:
                any_done = True

        # Team reward logic
        team_reward = -0.1  # step penalty
        for i, hit in enumerate(coin_flags):
            if hit:
                if self._first_coin_agent is None:
                    self._first_coin_agent = i
                    self._steps_since_first_coin = 0
                elif (
                    i != self._first_coin_agent
                    and self._steps_since_first_coin is not None
                ):
                    if self._steps_since_first_coin <= self.delta_t:
                        team_reward = 1.0
                    self._first_coin_agent = None
                    self._steps_since_first_coin = None
        if self._steps_since_first_coin is not None:
            self._steps_since_first_coin += 1
            if self._steps_since_first_coin > self.delta_t:
                self._first_coin_agent = None
                self._steps_since_first_coin = None

        done = any_done
        return CoopStepResult(
            obs_per_agent=self._observe(),
            team_reward=team_reward,
            individual_rewards=individual_rewards,
            done=done,
            info={"coin_flags": coin_flags},
        )

    def _observe(self) -> list[np.ndarray]:
        obs_list: list[np.ndarray] = []
        for env in self._envs:
            _, obs, _ = env.observe()
            # obs shape: [1, H, W, 3] uint8 ; squeeze batch dim
            obs_list.append(np.squeeze(obs, axis=0))
        return obs_list

    def close(self) -> None:
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass


class _StubProcgenEnv:
    """Fallback stub if procgen unavailable on this platform.

    Returns dummy 64x64x3 zero-image observations + zero rewards.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def act(self, actions: np.ndarray) -> None:
        self._step += 1

    def observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # rewards [1], obs [1, 64, 64, 3], first [1]
        rewards = np.zeros((1,), dtype=np.float32)
        # Random tiny chance of coin reached, just for smoke variability
        if self._rng.random() < 0.005:
            rewards[0] = 10.0
        obs = self._rng.integers(0, 255, size=(1, 64, 64, 3), dtype=np.uint8)
        first = np.array([self._step % 1000 == 0], dtype=np.bool_)
        return rewards, obs, first

    def close(self) -> None:
        pass
