"""Vectorised CoinRun-coop pairs for the N10-C MLX harness (numpy only)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.procgen_multiagent_gtm_comm.coop_env import (
    TEAM_SUMMARY_DIM,
    ProcgenCoopEnv,
)


@dataclass
class PairObs:
    obs: np.ndarray  # uint8 [n, 2, 64, 64, 3]
    summary: np.ndarray  # float32 [n, 2, 2], zeros when the channel is closed


@dataclass
class PairStep:
    """One vectorised step.

    ``done`` and ``finished`` describe the episode that just ran.
    ``obs``, ``summary`` and ``window`` describe the state to act on
    next: for a pair with ``done`` they belong to the auto-reset new
    episode (so its summary and window are zero).
    """

    obs: np.ndarray
    summary: np.ndarray
    team_reward: np.ndarray  # float32 [n]
    done: np.ndarray  # bool [n]
    finished: np.ndarray  # bool [n, 2]
    window: tuple[np.ndarray, np.ndarray]  # (in_window bool, remaining int32)


class VecCoopEnv:
    """``n_pairs`` independent ProcgenCoopEnv pairs, auto-resetting.

    Pair ``k`` is seeded from ``base_seed_seq.spawn(n_pairs)[k]``.
    """

    def __init__(
        self,
        n_pairs: int,
        base_seed_seq: np.random.SeedSequence,
        team_summary: bool,
        num_levels: int = 16,
        start_level: int = 0,
        env_factory: Callable[[int], Any] | None = None,
    ) -> None:
        self.n_pairs = n_pairs
        self.team_summary = team_summary
        self.envs = [
            ProcgenCoopEnv(
                seed=int(child.generate_state(1)[0] % (2**31 - 1)),
                team_summary=team_summary,
                num_levels=num_levels,
                start_level=start_level,
                env_factory=env_factory,
            )
            for child in base_seed_seq.spawn(n_pairs)
        ]
        self._obs = np.stack([np.stack(e._last_obs) for e in self.envs])

    def _summary(self) -> np.ndarray:
        out = np.zeros((self.n_pairs, 2, TEAM_SUMMARY_DIM), dtype=np.float32)
        if self.team_summary:
            for k, env in enumerate(self.envs):
                out[k] = np.stack(env.team_summaries())
        return out

    def _window(self) -> tuple[np.ndarray, np.ndarray]:
        flag = np.zeros((self.n_pairs, 2), dtype=bool)
        rem = np.zeros((self.n_pairs, 2), dtype=np.int32)
        for k, env in enumerate(self.envs):
            for i in range(2):
                flag[k, i], rem[k, i] = env.window_view(i)
        return flag, rem

    def reset(self) -> PairObs:
        self._obs = np.stack([np.stack(e.reset()) for e in self.envs])
        return PairObs(self._obs.copy(), self._summary())

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """Current window view (state to act on), for the reset obs."""
        return self._window()

    def step(self, actions: np.ndarray) -> PairStep:
        assert actions.shape == (self.n_pairs, 2)
        team = np.zeros(self.n_pairs, dtype=np.float32)
        done = np.zeros(self.n_pairs, dtype=bool)
        finished = np.zeros((self.n_pairs, 2), dtype=bool)
        for k, env in enumerate(self.envs):
            res = env.step([int(actions[k, 0]), int(actions[k, 1])])
            team[k] = res.team_reward
            done[k] = res.done
            finished[k] = res.info["finished"]
            if res.done:
                self._obs[k] = np.stack(env.reset())
            else:
                self._obs[k] = np.stack(res.obs_per_agent)
        return PairStep(
            obs=self._obs.copy(),
            summary=self._summary(),
            team_reward=team,
            done=done,
            finished=finished,
            window=self._window(),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
