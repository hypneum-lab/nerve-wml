"""ProcgenCoopEnv — 2-agent cooperative variant wrapper.

Wraps two single-agent procgen instances (e.g. CoinRun) into a Python
multi-agent environment with a shared team reward signal.

For CoinRun-coop: team reward = +1 on the step where both agents have
reached a coin within DELTA_T timesteps of each other, else 0. There
is no step penalty: with the old -0.1 per step, a team succeeding at
step 13 or later scored below a team whose agent died at step 1 (N10-C
2026-10-23 amendment). Completion time is reported separately. Both
agents play the same level (same procgen level seed) in separate
instances, so neither agent's image shows its partner. Each agent
observes its own raw image. With ``team_summary=True`` it also gets a
declared low-dim "team summary" side channel about its partner (see
``TEAM_SUMMARY_DIM``) ; ``team_summary=False`` is the
side-channel-closed control of the N10-C 2026-10-23 amendment.

Procgen does not expose agent or coin positions, so the summary
carries the partner's coin timing rather than a distance-to-coin
estimate: [partner reached its coin inside the open DELTA_T window
(0/1), fraction of the window's scoring steps still remaining
(0..1)]. The fraction is > 0 exactly while a coin by this agent on a
later step would still earn the team reward.

Episode semantics (N10-C 2026-10-23 amendment). In CoinRun an agent's
own procgen episode ends when it collects its coin, dies or times
out. Such an agent is *finished*: it is frozen (its actions are
ignored, its observation stays at its last pre-termination frame and
its individual reward is 0) while its partner keeps playing. The
cooperative episode ends (``done=True``) on the first of:

* the team reward is earned ;
* the DELTA_T window opened by the first coin expires ;
* an agent finishes without a coin (the team reward is then
  unreachable) ;
* every agent has finished.

The team summary is part of the step observation, computed after the
reward update of that step. A summary returned with ``done=False``
can be acted on at the next step ; one returned with ``done=True`` is
a terminal observation. All window state, and hence the summary, is
reset by ``reset()``.

Scientific runs must use real procgen: the stub fallback is only
available with ``allow_stub=True`` (smoke tests on platforms without
procgen) and is flagged by ``using_stub``.

Plan: HYPNEUM-PLANS/2026-05-11-niveau10-scaling-up.md (N10-C kickoff).
Pre-reg: docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md.
Arch sketch: HYPNEUM-PLANS/2026-05-11-n10c-architecture-sketch.md.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

TEAM_SUMMARY_DIM = 2


@dataclass
class CoopStepResult:
    obs_per_agent: list[np.ndarray]
    team_reward: float
    individual_rewards: list[float]
    done: bool
    info: dict
    # Per-agent team-summary vectors (shape [TEAM_SUMMARY_DIM]) when the
    # side channel is open, else None. Part of the step observation.
    summary_per_agent: list[np.ndarray] | None = None


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
        Base seed. The level seed of cooperative episode ``k`` is
        derived from ``(seed, k)`` and is shared by all agents.
    team_summary : bool
        Open the declared "team summary" side channel (default False,
        the side-channel-closed condition).
    num_levels, start_level : int
        Procgen level-set parameters (train / held-out split).
    allow_stub : bool
        Fall back to a random stub when procgen is not installed.
        Never set in scientific runs.
    env_factory : callable | None
        ``env_factory(rand_seed) -> env`` with the gym3 ``act`` /
        ``observe`` / ``close`` interface. Overrides procgen (tests).
    """

    def __init__(
        self,
        env_name: str = "coinrun",
        n_agents: int = 2,
        delta_t: int = 20,
        seed: int | None = None,
        team_summary: bool = False,
        num_levels: int = 0,
        start_level: int = 0,
        allow_stub: bool = False,
        env_factory: Callable[[int], Any] | None = None,
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
        if delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {delta_t}")
        self.env_name = env_name
        self.n_agents = n_agents
        self.delta_t = delta_t
        self.seed = seed
        self.team_summary = team_summary
        self.num_levels = num_levels
        self.start_level = start_level
        self.allow_stub = allow_stub
        self.using_stub = False
        self._env_factory = env_factory
        self._episode = -1
        self._envs: list[Any] = []  # underlying procgen envs
        self._finished: list[bool] = []
        self._last_obs: list[np.ndarray] = []
        self._steps_since_first_coin: int | None = None
        self._first_coin_agent: int | None = None
        self._reset_envs()

    def _level_seed(self) -> int:
        seq = np.random.SeedSequence([self.seed or 0, self._episode])
        return int(seq.generate_state(1)[0] % (2**31 - 1))

    def _make_env(self, rand_seed: int) -> Any:
        if self._env_factory is not None:
            return self._env_factory(rand_seed)
        try:
            from procgen import ProcgenGym3Env
        except ImportError as exc:
            if not self.allow_stub:
                raise RuntimeError(
                    "procgen is not installed ; scientific N10-C runs "
                    "require real procgen (pass allow_stub=True only "
                    "for smoke tests)."
                ) from exc
            self.using_stub = True
            return _StubProcgenEnv(seed=rand_seed)
        return ProcgenGym3Env(
            num=1,
            env_name=self.env_name,
            rand_seed=rand_seed,
            num_levels=self.num_levels,
            start_level=self.start_level,
            distribution_mode="easy",
        )

    def _reset_envs(self) -> None:
        self.close()
        self._episode += 1
        rand_seed = self._level_seed()
        # Same rand_seed for every agent: both play the same level.
        self._envs = []
        try:
            for _ in range(self.n_agents):
                self._envs.append(self._make_env(rand_seed))
        except BaseException:
            self.close()
            raise
        self._finished = [False] * self.n_agents
        self._steps_since_first_coin = None
        self._first_coin_agent = None
        self._last_obs = [self._read_obs(env) for env in self._envs]

    def reset(self) -> list[np.ndarray]:
        self._reset_envs()
        return list(self._last_obs)

    def step(self, actions: list[int]) -> CoopStepResult:
        assert len(actions) == self.n_agents, "actions per-agent count mismatch"
        individual_rewards: list[float] = [0.0] * self.n_agents
        coin_flags: list[bool] = [False] * self.n_agents
        newly_finished_without_coin = False
        for i, env in enumerate(self._envs):
            if self._finished[i]:
                continue  # frozen: action ignored, obs held
            env.act(np.array([actions[i]], dtype=np.int32))
            r, obs, first = env.observe()
            individual_rewards[i] = float(r[0])
            # CoinRun: reward = 10 when coin reached
            if individual_rewards[i] > 0.5:
                coin_flags[i] = True
            if coin_flags[i] or bool(first[0]):
                self._finished[i] = True
                if not coin_flags[i]:
                    newly_finished_without_coin = True
            else:
                # On episode end gym3 returns the next level's first
                # frame ; a finished agent keeps its last frame.
                self._last_obs[i] = self._to_image(obs)

        # Team reward logic (no step penalty, see module docstring)
        team_reward = 0.0
        window_expired = False
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
                window_expired = True

        done = (
            team_reward == 1.0
            or window_expired
            or newly_finished_without_coin
            or all(self._finished)
        )
        return CoopStepResult(
            obs_per_agent=list(self._last_obs),
            team_reward=team_reward,
            individual_rewards=individual_rewards,
            done=done,
            info={"coin_flags": coin_flags, "finished": list(self._finished)},
            summary_per_agent=self.team_summaries(),
        )

    def team_summaries(self) -> list[np.ndarray] | None:
        """Per-agent partner summary, or None when the channel is closed.

        For agent i: ``[1, f]`` if its partner reached a coin inside
        the currently open DELTA_T window, else ``[0, 0]``. ``f`` is
        the number of later steps on which a coin by agent i still
        earns the team reward, divided by DELTA_T: it is in
        ``(0, 1]`` exactly while that is possible.
        """
        if not self.team_summary:
            return None
        out: list[np.ndarray] = []
        for i in range(self.n_agents):
            vec = np.zeros(TEAM_SUMMARY_DIM, dtype=np.float32)
            first = self._first_coin_agent
            elapsed = self._steps_since_first_coin
            if first is not None and first != i and elapsed is not None:
                # A partner coin on the next step is rewarded iff
                # elapsed <= delta_t, which the window state guarantees.
                remaining = self.delta_t - elapsed + 1
                vec[0] = 1.0
                vec[1] = remaining / self.delta_t
            out.append(vec)
        return out

    def window_view(self, i: int) -> tuple[bool, int]:
        """Agent ``i``'s own coin timing inside the open DELTA_T window.

        Returns ``(reached, remaining)``: whether agent ``i`` reached
        its coin inside the currently open window, and the number of
        later steps (0..DELTA_T) on which its partner's coin still
        earns the team reward. This is exactly what the partner's
        team summary carries, computed whether or not ``team_summary``
        is open. It is never delivered as an observation (the
        engineered reference sender reads it, see the amendment).
        """
        elapsed = self._steps_since_first_coin
        if self._first_coin_agent != i or elapsed is None:
            return False, 0
        return True, self.delta_t - elapsed + 1

    @staticmethod
    def _to_image(obs: Any) -> np.ndarray:
        # Real procgen gym3 envs return {"rgb": [1, H, W, 3]} ; the
        # stub returns the array directly.
        if isinstance(obs, dict):
            obs = obs["rgb"]
        return np.squeeze(np.asarray(obs), axis=0)

    def _read_obs(self, env: Any) -> np.ndarray:
        _, obs, _ = env.observe()
        return self._to_image(obs)

    def close(self) -> None:
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
        self._envs = []


class _StubProcgenEnv:
    """Fallback stub if procgen unavailable on this platform.

    Returns random 64x64x3 image observations + rare coin rewards.
    Smoke tests only (``allow_stub=True``).
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
