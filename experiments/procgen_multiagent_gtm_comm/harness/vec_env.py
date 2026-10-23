"""Vectorised CoinRun-coop pairs for the N10-C MLX harness (numpy only)."""
from __future__ import annotations

import multiprocessing as mp
import traceback
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


class _Block:
    """A contiguous block of pairs, stepped in the calling process."""

    def __init__(
        self,
        seeds: list[int],
        team_summary: bool,
        num_levels: int,
        start_level: int,
        env_factory: Callable[[int], Any] | None,
    ) -> None:
        self.n = len(seeds)
        self.team_summary = team_summary
        self.envs = [
            ProcgenCoopEnv(
                seed=seed,
                team_summary=team_summary,
                num_levels=num_levels,
                start_level=start_level,
                env_factory=env_factory,
            )
            for seed in seeds
        ]
        self.obs = np.stack([np.stack(e._last_obs) for e in self.envs])

    def summary(self) -> np.ndarray:
        out = np.zeros((self.n, 2, TEAM_SUMMARY_DIM), dtype=np.float32)
        if self.team_summary:
            for k, env in enumerate(self.envs):
                out[k] = np.stack(env.team_summaries())
        return out

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        flag = np.zeros((self.n, 2), dtype=bool)
        rem = np.zeros((self.n, 2), dtype=np.int32)
        for k, env in enumerate(self.envs):
            for i in range(2):
                flag[k, i], rem[k, i] = env.window_view(i)
        return flag, rem

    def reset(self) -> tuple:
        self.obs = np.stack([np.stack(e.reset()) for e in self.envs])
        return (self.obs.copy(), self.summary(), *self.window())

    def step(self, actions: np.ndarray) -> tuple:
        team = np.zeros(self.n, dtype=np.float32)
        done = np.zeros(self.n, dtype=bool)
        finished = np.zeros((self.n, 2), dtype=bool)
        for k, env in enumerate(self.envs):
            res = env.step([int(actions[k, 0]), int(actions[k, 1])])
            team[k] = res.team_reward
            done[k] = res.done
            finished[k] = res.info["finished"]
            if res.done:
                self.obs[k] = np.stack(env.reset())
            else:
                self.obs[k] = np.stack(res.obs_per_agent)
        return (self.obs.copy(), self.summary(), team, done, finished,
                *self.window())

    def using_stub(self) -> bool:
        return any(getattr(e, "using_stub", False) for e in self.envs)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker_main(conn, args) -> None:
    """Worker process: own one block, serve commands over ``conn``."""
    block = None
    try:
        block = _Block(*args)
        conn.send(("ok", block.using_stub()))
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                conn.send(("ok", block.reset()))
            else:
                conn.send(("ok", block.step(payload)))
    except EOFError:
        pass
    except BaseException:  # report, then exit
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass
    finally:
        if block is not None:
            try:
                block.close()
            except Exception:
                pass
        conn.close()


class VecCoopEnv:
    """``n_pairs`` independent ProcgenCoopEnv pairs, auto-resetting.

    Pair ``k`` is seeded from ``base_seed_seq.spawn(n_pairs)[k]``.
    With ``n_workers > 1`` the pairs are split into contiguous blocks,
    each stepped in its own spawn-context process (``env_factory`` must
    then be picklable). Results never depend on ``n_workers``.
    """

    def __init__(
        self,
        n_pairs: int,
        base_seed_seq: np.random.SeedSequence,
        team_summary: bool,
        num_levels: int = 16,
        start_level: int = 0,
        env_factory: Callable[[int], Any] | None = None,
        n_workers: int = 1,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self.n_pairs = n_pairs
        self.team_summary = team_summary
        seeds = [
            int(child.generate_state(1)[0] % (2**31 - 1))
            for child in base_seed_seq.spawn(n_pairs)
        ]
        n_workers = min(n_workers, n_pairs)
        self._block: _Block | None = None
        self._procs: list[Any] = []
        self._conns: list[Any] = []
        self._stub = False
        # contiguous blocks, sizes differ by at most one
        edges = np.linspace(0, n_pairs, n_workers + 1).astype(int)
        self._slices = [
            slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)
        ]
        if n_workers == 1:
            self._block = _Block(
                seeds, team_summary, num_levels, start_level, env_factory
            )
            self.envs = self._block.envs
            self._win = self._block.window()
            return
        self.envs = []
        ctx = mp.get_context("spawn")
        try:
            for sl in self._slices:
                parent, child = ctx.Pipe()
                args = (seeds[sl], team_summary, num_levels, start_level,
                        env_factory)
                p = ctx.Process(target=_worker_main, args=(child, args),
                                daemon=True)
                p.start()
                child.close()
                self._procs.append(p)
                self._conns.append(parent)
            self._stub = any(self._recv(c) for c in self._conns)
            self._win = (np.zeros((n_pairs, 2), dtype=bool),
                         np.zeros((n_pairs, 2), dtype=np.int32))
        except BaseException:
            self.close()
            raise

    def _recv(self, conn):
        try:
            tag, payload = conn.recv()
        except EOFError as exc:
            raise RuntimeError("env worker died") from exc
        if tag == "err":
            raise RuntimeError(f"env worker failed:\n{payload}")
        return payload

    def _rpc(self, cmd: str, actions: np.ndarray | None = None) -> list:
        try:
            for conn, sl in zip(self._conns, self._slices, strict=True):
                conn.send((cmd, None if actions is None else actions[sl]))
            parts = [self._recv(c) for c in self._conns]
        except BaseException:
            self.close()
            raise
        return [np.concatenate(f) for f in zip(*parts, strict=True)]

    def using_stub(self) -> bool:
        if self._block is not None:
            return self._block.using_stub()
        return self._stub

    def reset(self) -> PairObs:
        if self._block is not None:
            obs, summ, *self._win = self._block.reset()
            self._win = tuple(self._win)
        else:
            obs, summ, *win = self._rpc("reset")
            self._win = tuple(win)
        return PairObs(obs, summ)

    def window(self) -> tuple[np.ndarray, np.ndarray]:
        """Current window view (state to act on), for the reset obs."""
        return self._win[0].copy(), self._win[1].copy()

    def step(self, actions: np.ndarray) -> PairStep:
        assert actions.shape == (self.n_pairs, 2)
        if self._block is not None:
            res = self._block.step(actions)
        else:
            res = self._rpc("step", np.ascontiguousarray(actions))
        obs, summ, team, done, finished, flag, rem = res
        self._win = (flag, rem)
        return PairStep(
            obs=obs, summary=summ, team_reward=team, done=done,
            finished=finished, window=(flag.copy(), rem.copy()),
        )

    def close(self) -> None:
        if self._block is not None:
            self._block.close()
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._procs = []
        self._conns = []
