"""Backend-neutral interfaces for the N10-D environment (stubs).

A backend (JaxMARL or EGG, TBD by PI) implements ``RefGame``; the harness
only sees these Protocols, so the choice does not change the analysis.
The environment itself is specified in docs/prereg/n10d/protocol.md,
section 7 (A11).
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Protocol

from .spec import Arm, TaskSpec


class RefGame(Protocol):
    spec: TaskSpec

    def reset(self, seed: int) -> Any:
        """Return the initial observation pair (speaker_obs, listener_obs)."""

    def step(self, message: Sequence[int], action: int) -> tuple[Any, float, bool]:
        """Return (next_obs, team_reward, done); action = candidate index in [0, C)."""


class Sender(Protocol):
    def emit(self, speaker_obs: Any) -> Sequence[int]: ...


class Channel(Protocol):
    """Arm-specific transform between emit and listener input."""

    arm: Arm

    def transmit(self, message: Sequence[int], *, episode: int) -> Sequence[int]:
        """eng: identity of the engineered code; nocomm: zeros;
        msg_shuffle (evaluation only, A9/A14): the message emitted in episode
        ``derangement[episode]`` of the same evaluation batch, where the
        derangement comes from ``derange_episodes``. It permutes EPISODES,
        never target identities: a fixed map on identities would keep the
        message a deterministic function of the target."""


def derange_episodes(n_episodes: int, rng: random.Random) -> list[int]:
    """Uniform random derangement of episode indices (no fixed point).

    No label exclusion: the source episode may share the current target
    (probability about 1/K). An eng-trained listener that guesses uniformly
    when the message names no candidate then scores 1/K + (1 - C/K) / C
    = 1/C in expectation: the blind ceiling.
    """
    if n_episodes < 2:
        raise ValueError("a derangement needs at least 2 episodes")
    perm = list(range(n_episodes))
    while True:
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            return perm


class Trainer(Protocol):
    def train(self, game: RefGame, channel: Channel, seed: int, steps: int) -> Any:
        """Train and return the FROZEN policy pair (no success read here)."""

    def evaluate(
        self, policy: Any, game: RefGame, channel: Channel, seed: int, n_eval: int,
    ) -> float:
        """Success of the frozen policy on n_eval fresh episodes (A8)."""
