"""Backend-neutral interfaces for the N10-D environment (stubs).

L0 DRAFT, not frozen (amended 2026-11-06 after the blind contradictor pass,
reconciled with PR #100 the same day, protocol section 8).
A backend (JaxMARL or EGG) implements ``RefGame``; the harness only sees
these Protocols, so the choice does not change the analysis.
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
        """Return (next_obs, team_reward, done); action = candidate index in [0, C).

        Message length L of the learned arm: TBD-11.
        """


class Sender(Protocol):
    def emit(self, speaker_obs: Any) -> Sequence[int]: ...


class Channel(Protocol):
    """Arm-specific transform between emit and listener input, per batch."""

    arm: Arm

    def transmit(
        self, messages: Sequence[Sequence[int]], *, batch_seed: int
    ) -> list[Sequence[int]]:
        """Map the messages of one batch (one per episode) to listener inputs.

        eng: identity of the engineered code; nocomm: zeros of the same length;
        msg_shuffle (evaluation only, on the trained eng pair, A9/A14):
        ``messages[perm[i]]`` for episode i of the evaluation batch, with
        ``perm = derange_episodes(len(messages), random.Random(batch_seed))``.
        It permutes EPISODES, never target identities: a fixed map on
        identities would keep the message a deterministic function of the
        target, and a listener could learn it up to 1.0.
        """


def derange_episodes(n_episodes: int, rng: random.Random) -> list[int]:
    """Uniform random derangement of episode indices (no fixed point).

    Each episode receives the message emitted for another episode of the same
    batch; the marginal of messages and their length are kept. No label
    exclusion: the source episode may share the current target (probability
    about 1/K, not a leak). An eng-trained listener that guesses uniformly
    when the message names no candidate then scores 1/K + (1 - C/K) / C
    = 1/C in expectation: the blind ceiling. Rejection sampling from uniform
    permutations (about e tries).
    """
    if n_episodes < 2:
        raise ValueError("a derangement needs at least 2 episodes")
    perm = list(range(n_episodes))
    while True:
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            return list(perm)


class Trainer(Protocol):
    def train(self, game: RefGame, channel: Channel, seed: int, steps: int) -> Any:
        """Train and return the FROZEN policy pair (no success read here).

        The former last-5-logs training summary is not a metric (A8).
        """

    def evaluate(
        self, policy: Any, game: RefGame, channel: Channel, seed: int, n_eval: int,
    ) -> tuple[int, int]:
        """Primary metric (A8): the frozen policy on n_eval = 500 fresh episodes
        (fresh targets and candidate orders, episode seeds disjoint from the
        training stream, no learning). Return (successes, n_eval); S is
        successes / n_eval, a Binomial(n_eval, p) / n_eval draw given the seed."""
