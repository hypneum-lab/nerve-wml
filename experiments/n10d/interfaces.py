"""Backend-neutral interfaces for the N10-D environment (stubs).

L0 DRAFT, not frozen (revised 2026-11-06 after the blind contradictor pass).
A backend (JaxMARL or EGG) implements ``RefGame``; the harness only sees
these Protocols, so the choice does not change the analysis.
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
        """Return (next_obs, team_reward, done). Message length L: TBD-11."""


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
        msg_shuffle: ``messages[perm[i]]`` for episode i, with
        ``perm = episode_derangement(len(messages), random.Random(batch_seed))``,
        at training and at evaluation. A fixed permutation of target
        identities is NOT allowed: the message would stay a function of the
        target and a listener could learn it.
        """


def episode_derangement(n: int, rng: random.Random) -> list[int]:
    """Uniform fixed-point-free permutation of the n episodes of a batch.

    Each episode receives the message emitted for another episode of the same
    batch; the marginal of messages and their length are kept. Identical
    target labels across episodes are not excluded (a 1/K coincidence is not
    a leak). Rejection sampling from uniform permutations (about e tries).
    """
    if n < 2:
        raise ValueError("a derangement needs at least 2 episodes in the batch")
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(i != p for i, p in enumerate(perm)):
            return list(perm)


class Trainer(Protocol):
    def train(self, game: RefGame, channel: Channel, seed: int, steps: int) -> Any:
        """Train and return the frozen policy pair.

        The training curve (mean of the last 5 logs, gate G2 definition) is
        logged as a secondary summary only; it is not a binomial of n_eval.
        """


class Evaluator(Protocol):
    def evaluate(
        self, game: RefGame, channel: Channel, policy: Any, seed: int, n_eval: int
    ) -> tuple[int, int]:
        """Primary metric: the frozen policy on n_eval independent episodes
        (fresh targets and candidate orders, no learning). Return
        (successes, n_eval). n_eval: TBD-11."""
