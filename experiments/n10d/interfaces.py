"""Backend-neutral interfaces for the N10-D environment (stubs).

A backend (JaxMARL or EGG, TBD by PI) implements ``RefGame``; the harness
only sees these Protocols, so the choice does not change the analysis.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from .spec import Arm, TaskSpec


class RefGame(Protocol):
    spec: TaskSpec

    def reset(self, seed: int) -> Any:
        """Return the initial observation pair (speaker_obs, listener_obs)."""

    def step(self, message: Sequence[int], action: int) -> tuple[Any, float, bool]:
        """Return (next_obs, team_reward, done)."""


class Sender(Protocol):
    def emit(self, speaker_obs: Any) -> Sequence[int]: ...


class Channel(Protocol):
    """Arm-specific transform between emit and listener input."""

    arm: Arm

    def transmit(self, message: Sequence[int], *, episode: int) -> Sequence[int]:
        """eng: identity of the engineered code; nocomm: zeros;
        msg_shuffle: message of another episode (permutation, fixed seed)."""


class Trainer(Protocol):
    def train(self, game: RefGame, channel: Channel, seed: int, steps: int) -> dict[str, float]:
        """Return the training success curve summary (last-5-logs mean)."""
