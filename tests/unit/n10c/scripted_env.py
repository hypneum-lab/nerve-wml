"""Picklable, seed-driven scripted envs for worker-process tests."""
from __future__ import annotations

import numpy as np

_CALLS: dict[int, int] = {}


class RandScripted:
    """Behaviour is a pure function of (rand_seed, agent index, step).

    The agent index is the creation order for a given level seed (the
    coop wrapper builds agent 0 then agent 1 with the same seed).
    """

    def __init__(self, rand_seed: int, fail_at: int | None = None) -> None:
        n = _CALLS.get(rand_seed, 0)
        _CALLS[rand_seed] = n + 1
        self.agent = n % 2
        self.rng = np.random.default_rng([rand_seed, self.agent, n // 2])
        self.coin_at = int(self.rng.integers(0, 6))
        self.die_at = int(self.rng.integers(3, 9))
        self.fail_at = fail_at
        self.rand_seed = rand_seed
        self.t = -1

    def act(self, actions: np.ndarray) -> None:
        self.t += 1
        if self.fail_at is not None and self.t == self.fail_at:
            raise RuntimeError("scripted env failure")

    def observe(self):
        coin = self.t == self.coin_at
        dead = self.t == self.die_at
        img = np.random.default_rng([self.rand_seed, self.agent, self.t + 1])
        img = img.integers(0, 256, (1, 64, 64, 3), dtype=np.uint8)
        return (np.array([10.0 if coin else 0.0]), {"rgb": img},
                np.array([dead]))

    def close(self) -> None:
        pass


def rand_factory(rand_seed: int) -> RandScripted:
    return RandScripted(rand_seed)


def failing_factory(rand_seed: int) -> RandScripted:
    return RandScripted(rand_seed, fail_at=3)
