"""Task and arm specification for N10-D (L0 skeleton, stdlib only).

Every numeric default marked TBD is a placeholder pending a PI decision
listed at the top of docs/prereg/n10d/protocol.md. Nothing here is frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Condition = Literal["asym", "sym"]
Arm = Literal["eng", "nocomm", "learned", "msg_shuffle"]


@dataclass(frozen=True)
class TaskSpec:
    """Referential game: a speaker and a listener, one shared team reward.

    asym: only the speaker sees the target; the listener sees C candidates.
    sym: both agents see the target (message is redundant by construction).
    """

    condition: Condition
    n_targets: int = 16  # TBD by PI (K)
    n_candidates: int = 4  # TBD by PI (C)
    alphabet: int = 16  # TBD by PI (A)

    def __post_init__(self) -> None:
        if not 2 <= self.n_candidates <= self.n_targets:
            raise ValueError("need 2 <= C <= K")


def blind_ceiling(spec: TaskSpec) -> float:
    """Exact best team success of any policy pair that exchanges nothing.

    asym: the target is uniform among the C candidates and independent of
    everything the listener sees, so no listener policy beats 1/C.
    sym: the listener sees the target, so the ceiling is 1.0.
    """
    return 1.0 / spec.n_candidates if spec.condition == "asym" else 1.0
