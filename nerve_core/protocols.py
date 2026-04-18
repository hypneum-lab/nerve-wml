"""Nerve and WML protocols — the contract that both Track-P and Track-W obey.

See spec §4.2, §4.4.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from torch import Tensor

from .neuroletter import Neuroletter, Phase, Role


@runtime_checkable
class Nerve(Protocol):
    """Shared nerve contract.

    Track-P provides SimNerve (real γ/θ oscillators). Track-W starts against
    MockNerve (in-memory queue, no rhythms). Both must satisfy N-1..N-5 from
    the spec.
    """

    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def send(self, letter: Neuroletter) -> None: ...

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]: ...

    def time(self) -> float: ...

    def tick(self, dt: float) -> None: ...

    def routing_weight(self, src: int, dst: int) -> float: ...


@runtime_checkable
class WML(Protocol):
    """A WML (World Model Language) = a neuron cluster with a local codebook
    and a step() that listens, computes internally, and emits.
    """

    id:       int
    codebook: Tensor

    def step(self, nerve: Nerve, t: float) -> None: ...

    def parameters(self) -> Iterable[Tensor]: ...
