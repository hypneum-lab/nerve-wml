"""Backend stubs. Imports are lazy and the stubs raise: no dependency is installed."""
from __future__ import annotations

from .interfaces import RefGame
from .spec import TaskSpec


def make_jaxmarl_game(spec: TaskSpec) -> RefGame:
    """JaxMARL: custom JAX Lewis game with exactly ``spec`` (protocol section 7).

    MPE simple_reference does not fit (3 landmarks asserted, dim_c = 10,
    reward mixed by local_ratio = 0.5) and is not used. Details: TBD-11.
    """
    raise NotImplementedError("JaxMARL backend not implemented (L0 skeleton)")


def make_egg_game(spec: TaskSpec) -> RefGame:
    """TBD by PI: EGG (PyTorch) Lewis signalling game."""
    raise NotImplementedError("EGG backend not implemented (L0 skeleton)")


BACKENDS = {"jaxmarl": make_jaxmarl_game, "egg": make_egg_game}
