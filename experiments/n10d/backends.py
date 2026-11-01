"""Backend stubs. Imports are lazy and the stubs raise: no dependency is installed."""
from __future__ import annotations

from .interfaces import RefGame
from .spec import TaskSpec


def make_jaxmarl_game(spec: TaskSpec) -> RefGame:
    """TBD by PI: JaxMARL (MPE simple_reference-like or custom Lewis env)."""
    raise NotImplementedError("JaxMARL backend not implemented (L0 skeleton)")


def make_egg_game(spec: TaskSpec) -> RefGame:
    """TBD by PI: EGG (PyTorch) Lewis signalling game."""
    raise NotImplementedError("EGG backend not implemented (L0 skeleton)")


BACKENDS = {"jaxmarl": make_jaxmarl_game, "egg": make_egg_game}
