"""Backend stubs. Imports are lazy and the stubs raise: no dependency is installed."""
from __future__ import annotations

from .interfaces import RefGame
from .spec import TaskSpec


def make_jaxmarl_game(spec: TaskSpec) -> RefGame:
    """JaxMARL custom one-step referential game, exactly ``spec`` (section 7, A11).

    Not MPE simple_reference: it asserts 3 landmarks, uses dim_c = 10 and
    mixes the reward by local_ratio = 0.5 (JaxMARL @31b0a23,
    ``simple_reference.py:25,26,31,37,161``), so it is not used.
    Target uniform over K; C candidates containing it, shuffled per episode;
    speaker sees the target, listener sees the candidates (plus the target
    on sym); speaker emits one symbol of A, the arm's channel transforms it,
    listener picks a candidate index in [0, C); common reward 1 if right,
    else 0. Encoding, learned-arm message length and evaluation action rule
    stay TBD-11.
    """
    raise NotImplementedError("JaxMARL backend not implemented (L0 skeleton)")


def make_egg_game(spec: TaskSpec) -> RefGame:
    """EGG (PyTorch) Lewis signalling game, same environment (protocol section 7)."""
    raise NotImplementedError("EGG backend not implemented (L0 skeleton)")


BACKENDS = {"jaxmarl": make_jaxmarl_game, "egg": make_egg_game}
