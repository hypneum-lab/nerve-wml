"""Tests for the optional plasticity_schedule / constellation_lock kwargs.

These tests only exercise the plasticity controller; the canonical
multiplexer contract is still pinned by `test_multiplexer.py`. The
default behaviour of `GammaThetaMultiplexer()` must remain bit-identical
to v1.3.0, so existing consumers (bouba_sens v0.3 grids, in particular)
reproduce byte-for-byte.
"""

from __future__ import annotations

import torch

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer


def test_defaults_preserve_v130_behaviour() -> None:
    """No kwargs = constellation remains a free nn.Parameter."""
    mux = GammaThetaMultiplexer(seed=0)
    assert mux.constellation.requires_grad is True
    assert mux.plasticity_step == 0


def test_step_increments_plasticity_counter() -> None:
    mux = GammaThetaMultiplexer(seed=0)
    assert mux.plasticity_step == 0
    mux.step()
    assert mux.plasticity_step == 1
    mux.step()
    mux.step()
    assert mux.plasticity_step == 3


def test_constellation_lock_after_freezes_requires_grad() -> None:
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=2)
    assert mux.constellation.requires_grad is True
    mux.step()  # step 1, still plastic
    assert mux.constellation.requires_grad is True
    mux.step()  # step 2, crosses the threshold -> lock
    assert mux.constellation.requires_grad is False


def test_constellation_lock_is_permanent() -> None:
    """Once locked, the constellation must not unlock, even if step()
    keeps being called. A Phase-2 training loop must not accidentally
    re-enable plasticity after Phase 1 froze it."""
    mux = GammaThetaMultiplexer(seed=0, constellation_lock_after=1)
    mux.step()
    assert mux.constellation.requires_grad is False
    for _ in range(10):
        mux.step()
    assert mux.constellation.requires_grad is False
