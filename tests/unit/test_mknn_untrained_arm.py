"""Untrained-network arm of Test (6) mutual-kNN (issue #110).

Shape and plumbing only: no verdict, no comparison to the trained arm.
"""
from __future__ import annotations

import torch

from scripts.platonic_rh_alignment import (
    _train_lif,
    _train_mlp,
    untrained_arm,
)
from track_w.mlp_wml import MlpWML


def test_untrained_arm_runs_and_has_expected_shape() -> None:
    out = untrained_arm(seed=0, n_eval=64, ks=[5, 10])
    assert set(out) == {"5", "10"}
    for row in out.values():
        assert set(row) == {"mlp_lif_untrained", "floor"}
        assert isinstance(row["mlp_lif_untrained"], float)
        assert 0.0 <= row["mlp_lif_untrained"] <= 1.0


def test_untrained_mlp_equals_fresh_init() -> None:
    """Zero training steps leaves the seeded initial weights."""
    mlp0 = _train_mlp(0, seed=3)
    ref = MlpWML(id=0, d_hidden=16, seed=3)
    for a, b in zip(mlp0.core.parameters(), ref.core.parameters(), strict=True):
        assert torch.equal(a, b)


def test_untrained_lif_is_deterministic() -> None:
    lif_a, enc_a = _train_lif(0, seed=2)
    lif_b, enc_b = _train_lif(0, seed=2)
    assert torch.equal(enc_a.weight, enc_b.weight)
    assert torch.equal(lif_a.input_proj.weight, lif_b.input_proj.weight)
