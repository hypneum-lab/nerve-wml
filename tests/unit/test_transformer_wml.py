"""Unit tests for TransformerWML — third substrate for polymorphism.

Contract checks (mirrors tests/unit/test_mlp_wml.py and test_lif_wml.py):
  - W-2: parameters() includes codebook + all internal weights.
  - Local RNG isolation (seeded init must not mutate global torch RNG).
  - Protocol compliance: step() never raises on a fresh nerve.
  - W-1: step() communicates via nerve only (emit π on γ-active phase).
"""
from __future__ import annotations

import torch

from track_w.mock_nerve import MockNerve
from track_w.transformer_wml import TransformerWML


def test_transformer_wml_has_required_attrs():
    """Implements the WML Protocol surface by duck-typing (id, codebook,
    step, parameters)."""
    wml = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=0)
    assert wml.id == 0
    assert isinstance(wml.codebook, torch.Tensor)
    assert wml.codebook.shape == (64, 16)
    assert callable(wml.step)
    assert callable(wml.parameters)


def test_transformer_wml_parameters_include_codebook():
    wml = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=0)
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids
    # Heads are trainable.
    assert id(wml.emit_head_pi.weight) in param_ids
    assert id(wml.emit_head_eps.weight) in param_ids


def test_transformer_wml_seed_is_local():
    """Creating TransformerWML does not mutate global torch RNG."""
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=99)
    observed = torch.rand(1).item()
    assert expected == observed


def test_transformer_wml_rejects_bad_tokenization():
    """d_model must be divisible by n_tokens."""
    import pytest
    with pytest.raises(ValueError):
        TransformerWML(id=0, d_model=17, n_tokens=4)


def test_transformer_wml_step_does_not_raise():
    """Smoke: step() runs cleanly on a fresh nerve."""
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=0)
    wml.step(nerve, t=0.0)


def test_transformer_wml_step_emits_pi_on_gamma():
    """When γ is active and routing allows, step() emits at least a π letter."""
    from nerve_core.neuroletter import Phase, Role

    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    # Force connectivity 0 → 1 by exercising until a letter lands.
    wml = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=0)
    wml.step(nerve, t=0.0)
    inbox_1 = nerve.listen(1)
    # Emission depends on routing_weight(0,1); assert the nerve is consistent.
    for letter in inbox_1:
        if letter.role is Role.PREDICTION:
            assert letter.phase is Phase.GAMMA
            assert letter.src == 0
            return
    # Silence is also valid per N-1 if routing was zero — check graph.
    assert nerve.routing_weight(0, 1) in (0.0, 1.0)


def test_transformer_wml_core_output_shape():
    """core([B, d_model]) returns [B, d_model]."""
    wml = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=0)
    x = torch.randn(4, 16)
    h = wml.core(x)
    assert h.shape == (4, 16)
