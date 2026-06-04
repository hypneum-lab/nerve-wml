"""Tests for SpikingKikiBlockProvider Protocol and InMemoryBlockProvider.

Design ref:
  docs/superpowers/specs/2026-05-21-spike-attention-moe-design.md

Tests correspond to spec test-plan item 1 (Protocol conformance) and
item 2 (unit tests with small synthetic bundle). Items 3-4 (numerical
parity vs ANN reference and bench) are deferred pending audit of
convert_spikingkiki_35b.py from the upstream repo — see module docstring
in spiking_kiki_wml.py for details.

The v1 SpikingKikiWML.step() codepath (collapsed projection) must remain
unaffected by this Protocol extension — existing tests verify that.
"""
from __future__ import annotations

import pytest
import torch

from track_w.mock_nerve import MockNerve
from track_w.spiking_kiki_wml import (
    InMemoryBlockProvider,
    InMemoryProvider,
    SpikingKikiBlockProvider,
    SpikingKikiWeightsProvider,
    SpikingKikiWML,
)

# ---------------------------------------------------------------------------
# Test 1 — Protocol conformance
# ---------------------------------------------------------------------------

def test_block_provider_protocol_is_runtime_checkable() -> None:
    """SpikingKikiBlockProvider must be a runtime-checkable Protocol."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    assert isinstance(p, SpikingKikiBlockProvider)


def test_block_provider_is_also_weights_provider() -> None:
    """SpikingKikiBlockProvider must extend SpikingKikiWeightsProvider."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    assert isinstance(p, SpikingKikiWeightsProvider)


def test_in_memory_provider_does_not_satisfy_block_provider() -> None:
    """The v1 InMemoryProvider must NOT satisfy SpikingKikiBlockProvider
    (it does not implement the extended methods)."""
    p = InMemoryProvider([torch.randn(16, 16)])
    assert not isinstance(p, SpikingKikiBlockProvider)


def test_block_provider_has_required_attributes() -> None:
    """n_experts and n_heads must be accessible as plain attributes."""
    p = InMemoryBlockProvider(
        hidden_dim=16, n_blocks=2, n_experts=4, n_heads=2, seed=0
    )
    assert p.n_experts == 4
    assert p.n_heads == 2
    assert p.n_blocks == 2
    assert p.hidden_dim == 16


# ---------------------------------------------------------------------------
# Test 2 — Unit tests with small synthetic bundle
# ---------------------------------------------------------------------------

def test_get_qkv_returns_three_tensors() -> None:
    """get_qkv must return (W_Q, W_K, W_V) as a 3-tuple of tensors."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    q, k, v = p.get_qkv(0)
    assert q.shape == k.shape == v.shape
    assert q.ndim == 2
    assert q.shape[-1] == 16  # hidden_dim


def test_get_qkv_shape_matches_heads() -> None:
    """W_Q/W_K/W_V rows should be n_heads * head_dim = hidden_dim."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=4)
    q, k, v = p.get_qkv(0)
    head_dim = 16 // 4
    assert q.shape == (4 * head_dim, 16)  # (n_heads*head_dim, hidden_dim)


def test_get_attn_out_shape() -> None:
    """get_attn_out must return shape (hidden_dim, n_heads * head_dim)."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    w_out = p.get_attn_out(0)
    assert w_out.shape == (16, 16)


def test_get_router_shape() -> None:
    """get_router must return shape (n_experts, hidden_dim)."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=4, n_heads=2)
    router = p.get_router(0)
    assert router.shape == (4, 16)


def test_get_expert_returns_three_tensors() -> None:
    """get_expert must return (W_gate, W_up, W_down)."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    gate, up, down = p.get_expert(0, expert_idx=1)
    intermediate = 16 * 2  # 2x ratio
    assert gate.shape == (intermediate, 16)
    assert up.shape == (intermediate, 16)
    assert down.shape == (16, intermediate)


def test_get_norms_returns_two_vectors() -> None:
    """get_norms must return (gamma_in, gamma_post) of shape (hidden_dim,)."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    gamma_in, gamma_post = p.get_norms(0)
    assert gamma_in.shape == (16,)
    assert gamma_post.shape == (16,)


def test_get_block_projection_v1_fallback() -> None:
    """get_block_projection must still work on InMemoryBlockProvider."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    proj = p.get_block_projection(0)
    assert proj.shape == (16, 16)


def test_block_provider_deterministic_with_seed() -> None:
    """Same seed -> identical weights across two InMemoryBlockProvider instances."""
    p1 = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2, seed=42)
    p2 = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2, seed=42)
    q1, k1, v1 = p1.get_qkv(0)
    q2, k2, v2 = p2.get_qkv(0)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


def test_block_provider_multi_block() -> None:
    """Multi-block providers must return different weights per block."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=3, n_experts=2, n_heads=2, seed=0)
    q0, _, _ = p.get_qkv(0)
    q1, _, _ = p.get_qkv(1)
    q2, _, _ = p.get_qkv(2)
    # With distinct random draws, all three Q matrices should differ.
    assert not torch.equal(q0, q1)
    assert not torch.equal(q1, q2)


def test_out_of_range_block_idx_raises() -> None:
    """get_qkv with out-of-range block_idx must raise IndexError."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=2, n_experts=2, n_heads=2)
    with pytest.raises(IndexError, match="block_idx"):
        p.get_qkv(5)


def test_out_of_range_expert_idx_raises() -> None:
    """get_expert with out-of-range expert_idx must raise IndexError."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2)
    with pytest.raises(IndexError, match="expert_idx"):
        p.get_expert(0, expert_idx=99)


def test_hidden_dim_not_divisible_by_heads_raises() -> None:
    """hidden_dim not divisible by n_heads must raise ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        InMemoryBlockProvider(hidden_dim=15, n_blocks=1, n_experts=2, n_heads=4)


# ---------------------------------------------------------------------------
# Backward-compatibility: v1 SpikingKikiWML still works with block provider
# ---------------------------------------------------------------------------

def test_spiking_kiki_wml_accepts_block_provider() -> None:
    """SpikingKikiWML must accept InMemoryBlockProvider (satisfies v1 Protocol)."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2, seed=0)
    # Construction must not raise (block provider is a SpikingKikiWeightsProvider).
    wml = SpikingKikiWML(id=0, provider=p, block_idx=0, seed=0)
    assert wml.block_projection.shape == (16, 16)


def test_spiking_kiki_wml_step_unaffected_by_block_provider() -> None:
    """The v1 SpikingKikiWML.step() codepath must be unaffected by the
    extended Protocol — existing collapsed-projection logic runs normally."""
    p = InMemoryBlockProvider(hidden_dim=16, n_blocks=1, n_experts=2, n_heads=2, seed=0)
    wml = SpikingKikiWML(id=0, provider=p, n_micro_ticks=8, seed=0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    # Must not raise; the step uses the v1 collapsed projection.
    wml.step(nerve, t=0.0)
    assert torch.is_tensor(wml.v_mem)
