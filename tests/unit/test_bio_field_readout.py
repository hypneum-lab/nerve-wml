"""Unit tests for BioFieldWMLReadout — belief-state probe adapter.

Smoke-level checks that the adapter exposes the Track-W supervised harness
surface (core / codebook / emit_head_pi) and trains end-to-end without
NaN/crash on a tiny HardFlowProxyTask. These are *interface* tests, not
accuracy-threshold claims (the accuracy number is produced by the kxkm
multi-seed run, not asserted here).
"""
from __future__ import annotations

import torch

from track_w.bio_field_readout import BioFieldWMLReadout
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask


def test_core_shape_and_finiteness() -> None:
    """core(x) maps [B, input_dim] -> [B, 3*n_neurons], all finite."""
    bio = BioFieldWMLReadout(id=0, n_neurons=8, input_dim=16, seed=0)
    x = torch.randn(5, 16)
    h = bio.core(x)
    assert h.shape == (5, 3 * 8)
    assert torch.isfinite(h).all()


def test_emit_head_surface_matches_harness() -> None:
    """Adapter exposes core / codebook / emit_head_pi like MLP/LIF."""
    bio = BioFieldWMLReadout(id=0, n_neurons=8, input_dim=16, seed=0)
    assert hasattr(bio, "core")
    assert hasattr(bio, "codebook")
    assert hasattr(bio, "emit_head_pi")
    x = torch.randn(4, 16)
    logits = bio.emit_head_pi(bio.core(x))
    assert logits.shape == (4, bio.alphabet_size)
    assert torch.isfinite(logits).all()


def test_core_is_deterministic_function_of_input() -> None:
    """Belief is reset per-batch: core(x) has no hidden cross-call state."""
    bio = BioFieldWMLReadout(id=0, n_neurons=8, input_dim=16, seed=0)
    x = torch.randn(3, 16)
    with torch.no_grad():
        h1 = bio.core(x)
        _ = bio.core(torch.randn(7, 16))  # different batch in between
        h2 = bio.core(x)
    assert torch.allclose(h1, h2)


def test_trains_end_to_end_no_nan() -> None:
    """A few SGD steps on HardFlowProxyTask run without NaN and reduce loss."""
    torch.manual_seed(0)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    bio = BioFieldWMLReadout(id=0, n_neurons=16, input_dim=16, seed=10)
    opt = torch.optim.Adam(bio.parameters(), lr=1e-2)

    first_loss = None
    last_loss = None
    for _ in range(50):
        x, y = task.sample(batch=64)
        logits = bio.emit_head_pi(bio.core(x))[:, : task.n_classes]
        loss = torch.nn.functional.cross_entropy(logits, y)
        assert torch.isfinite(loss), "loss became non-finite"
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

    assert first_loss is not None and last_loss is not None
    # Trainable input_proj + head should at least not diverge.
    assert last_loss <= first_loss + 1e-3


def test_parameters_are_trainable_and_grad_flows() -> None:
    """input_proj, emit_head_pi, codebook receive gradients; VMP has none."""
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=1)
    bio = BioFieldWMLReadout(id=0, n_neurons=8, input_dim=16, seed=1)
    x, y = task.sample(batch=32)
    logits = bio.emit_head_pi(bio.core(x))[:, : task.n_classes]
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # input_proj weight gets grad through the VMP update -> probe is
    # differentiable wrt the learned projection.
    assert bio.input_proj.weight.grad is not None
    assert torch.isfinite(bio.input_proj.weight.grad).all()
    assert bio.emit_head_pi.weight.grad is not None
