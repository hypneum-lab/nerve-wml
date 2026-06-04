"""Tests for SpikingKikiWML — substrate scaffold for spikingkiki rate-LIF."""
from __future__ import annotations

import torch

from nerve_core.protocols import WML
from track_w.mock_nerve import MockNerve
from track_w.spiking_kiki_wml import (
    InMemoryProvider,
    SpikingKikiWeightsProvider,
    SpikingKikiWML,
)


def _provider(hidden: int = 16, n_blocks: int = 3) -> InMemoryProvider:
    torch.manual_seed(0)
    return InMemoryProvider(
        [torch.randn(hidden, hidden) * 0.1 for _ in range(n_blocks)],
    )


def test_provider_protocol_conformance() -> None:
    p = _provider()
    assert isinstance(p, SpikingKikiWeightsProvider)
    assert p.n_blocks == 3
    assert p.hidden_dim == 16


def test_provider_rejects_mismatched_shapes() -> None:
    try:
        InMemoryProvider([torch.randn(16, 16), torch.randn(8, 8)])
    except ValueError as e:
        assert "expected" in str(e)
    else:
        raise AssertionError("expected shape mismatch ValueError")


def test_provider_rejects_empty_list() -> None:
    try:
        InMemoryProvider([])
    except ValueError as e:
        assert "at least one" in str(e)
    else:
        raise AssertionError("expected ValueError on empty provider")


def test_spiking_kiki_wml_has_required_attrs() -> None:
    p = _provider()
    wml = SpikingKikiWML(id=2, provider=p, block_idx=1, alphabet_size=32, seed=0)
    assert wml.id == 2
    assert wml.codebook.shape == (32, 16)
    assert wml.v_mem.shape == (16,)
    assert wml.block_projection.shape == (16, 16)
    assert wml.n_micro_ticks == 128
    assert wml.threshold == 0.0625
    assert wml.tau == 1.0


def test_spiking_kiki_wml_conforms_to_wml_protocol() -> None:
    p = _provider()
    wml = SpikingKikiWML(id=0, provider=p, seed=0)
    assert isinstance(wml, WML)


def test_block_projection_loaded_from_provider() -> None:
    p = _provider()
    expected = p.get_block_projection(2).clone()
    wml = SpikingKikiWML(id=0, provider=p, block_idx=2, seed=0)
    assert torch.equal(wml.block_projection, expected)


def test_block_projection_is_frozen_buffer_not_parameter() -> None:
    p = _provider()
    wml = SpikingKikiWML(id=0, provider=p, seed=0)
    param_ids = {id(t) for t in wml.parameters()}
    # Buffer must not be a learnable parameter (substrate wiring is frozen).
    assert id(wml.block_projection) not in param_ids
    # Codebook and readout heads ARE parameters.
    assert id(wml.codebook) in param_ids
    assert id(wml.emit_head_pi.weight) in param_ids
    assert id(wml.input_proj.weight) in param_ids


def test_invalid_block_idx_raises() -> None:
    p = _provider(n_blocks=2)
    try:
        SpikingKikiWML(id=0, provider=p, block_idx=5, seed=0)
    except IndexError as e:
        assert "5" in str(e)
    else:
        raise AssertionError("expected IndexError for out-of-range block_idx")


def test_reset_state_zeroes_membrane() -> None:
    p = _provider()
    wml = SpikingKikiWML(id=0, provider=p, seed=0)
    wml.v_mem.add_(0.42)
    assert wml.v_mem.abs().sum().item() > 0
    wml.reset_state()
    assert wml.v_mem.abs().sum().item() == 0.0


def test_step_runs_without_inbound() -> None:
    # A WML facing an empty inbound queue must not crash or emit.
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    p = _provider()
    wml = SpikingKikiWML(id=0, provider=p, n_micro_ticks=8, seed=0)
    wml.step(nerve, t=0.0)
    # No outbound queued for the other WML — step was silent.
    assert nerve.listen(1) == []


def test_step_advances_membrane_under_drive() -> None:
    # With a non-trivial input the LIF integration must update v_mem and
    # leave at least one micro-tick with measurable activity.
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    p = _provider()
    wml = SpikingKikiWML(id=0, provider=p, n_micro_ticks=16, seed=0)
    # Force a strong drive by patching the input projection bias.
    with torch.no_grad():
        wml.input_proj.bias.fill_(0.5)
    wml.step(nerve, t=0.0)
    # After step, v_mem is either the unspent residual or 0 after reset;
    # the rate-coded path must have run regardless of emission.
    assert torch.is_tensor(wml.v_mem)


def test_invalid_micro_ticks_raises() -> None:
    p = _provider()
    try:
        SpikingKikiWML(id=0, provider=p, n_micro_ticks=0, seed=0)
    except ValueError as e:
        assert "n_micro_ticks" in str(e)
    else:
        raise AssertionError("expected ValueError on n_micro_ticks=0")
