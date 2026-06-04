"""Tests for the mmap-backed spikingkiki provider."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from track_w.spiking_kiki_mmap_provider import MmapNpzProvider
from track_w.spiking_kiki_wml import SpikingKikiWeightsProvider, SpikingKikiWML


def _write_block(out_dir: Path, block_idx: int, hidden: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed + block_idx)
    qkv = rng.standard_normal((3 * hidden, hidden), dtype=np.float32) * 0.1
    path = out_dir / f"model_layers_{block_idx}_linear_attn_in_proj_qkv.npz"
    np.savez(path, weight=qkv)
    return path


def _write_bundle(out_dir: Path, blocks: list[int], hidden: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        _write_block(out_dir, b, hidden)
    # An unrelated file the provider should ignore.
    (out_dir / "lif_metadata.json").write_text("{}", encoding="utf-8")
    np.savez(
        out_dir / "model_layers_0_mlp_experts_0_gate_proj.npz",
        weight=np.zeros((hidden, hidden), dtype=np.float32),
    )
    return out_dir


def test_mmap_provider_conforms_to_protocol(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, blocks=[0, 1, 2], hidden=8)
    p = MmapNpzProvider(bundle)
    assert isinstance(p, SpikingKikiWeightsProvider)


def test_mmap_provider_discovers_blocks_and_infers_hidden(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, blocks=[0, 1, 2, 3], hidden=16)
    p = MmapNpzProvider(bundle)
    assert p.n_blocks == 4
    assert p.hidden_dim == 16
    assert p.block_indices == (0, 1, 2, 3)


def test_mmap_provider_ignores_non_qkv_files(tmp_path: Path) -> None:
    # The bundle helper adds an MLP-expert file + a JSON; only QKV files count.
    bundle = _write_bundle(tmp_path, blocks=[0], hidden=8)
    p = MmapNpzProvider(bundle)
    assert p.n_blocks == 1


def test_mmap_provider_supports_sparse_blocks(tmp_path: Path) -> None:
    # Only blocks 0, 10, 12 — n_blocks is the count present, not the max.
    bundle = _write_bundle(tmp_path, blocks=[0, 10, 12], hidden=8)
    p = MmapNpzProvider(bundle)
    assert p.n_blocks == 3
    assert p.block_indices == (0, 10, 12)


def test_get_block_projection_returns_q_slice(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, blocks=[0], hidden=8)
    p = MmapNpzProvider(bundle)
    proj = p.get_block_projection(0)
    assert isinstance(proj, torch.Tensor)
    assert proj.shape == (8, 8)
    assert proj.dtype == torch.float32

    # Confirm the slice matches the first hidden rows of the on-disk QKV.
    raw = np.load(
        bundle / "model_layers_0_linear_attn_in_proj_qkv.npz",
    )["weight"][:8, :8]
    assert torch.allclose(proj, torch.from_numpy(raw.astype(np.float32)))


def test_get_block_projection_raises_for_missing_block(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, blocks=[0, 10], hidden=4)
    p = MmapNpzProvider(bundle)
    with pytest.raises(IndexError, match="not present"):
        p.get_block_projection(5)


def test_mmap_provider_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no spikingkiki QKV"):
        MmapNpzProvider(tmp_path / "empty")


def test_mmap_provider_non_dir_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_dir.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        MmapNpzProvider(bad)


def test_mmap_provider_square_qkv_fallback(tmp_path: Path) -> None:
    # Some experimental bundles ship a square (hidden, hidden) QKV instead
    # of the canonical (3*hidden, hidden) stack — the provider should still
    # work, falling back to the square layout.
    (tmp_path / "sq").mkdir()
    qkv_square = np.eye(6, dtype=np.float32)
    np.savez(
        tmp_path / "sq" / "model_layers_0_linear_attn_in_proj_qkv.npz",
        weight=qkv_square,
    )
    p = MmapNpzProvider(tmp_path / "sq")
    assert p.hidden_dim == 6
    proj = p.get_block_projection(0)
    assert torch.equal(proj, torch.from_numpy(qkv_square))


def test_mmap_provider_drives_spiking_kiki_wml(tmp_path: Path) -> None:
    # End-to-end: mmap provider plugged into SpikingKikiWML — exercises the
    # full provider Protocol path the substrate relies on.
    bundle = _write_bundle(tmp_path, blocks=[0, 1], hidden=8)
    p = MmapNpzProvider(bundle)
    wml = SpikingKikiWML(
        id=0, provider=p, block_idx=1, alphabet_size=16,
        n_micro_ticks=4, seed=0,
    )
    assert wml.block_projection.shape == (8, 8)
    # Block projection matches the on-disk Q slice for block 1.
    raw = np.load(
        bundle / "model_layers_1_linear_attn_in_proj_qkv.npz",
    )["weight"][:8, :8].astype(np.float32)
    assert torch.allclose(wml.block_projection, torch.from_numpy(raw))
