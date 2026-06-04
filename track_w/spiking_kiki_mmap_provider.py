"""Mmap-backed ``SpikingKikiWeightsProvider`` over the on-disk .npz bundle.

The clemsail/spikingkiki-35b-a3b-v4 HF mirror stores ~775 module files
per transformer block, with names like
``model_layers_<block_idx>_<module>.npz``. This provider scans a local
directory matching that layout and lazily ``np.load(..., mmap_mode='r')``
each block's attention QKV projection on demand, so RAM pressure stays
flat regardless of the bundle's 70 GB on-disk footprint.

The WML-protocol cluster needs one ``(hidden_dim, hidden_dim)``
projection per block. We use the **Q projection** — the first
``hidden_dim`` rows of the stacked QKV weight, matching the standard
attention layout ``W_qkv ∈ R^{3·h × h}``. Future blocks can refine this
to a learned synthesis once the SP3 corpus is wired to learn from
spikingkiki activations directly.

Test fixtures save synthetic numpy arrays with ``np.savez`` so the
provider can be exercised without downloading the 70 GB bundle.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

_BLOCK_QKV_RE = re.compile(
    r"^model_layers_(\d+)_linear_attn_in_proj_qkv\.npz$",
)


class MmapNpzProvider:
    """Lazy mmap-backed provider over a spikingkiki ``.npz`` directory.

    On construction, scans ``weights_dir`` for files matching the
    spikingkiki QKV naming pattern, enumerates the available block
    indices, and infers ``hidden_dim`` from the first file. Per-block
    weight tensors are mmapped (not loaded into RAM) on each
    ``get_block_projection`` call.

    Sparse bundles are supported: if only a subset of the 40 transformer
    blocks is present (e.g. an experimental download of blocks 0, 10,
    12), the provider exposes exactly those indices via
    :attr:`block_indices`. ``n_blocks`` reflects the count of blocks
    actually present, not the upstream model's total.
    """

    def __init__(self, weights_dir: Path) -> None:
        self._weights_dir = Path(weights_dir)
        if not self._weights_dir.is_dir():
            raise ValueError(
                f"weights_dir is not a directory: {self._weights_dir}",
            )

        index: dict[int, Path] = {}
        for path in self._weights_dir.iterdir():
            m = _BLOCK_QKV_RE.match(path.name)
            if m is None:
                continue
            block_idx = int(m.group(1))
            index[block_idx] = path

        if not index:
            raise ValueError(
                f"no spikingkiki QKV files under {self._weights_dir}; "
                "expected model_layers_NN_linear_attn_in_proj_qkv.npz",
            )

        first_path = index[min(index)]
        hidden = self._infer_hidden_dim(first_path)

        self._index = index
        self._sorted_blocks = tuple(sorted(index))
        self.hidden_dim = hidden
        self.n_blocks = len(index)

    @staticmethod
    def _infer_hidden_dim(path: Path) -> int:
        """Read just enough of the .npz header to learn the QKV's shape."""
        with np.load(path, mmap_mode="r") as data:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"empty .npz: {path}")
            arr = data[keys[0]]
            if arr.ndim != 2:
                raise ValueError(
                    f"expected 2D QKV weight in {path}, got shape {arr.shape}",
                )
            n_rows, n_cols = arr.shape
        if n_rows == 3 * n_cols:
            return int(n_cols)
        if n_rows == n_cols:
            return int(n_cols)
        # Conservative fallback: take the smaller dim as hidden.
        return int(min(n_rows, n_cols))

    @property
    def block_indices(self) -> tuple[int, ...]:
        return self._sorted_blocks

    def get_block_projection(self, block_idx: int) -> Tensor:
        if block_idx not in self._index:
            available = self._sorted_blocks
            head = available[:5]
            tail = "..." if len(available) > 5 else ""
            raise IndexError(
                f"block_idx {block_idx} not present in bundle; "
                f"available: {list(head)}{tail}",
            )
        path = self._index[block_idx]
        with np.load(path, mmap_mode="r") as data:
            keys = list(data.keys())
            arr = np.asarray(data[keys[0]], dtype=np.float32)
        h = self.hidden_dim
        q = arr[:h, :h]
        return torch.from_numpy(q.copy())
