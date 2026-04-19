"""LifWML → INT8 artefact export.

quantize_lif_wml produces a serializable dict with INT8 codebook and
input_proj weights plus per-tensor quantization metadata (scale, zero
point). save_neuromorphic_artefact writes JSON + NPZ; load_neuromorphic_
artefact is the inverse. Round-trip is bit-stable.

Plan 6 Task 4.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def _quantize_symmetric(t: Tensor, bits: int = 8) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization. Returns (int array, scale)."""
    max_abs = float(t.abs().max().item()) or 1.0
    qmax = 2 ** (bits - 1) - 1
    scale = max_abs / qmax
    q = (t.detach().cpu().numpy() / scale).round().astype(np.int8)
    return q, scale


def quantize_lif_wml(wml, *, bits: int = 8) -> dict:
    """Convert a LifWML to a serializable int-quantized dict.

    Returns keys:
      - codebook_int8, codebook_scale
      - input_proj_int8, input_proj_scale, input_proj_bias
      - v_thr, tau_mem, n_neurons, alphabet_size
    """
    cb_q, cb_scale = _quantize_symmetric(wml.codebook, bits=bits)
    wproj_q, wproj_scale = _quantize_symmetric(
        wml.input_proj.weight.detach(), bits=bits,
    )
    bias = wml.input_proj.bias.detach().cpu().numpy().astype(np.float32)
    return {
        "codebook_int8":     cb_q,
        "codebook_scale":    cb_scale,
        "input_proj_int8":   wproj_q,
        "input_proj_scale":  wproj_scale,
        "input_proj_bias":   bias,
        "v_thr":             float(wml.v_thr),
        "tau_mem":           float(wml.tau_mem),
        "n_neurons":         int(wml.n_neurons),
        "alphabet_size":     int(wml.alphabet_size),
        "bits":              int(bits),
    }


def save_neuromorphic_artefact(export: dict, path: str | Path) -> None:
    """Write the export dict to disk: artefact.json (meta) + weights.npz (arrays)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    # Split into arrays (weights.npz) and scalars (artefact.json).
    arrays = {k: v for k, v in export.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in export.items() if not isinstance(v, np.ndarray)}

    np.savez(out / "weights.npz", **arrays)
    (out / "artefact.json").write_text(json.dumps(scalars, indent=2))


def load_neuromorphic_artefact(path: str | Path) -> dict:
    """Inverse of save_neuromorphic_artefact. Returns the same dict shape."""
    src = Path(path)
    scalars = json.loads((src / "artefact.json").read_text())
    arrays = dict(np.load(src / "weights.npz"))
    out = {**scalars, **arrays}
    # Restore dtypes that JSON turned into Python ints.
    for k in ("n_neurons", "alphabet_size", "bits"):
        if k in out:
            out[k] = int(out[k])
    return out
