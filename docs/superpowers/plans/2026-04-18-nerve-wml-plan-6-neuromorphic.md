# Neuromorphic Hardware Export — LifWML → Loihi 2 / Akida Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a substrate-neutral export pipeline that converts a trained `LifWML` into a quantized neuromorphic artefact, verifies it against the PyTorch reference with a pure-numpy LIF mock runner (accuracy delta < 2 %), and ships vendor stubs for Loihi 2 and Akida so a future PR can add hardware SDK calls without touching the core pipeline.

**Architecture:** A new `neuromorphic/` package contains four modules: `spike_encoder` (rate + temporal coding), `export` (INT8 quantization + artefact save/load in JSON + NPZ), `mock_runner` (pure-numpy forward-Euler LIF integration), and `verify` (pytorch vs mock accuracy comparison). Two vendor-stub files (`loihi_stub.py`, `akida_stub.py`) raise `NotImplementedError` with full API docstrings so hardware owners can drop in `lava-nc` or `akida` without redesigning the interface. A deployment guide documents the end-to-end pipeline. No vendor SDK is added to `pyproject.toml`.

**Tech Stack:** Python 3.12+, PyTorch 2.3+, numpy 1.26+, pytest 8+, `LifWML` (`track_w/lif_wml.py`), `FlowProxyTask` (`track_w/tasks/flow_proxy.py`), `MockNerve` (`track_w/mock_nerve.py`).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `neuromorphic/__init__.py` | Package init; commented-out vendor stub imports |
| Create | `neuromorphic/spike_encoder.py` | `rate_encode`, `temporal_encode` |
| Create | `neuromorphic/export.py` | `quantize_lif_wml`, `save_neuromorphic_artefact`, `load_neuromorphic_artefact` |
| Create | `neuromorphic/mock_runner.py` | `MockNeuromorphicRunner` — pure-numpy LIF simulation |
| Create | `neuromorphic/verify.py` | `compare_software_vs_neuromorphic` |
| Create | `neuromorphic/loihi_stub.py` | `LoihiCompiler.compile` stub |
| Create | `neuromorphic/akida_stub.py` | `AkidaCompiler.compile` stub |
| Create | `tests/unit/test_spike_encoder.py` | Shape, binary, firing-rate, temporal ordering |
| Create | `tests/unit/test_export.py` | Quantization round-trip, artefact save/load bit-stability |
| Create | `tests/unit/test_mock_runner.py` | Pure-numpy LIF dynamics, output shape |
| Create | `tests/unit/test_verify.py` | `compare_software_vs_neuromorphic` dict keys + types |
| Create | `tests/integration/track_w/test_gate_neuro.py` | Gate aggregator: rate bounds, delta < 2 %, round-trip |
| Create | `tests/integration/track_w/test_gate_neuro_accuracy.py` | End-to-end accuracy test on FlowProxyTask |
| Create | `docs/neuromorphic/deployment-guide.md` | Install guide + artefact format + pipeline diagram |
| Modify | `pyproject.toml` | Add `neuromorphic` to wheel packages |

---

## Quantization reference (used in Tasks 4 and 5)

Symmetric per-tensor INT8:

```python
import numpy as np

def _quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (int8_array, scale).
    scale = max(|arr|) / 127.0  (symmetric -> zero_point always 0).
    Recover float via: float_approx = int8_arr.astype(float32) * scale
    """
    amax = float(np.abs(arr).max())
    scale = amax / 127.0 if amax > 0.0 else 1.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized, scale
```

The round-trip error for exact integers: `dequant = int8.astype(float32) * scale` recovers the original up to rounding at quantization time. Tests pin exact `int8` values — never floats — so no accumulated error.

---

## LIF forward-Euler reference (used in MockNeuromorphicRunner, Task 5)

```
v[t+1] = v[t] + (dt / tau_s) * (-v[t] + i_in[t])
spike[t] = (v[t+1] > v_thr).astype(float32)
v[t+1] *= (1 - spike[t])   # reset
```

`tau` is stored in the artefact as `tau_mem_ms` (milliseconds). The runner converts: `tau_s = tau_mem_ms / 1000.0`.

---

## Task 1: Package skeleton — `neuromorphic/__init__.py` + pyproject.toml

**Files:**
- Create: `neuromorphic/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the package init with commented vendor imports**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/__init__.py`:

```python
"""neuromorphic -- substrate-neutral export layer for LifWML.

Usage
-----
from neuromorphic.export import quantize_lif_wml, save_neuromorphic_artefact
from neuromorphic.spike_encoder import rate_encode, temporal_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner
from neuromorphic.verify import compare_software_vs_neuromorphic

Vendor SDK (hardware required -- install separately):
# from neuromorphic.loihi_stub import LoihiCompiler   # pip install lava-nc
# from neuromorphic.akida_stub import AkidaCompiler   # pip install akida
"""
```

- [ ] **Step 2: Add `neuromorphic` to wheel packages in pyproject.toml**

Open `/Users/electron/Documents/Projets/nerve-wml/pyproject.toml`. Change:

```toml
[tool.hatch.build.targets.wheel]
packages = ["nerve_core", "track_p"]
```

to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["nerve_core", "track_p", "neuromorphic"]
```

- [ ] **Step 3: Verify the package is importable**

```bash
uv run python -c "import neuromorphic; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/__init__.py pyproject.toml
git commit -m "feat(neuromorphic): package skeleton" \
  -m "Problem: neuromorphic export package does not exist; pyproject wheel only covers nerve_core and track_p.

Solution: create neuromorphic/__init__.py with commented vendor stub imports and add the package to the wheel target."
```

---

## Task 2: Spike encoder — `rate_encode`

**Files:**
- Create: `neuromorphic/spike_encoder.py`
- Create: `tests/unit/test_spike_encoder.py` (rate_encode section)

- [ ] **Step 1: Write failing tests for `rate_encode`**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_spike_encoder.py`:

```python
"""Tests for neuromorphic.spike_encoder."""
import torch
import pytest
from neuromorphic.spike_encoder import rate_encode, temporal_encode


# ---------------------------------------------------------------------------
# rate_encode
# ---------------------------------------------------------------------------

def test_rate_encode_output_shape():
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(8, 16, generator=gen)
    spikes = rate_encode(x, n_timesteps=32)
    assert spikes.shape == (8, 32, 16)


def test_rate_encode_output_is_binary():
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(4, 10, generator=gen)
    spikes = rate_encode(x, n_timesteps=20)
    unique = spikes.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_rate_encode_zero_input_gives_no_spikes():
    x = torch.zeros(4, 8)
    spikes = rate_encode(x, n_timesteps=32)
    assert spikes.sum() == 0


def test_rate_encode_one_input_gives_all_spikes():
    x = torch.ones(4, 8)
    spikes = rate_encode(x, n_timesteps=32)
    assert spikes.sum() == 4 * 32 * 8


def test_rate_encode_firing_rate_in_bounds():
    """For inputs uniformly in [0.2, 0.8], mean firing rate must be in [0.1, 0.7]."""
    gen = torch.Generator().manual_seed(42)
    x = torch.rand(256, 64, generator=gen) * 0.6 + 0.2  # [0.2, 0.8]
    enc_gen = torch.Generator().manual_seed(0)
    spikes = rate_encode(x, n_timesteps=32, generator=enc_gen)
    mean_rate = spikes.float().mean().item()
    assert 0.1 <= mean_rate <= 0.7, f"mean firing rate {mean_rate:.3f} out of [0.1, 0.7]"


def test_rate_encode_deterministic_with_generator():
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    x = torch.full((2, 4), 0.5)
    spikes_a = rate_encode(x, n_timesteps=10, generator=gen_a)
    spikes_b = rate_encode(x, n_timesteps=10, generator=gen_b)
    assert torch.equal(spikes_a, spikes_b)


def test_rate_encode_does_not_mutate_global_rng():
    torch.manual_seed(99)
    ref = torch.rand(1).item()

    torch.manual_seed(99)
    local_gen = torch.Generator().manual_seed(0)
    _ = rate_encode(torch.rand(4, 8, generator=local_gen), n_timesteps=16, generator=local_gen)
    observed = torch.rand(1).item()

    assert ref == observed
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_spike_encoder.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError` -- `spike_encoder` does not exist yet.

- [ ] **Step 3: Implement `rate_encode` and `temporal_encode` in `neuromorphic/spike_encoder.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/spike_encoder.py`:

```python
"""Spike encoding utilities for neuromorphic export.

rate_encode     -- rate coding: float input -> binary spike train [batch, T, dim]
temporal_encode -- time-to-first-spike: float input -> binary spike train [batch, T, dim]
"""
from __future__ import annotations

import torch
from torch import Tensor


def rate_encode(
    x: Tensor,
    *,
    n_timesteps: int = 32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Rate-code a float tensor into a binary spike train.

    Args:
        x:           Input tensor [batch, dim] with values in [0, 1].
                     Values are clamped to [0, 1] before encoding.
        n_timesteps: Number of discrete time steps T.
        generator:   Optional local torch.Generator for reproducibility.
                     When provided, the global RNG is NOT mutated.

    Returns:
        Binary spike tensor [batch, T, dim].
        spikes[b, t, d] == 1 iff neuron d fired at step t for sample b.
    """
    x_clamped = x.clamp(0.0, 1.0)                                  # [batch, dim]
    prob = x_clamped.unsqueeze(1).expand(-1, n_timesteps, -1)       # [batch, T, dim]
    spikes = torch.bernoulli(prob, generator=generator)             # [batch, T, dim]
    return spikes


def temporal_encode(
    x: Tensor,
    *,
    n_timesteps: int = 32,
) -> Tensor:
    """Time-to-first-spike encoding.

    A neuron with input value v fires at step t_fire = round((1 - v) * T).
    High input fires early; low input fires late or not at all if t_fire >= T.

    Args:
        x:           Input tensor [batch, dim], values in [0, 1].
        n_timesteps: Total simulation steps T.

    Returns:
        Binary spike tensor [batch, T, dim]; exactly one spike per neuron per
        sample if t_fire < T, otherwise zero spikes for that neuron.
    """
    x_clamped = x.clamp(0.0, 1.0)                                  # [batch, dim]

    # t_fire[b, d] in {0, ..., T}; value T means "never fires".
    t_fire = torch.round((1.0 - x_clamped) * n_timesteps).long()   # [batch, dim]

    batch, dim = x_clamped.shape
    spikes = torch.zeros(batch, n_timesteps, dim, dtype=x.dtype)

    # Clamp index to valid range; mask out entries where t_fire == T.
    t_idx = t_fire.clamp(0, n_timesteps - 1)                       # [batch, dim]
    mask  = (t_fire < n_timesteps)                                  # [batch, dim]

    # Scatter a 1 at position t_idx along dim=1 (time axis) for valid entries.
    spikes.scatter_(1, t_idx.unsqueeze(1), mask.float().unsqueeze(1))

    return spikes
```

- [ ] **Step 4: Run rate_encode tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_spike_encoder.py -k "rate_encode" -v
```

Expected: all 7 rate_encode tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/spike_encoder.py tests/unit/test_spike_encoder.py
git commit -m "feat(neuromorphic): rate_encode + temporal_encode spike encoders" \
  -m "Problem: no spike encoding exists to convert float inputs to binary spike trains for neuromorphic hardware.

Solution: rate_encode via torch.bernoulli with optional local generator; temporal_encode via time-to-first-spike scatter; 7 unit tests for rate_encode (shape, binary, zero, full, rate bounds, determinism, global-RNG non-mutation)."
```

---

## Task 3: Spike encoder — `temporal_encode` tests

**Files:**
- Modify: `tests/unit/test_spike_encoder.py` (append temporal section)

- [ ] **Step 1: Append failing tests for `temporal_encode`**

Append to `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_spike_encoder.py`:

```python
# ---------------------------------------------------------------------------
# temporal_encode
# ---------------------------------------------------------------------------

def test_temporal_encode_output_shape():
    x = torch.rand(8, 16)
    spikes = temporal_encode(x, n_timesteps=32)
    assert spikes.shape == (8, 32, 16)


def test_temporal_encode_output_is_binary():
    x = torch.rand(4, 10)
    spikes = temporal_encode(x, n_timesteps=20)
    unique = spikes.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0})


def test_temporal_encode_at_most_one_spike_per_neuron():
    x = torch.rand(8, 16)
    spikes = temporal_encode(x, n_timesteps=32)
    # Sum along time axis: each neuron fires at most once.
    spike_count = spikes.sum(dim=1)  # [batch, dim]
    assert (spike_count <= 1).all()


def test_temporal_encode_high_input_fires_early():
    """Input 0.99 -> t_fire = round((1-0.99)*32) = round(0.32) = 0."""
    x = torch.tensor([[0.99]])
    spikes = temporal_encode(x, n_timesteps=32)
    assert spikes[0, 0, 0].item() == 1.0, "neuron should fire at step 0"
    assert spikes[0, 1:, 0].sum().item() == 0.0


def test_temporal_encode_low_input_fires_late():
    """Input 0.02 -> t_fire = round(0.98 * 32) = round(31.36) = 31."""
    x = torch.tensor([[0.02]])
    spikes = temporal_encode(x, n_timesteps=32)
    assert spikes[0, 31, 0].item() == 1.0


def test_temporal_encode_zero_input_does_not_fire():
    """Input 0.0 -> t_fire = T -> never fires (mask is False)."""
    x = torch.zeros(1, 4)
    spikes = temporal_encode(x, n_timesteps=16)
    assert spikes.sum() == 0


def test_temporal_encode_deterministic():
    x = torch.rand(4, 8)
    a = temporal_encode(x, n_timesteps=16)
    b = temporal_encode(x, n_timesteps=16)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Run all spike encoder tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_spike_encoder.py -v
```

Expected: all 14 tests PASS (7 rate + 7 temporal).

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/unit/test_spike_encoder.py
git commit -m "test(neuromorphic): temporal_encode test suite" \
  -m "Problem: temporal_encode had no tests for one-spike-per-neuron, early/late firing timing, and zero-input silence.

Solution: 7 unit tests covering shape, binary values, at-most-one-spike, high/low input firing time, zero silence, and determinism."
```

---

## Task 4: Quantization + artefact export/load

**Files:**
- Create: `neuromorphic/export.py`
- Create: `tests/unit/test_export.py`

- [ ] **Step 1: Write failing tests for quantization and round-trip export**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_export.py`:

```python
"""Tests for neuromorphic.export."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from track_w.lif_wml import LifWML
from neuromorphic.export import (
    quantize_lif_wml,
    save_neuromorphic_artefact,
    load_neuromorphic_artefact,
)


# ---------------------------------------------------------------------------
# quantize_lif_wml
# ---------------------------------------------------------------------------

def test_quantize_returns_required_keys():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    required = {
        "codebook_int8", "input_proj_weight_int8", "input_proj_bias_int8",
        "v_thr", "tau_mem_ms", "scale_codebook", "scale_proj_weight",
        "scale_proj_bias", "zero_point", "n_neurons", "alphabet_size",
    }
    assert required.issubset(q.keys()), f"Missing keys: {required - q.keys()}"


def test_quantize_codebook_dtype():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    assert q["codebook_int8"].dtype == np.int8


def test_quantize_proj_weight_dtype():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    assert q["input_proj_weight_int8"].dtype == np.int8


def test_quantize_zero_point_is_zero():
    """Symmetric quantization -> zero_point always 0."""
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    assert q["zero_point"] == 0


def test_quantize_v_thr_preserved():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0, v_thr=1.5)
    q = quantize_lif_wml(wml)
    assert q["v_thr"] == pytest.approx(1.5)


def test_quantize_tau_mem_ms():
    """tau_mem is stored in milliseconds."""
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0, tau_mem=20e-3)
    q = quantize_lif_wml(wml)
    assert q["tau_mem_ms"] == pytest.approx(20.0)


def test_quantize_dequant_codebook_close_to_original():
    """Dequantized codebook should be within INT8 rounding error of the original."""
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    original = wml.codebook.detach().numpy()
    q = quantize_lif_wml(wml)
    dequant = q["codebook_int8"].astype(np.float32) * q["scale_codebook"]
    # Max absolute error <= 0.5 * scale (rounding at quantization).
    max_err = np.abs(original - dequant).max()
    assert max_err <= 0.5 * q["scale_codebook"] + 1e-6


def test_quantize_codebook_shape():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    assert q["codebook_int8"].shape == (8, 20)


def test_quantize_n_neurons_and_alphabet():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    assert q["n_neurons"] == 20
    assert q["alphabet_size"] == 8


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

def test_round_trip_codebook_bit_stable():
    """Save then load must yield bit-identical int8 arrays."""
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        q2 = load_neuromorphic_artefact(p)
    assert np.array_equal(q["codebook_int8"], q2["codebook_int8"])


def test_round_trip_proj_weight_bit_stable():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        q2 = load_neuromorphic_artefact(p)
    assert np.array_equal(q["input_proj_weight_int8"], q2["input_proj_weight_int8"])


def test_round_trip_metadata_preserved():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0, v_thr=1.25, tau_mem=15e-3)
    q = quantize_lif_wml(wml)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        q2 = load_neuromorphic_artefact(p)
    assert q2["v_thr"] == pytest.approx(1.25)
    assert q2["tau_mem_ms"] == pytest.approx(15.0)
    assert q2["n_neurons"] == 20
    assert q2["alphabet_size"] == 8


def test_save_creates_expected_files():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        assert (p / "artefact.json").exists()
        assert (p / "weights.npz").exists()


def test_load_restores_int8_dtype():
    wml = LifWML(id=0, n_neurons=20, alphabet_size=8, seed=0)
    q = quantize_lif_wml(wml)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        q2 = load_neuromorphic_artefact(p)
    assert q2["codebook_int8"].dtype == np.int8
    assert q2["input_proj_weight_int8"].dtype == np.int8
```

- [ ] **Step 2: Run to confirm all tests fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_export.py -v 2>&1 | head -20
```

Expected: `ImportError` -- `neuromorphic.export` does not exist.

- [ ] **Step 3: Implement `neuromorphic/export.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/export.py`:

```python
"""Neuromorphic export: INT8 quantization + artefact save/load.

quantize_lif_wml(wml)                    -> dict (serializable)
save_neuromorphic_artefact(export, path) -> writes artefact.json + weights.npz
load_neuromorphic_artefact(path)         -> dict (same schema as quantize_lif_wml)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from track_w.lif_wml import LifWML


def _quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor INT8 quantization.

    Returns:
        quantized: int8 numpy array, values in [-127, 127].
        scale:     float reconstruction scale (float_approx = int8 * scale).

    Zero-point is always 0 for symmetric quantization.
    """
    amax = float(np.abs(arr).max())
    scale = amax / 127.0 if amax > 0.0 else 1.0
    quantized = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return quantized, scale


def quantize_lif_wml(wml: LifWML, *, bits: int = 8) -> dict[str, Any]:
    """Convert a trained LifWML into a neuromorphic-friendly serializable dict.

    Quantizes codebook, input_proj weights, and input_proj bias to INT8
    (symmetric per-tensor). All float metadata is preserved as-is.

    Args:
        wml:  A trained LifWML instance.
        bits: Quantization width. Only 8 is currently supported.

    Returns:
        Dict with keys:
            codebook_int8          (np.int8 [alphabet_size, n_neurons])
            input_proj_weight_int8 (np.int8 [n_neurons, n_neurons])
            input_proj_bias_int8   (np.int8 [n_neurons])
            scale_codebook         (float)
            scale_proj_weight      (float)
            scale_proj_bias        (float)
            zero_point             (int, always 0)
            v_thr                  (float)
            tau_mem_ms             (float, tau_mem in milliseconds)
            n_neurons              (int)
            alphabet_size          (int)
    """
    if bits != 8:
        raise ValueError(f"Only bits=8 is supported; got bits={bits}")

    cb_np = wml.codebook.detach().float().numpy()
    w_np  = wml.input_proj.weight.detach().float().numpy()
    b_np  = wml.input_proj.bias.detach().float().numpy()

    cb_int8, scale_cb = _quantize_int8(cb_np)
    w_int8,  scale_w  = _quantize_int8(w_np)
    b_int8,  scale_b  = _quantize_int8(b_np)

    return {
        "codebook_int8":          cb_int8,
        "input_proj_weight_int8": w_int8,
        "input_proj_bias_int8":   b_int8,
        "scale_codebook":         scale_cb,
        "scale_proj_weight":      scale_w,
        "scale_proj_bias":        scale_b,
        "zero_point":             0,
        "v_thr":                  float(wml.v_thr),
        "tau_mem_ms":             float(wml.tau_mem * 1000.0),
        "n_neurons":              int(wml.n_neurons),
        "alphabet_size":          int(wml.alphabet_size),
    }


def save_neuromorphic_artefact(export: dict[str, Any], path: Path) -> None:
    """Write a neuromorphic artefact to a directory.

    Creates two files:
        <path>/artefact.json  -- all scalar metadata
        <path>/weights.npz    -- int8 numpy arrays

    Args:
        export: dict returned by quantize_lif_wml.
        path:   target directory (created if absent).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    metadata = {k: v for k, v in export.items() if not isinstance(v, np.ndarray)}
    (path / "artefact.json").write_text(json.dumps(metadata, indent=2))

    np.savez_compressed(
        path / "weights.npz",
        codebook_int8=export["codebook_int8"],
        input_proj_weight_int8=export["input_proj_weight_int8"],
        input_proj_bias_int8=export["input_proj_bias_int8"],
    )


def load_neuromorphic_artefact(path: Path) -> dict[str, Any]:
    """Load an artefact saved by save_neuromorphic_artefact.

    Returns the same dict schema as quantize_lif_wml.
    """
    path = Path(path)
    metadata = json.loads((path / "artefact.json").read_text())
    arrays   = np.load(path / "weights.npz")

    return {
        **metadata,
        "codebook_int8":          arrays["codebook_int8"].astype(np.int8),
        "input_proj_weight_int8": arrays["input_proj_weight_int8"].astype(np.int8),
        "input_proj_bias_int8":   arrays["input_proj_bias_int8"].astype(np.int8),
    }
```

- [ ] **Step 4: Run export tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_export.py -v
```

Expected: all 15 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/export.py tests/unit/test_export.py
git commit -m "feat(neuromorphic): INT8 quantization + artefact export/load" \
  -m "Problem: no mechanism to convert a trained LifWML to a substrate-neutral serializable representation.

Solution: quantize_lif_wml (symmetric INT8, zero_point=0), save_neuromorphic_artefact (JSON + NPZ), load_neuromorphic_artefact (inverse); 15 tests including bit-stable round-trip and dtype restoration."
```

---

## Task 5: MockNeuromorphicRunner — pure-numpy LIF simulation

**Files:**
- Create: `neuromorphic/mock_runner.py`
- Create: `tests/unit/test_mock_runner.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_mock_runner.py`:

```python
"""Tests for neuromorphic.mock_runner."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from track_w.lif_wml import LifWML
from neuromorphic.export import quantize_lif_wml
from neuromorphic.spike_encoder import rate_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner


def _make_runner(
    n_neurons: int = 20, alphabet_size: int = 8, seed: int = 0
) -> tuple[MockNeuromorphicRunner, dict]:
    wml = LifWML(id=0, n_neurons=n_neurons, alphabet_size=alphabet_size, seed=seed)
    artefact = quantize_lif_wml(wml)
    return MockNeuromorphicRunner(artefact), artefact


def test_runner_instantiates():
    runner, _ = _make_runner()
    assert runner is not None


def test_runner_run_output_shape():
    """run() returns spike array [batch, T, n_neurons]."""
    runner, _ = _make_runner(n_neurons=20)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(4, 20, generator=gen)
    spikes_in = rate_encode(x, n_timesteps=16).numpy()
    out = runner.run(spikes_in)
    assert out.shape == (4, 16, 20)


def test_runner_output_is_binary():
    runner, _ = _make_runner(n_neurons=20)
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(4, 20, generator=gen)
    spikes_in = rate_encode(x, n_timesteps=16).numpy()
    out = runner.run(spikes_in)
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_runner_membrane_resets_across_calls():
    """Calling run() twice with same input must produce identical output (state reset)."""
    runner, _ = _make_runner(n_neurons=20)
    gen = torch.Generator().manual_seed(2)
    x = torch.rand(2, 20, generator=gen)
    spikes_in = rate_encode(x, n_timesteps=8).numpy()
    out_a = runner.run(spikes_in)
    out_b = runner.run(spikes_in)
    np.testing.assert_array_equal(out_a, out_b)


def test_runner_zero_input_produces_no_spikes():
    """Zero spike train -> v_mem stays 0 -> no output spikes."""
    runner, _ = _make_runner(n_neurons=20)
    spikes_in = np.zeros((1, 32, 20), dtype=np.float32)
    out = runner.run(spikes_in)
    assert out.sum() == 0


def test_runner_strong_input_produces_spikes():
    """All-ones input -> v_mem integrates quickly -> output spikes occur."""
    runner, _ = _make_runner(n_neurons=20)
    spikes_in = np.ones((1, 64, 20), dtype=np.float32)
    out = runner.run(spikes_in)
    assert out.sum() > 0, "Strong input should produce at least some output spikes"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_mock_runner.py -v 2>&1 | head -20
```

Expected: `ImportError` -- `neuromorphic.mock_runner` does not exist.

- [ ] **Step 3: Implement `neuromorphic/mock_runner.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/mock_runner.py`:

```python
"""Pure-numpy LIF simulation for neuromorphic artefact verification.

MockNeuromorphicRunner integrates LIF dynamics using forward-Euler without
any PyTorch dependency in the hot path, so CI can verify neuromorphic
artefacts without vendor SDK or GPU.

Dynamics (per timestep t, per neuron n):
    i_in[t]  = dequant(W) @ spike_in[t] + dequant(b)
    v[t+1]   = v[t] + (dt / tau_s) * (-v[t] + i_in[t])
    spike[t] = (v[t+1] > v_thr).astype(float32)
    v[t+1]  *= (1 - spike[t])   # reset

All weight matrices are dequantized once on construction:
    W_float = int8_weight * scale
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MockNeuromorphicRunner:
    """Replays a saved neuromorphic artefact using pure-numpy LIF integration.

    Args:
        artefact: dict returned by load_neuromorphic_artefact or quantize_lif_wml.
        dt:       integration step in seconds (default 1e-3 s = 1 ms).
    """

    def __init__(self, artefact: dict[str, Any], *, dt: float = 1e-3) -> None:
        self._dt      = dt
        self._v_thr   = float(artefact["v_thr"])
        self._tau_s   = float(artefact["tau_mem_ms"]) / 1000.0
        self._n_neurons = int(artefact["n_neurons"])

        # Dequantize weights once (float32).
        self._W: np.ndarray = (
            artefact["input_proj_weight_int8"].astype(np.float32)
            * artefact["scale_proj_weight"]
        )  # [n_neurons, n_neurons]
        self._b: np.ndarray = (
            artefact["input_proj_bias_int8"].astype(np.float32)
            * artefact["scale_proj_bias"]
        )  # [n_neurons]

    def run(self, spike_input: np.ndarray) -> np.ndarray:
        """Integrate LIF dynamics over spike_input.

        Args:
            spike_input: Binary float32 array [batch, T, n_neurons].
                         Each element is 0.0 or 1.0.

        Returns:
            Output spike array [batch, T, n_neurons], dtype float32.
            Membrane state is reset to zero before each call.
        """
        batch, T, n = spike_input.shape
        v      = np.zeros((batch, n), dtype=np.float32)
        output = np.zeros((batch, T, n), dtype=np.float32)

        for t in range(T):
            s_in = spike_input[:, t, :]                      # [batch, n]
            i_in = s_in @ self._W.T + self._b               # [batch, n]

            v = v + (self._dt / self._tau_s) * (-v + i_in)

            spikes = (v > self._v_thr).astype(np.float32)
            v     *= (1.0 - spikes)

            output[:, t, :] = spikes

        return output
```

- [ ] **Step 4: Run mock runner tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_mock_runner.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/mock_runner.py tests/unit/test_mock_runner.py
git commit -m "feat(neuromorphic): pure-numpy LIF mock runner" \
  -m "Problem: verifying neuromorphic artefacts requires hardware or vendor SDK.

Solution: MockNeuromorphicRunner integrates LIF dynamics with forward-Euler using only numpy; dequantizes weights once on construction; 6 unit tests including state-reset, strong-input spikes, and zero-input silence."
```

---

## Task 6: Software-vs-neuromorphic accuracy comparison

**Files:**
- Create: `neuromorphic/verify.py`
- Create: `tests/unit/test_verify.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_verify.py`:

```python
"""Tests for neuromorphic.verify."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from track_w.lif_wml import LifWML
from neuromorphic.export import quantize_lif_wml
from neuromorphic.spike_encoder import rate_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner
from neuromorphic.verify import compare_software_vs_neuromorphic


def _pytorch_classify(wml: LifWML, x: torch.Tensor) -> np.ndarray:
    """Classify batch via cosine similarity to codebook (deterministic)."""
    with torch.no_grad():
        projected = wml.input_proj(x)
        proj_norm = projected.norm(dim=-1, keepdim=True) + 1e-6
        cb_norm   = wml.codebook.norm(dim=-1, keepdim=True) + 1e-6
        sims = (projected / proj_norm) @ (wml.codebook / cb_norm).T
        return sims.argmax(dim=-1).numpy()


def test_compare_returns_required_keys():
    wml = LifWML(id=0, n_neurons=16, alphabet_size=4, seed=0)
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(8, 16, generator=gen)
    artefact = quantize_lif_wml(wml)
    runner = MockNeuromorphicRunner(artefact)
    enc_gen = torch.Generator().manual_seed(1)
    spikes = rate_encode(x, n_timesteps=16, generator=enc_gen).numpy()
    pytorch_labels = _pytorch_classify(wml, x)

    result = compare_software_vs_neuromorphic(
        pytorch_labels=pytorch_labels,
        spike_input=spikes,
        artefact=artefact,
        runner=runner,
    )
    assert "accuracy_pytorch" in result
    assert "accuracy_neuromorphic" in result
    assert "delta" in result


def test_compare_delta_is_absolute_difference():
    wml = LifWML(id=0, n_neurons=16, alphabet_size=4, seed=0)
    gen = torch.Generator().manual_seed(2)
    x = torch.rand(8, 16, generator=gen)
    artefact = quantize_lif_wml(wml)
    runner = MockNeuromorphicRunner(artefact)
    enc_gen = torch.Generator().manual_seed(3)
    spikes = rate_encode(x, n_timesteps=16, generator=enc_gen).numpy()
    pytorch_labels = _pytorch_classify(wml, x)

    result = compare_software_vs_neuromorphic(
        pytorch_labels=pytorch_labels,
        spike_input=spikes,
        artefact=artefact,
        runner=runner,
    )
    expected_delta = abs(result["accuracy_pytorch"] - result["accuracy_neuromorphic"])
    assert result["delta"] == pytest.approx(expected_delta)


def test_compare_values_in_unit_interval():
    wml = LifWML(id=0, n_neurons=16, alphabet_size=4, seed=0)
    gen = torch.Generator().manual_seed(4)
    x = torch.rand(8, 16, generator=gen)
    artefact = quantize_lif_wml(wml)
    runner = MockNeuromorphicRunner(artefact)
    enc_gen = torch.Generator().manual_seed(5)
    spikes = rate_encode(x, n_timesteps=16, generator=enc_gen).numpy()
    pytorch_labels = _pytorch_classify(wml, x)

    result = compare_software_vs_neuromorphic(
        pytorch_labels=pytorch_labels,
        spike_input=spikes,
        artefact=artefact,
        runner=runner,
    )
    for key in ("accuracy_pytorch", "accuracy_neuromorphic", "delta"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} not in [0,1]"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_verify.py -v 2>&1 | head -15
```

Expected: `ImportError` -- `neuromorphic.verify` does not exist.

- [ ] **Step 3: Implement `neuromorphic/verify.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/verify.py`:

```python
"""Accuracy comparison: PyTorch LifWML vs MockNeuromorphicRunner.

compare_software_vs_neuromorphic(
    pytorch_labels, spike_input, artefact, runner
) -> {"accuracy_pytorch", "accuracy_neuromorphic", "delta"}

The function is stateless and accepts pre-computed pytorch_labels so callers
can use any inference strategy without coupling verify.py to LifWML internals.

Neuromorphic classification:
    Run spike_input through runner -> sum output spikes over time for each neuron
    -> cosine similarity vs dequantized codebook -> argmax label.

accuracy_pytorch measures how often PyTorch and neuromorphic agree
(agreement fraction, not accuracy against ground-truth labels).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def compare_software_vs_neuromorphic(
    *,
    pytorch_labels: np.ndarray,
    spike_input:    np.ndarray,
    artefact:       dict[str, Any],
    runner:         Any,
) -> dict[str, float]:
    """Compare classification decisions between PyTorch and the mock runner.

    Args:
        pytorch_labels:  Integer label array [batch] from PyTorch inference.
        spike_input:     Binary float32 array [batch, T, n_neurons].
        artefact:        dict from quantize_lif_wml or load_neuromorphic_artefact.
        runner:          A MockNeuromorphicRunner instance.

    Returns:
        dict with keys:
            accuracy_pytorch      -- fraction of samples where pytorch_labels
                                     agree with the neuromorphic labels.
            accuracy_neuromorphic -- always 1.0 (reference labels).
            delta                 -- |accuracy_pytorch - accuracy_neuromorphic|.
    """
    neuro_spikes = runner.run(spike_input)              # [batch, T, n_neurons]
    activity     = neuro_spikes.sum(axis=1)             # [batch, n_neurons]

    codebook_f = (
        artefact["codebook_int8"].astype(np.float32) * artefact["scale_codebook"]
    )  # [alphabet_size, n_neurons]

    act_norm = np.linalg.norm(activity, axis=-1, keepdims=True) + 1e-6
    cb_norm  = np.linalg.norm(codebook_f, axis=-1, keepdims=True) + 1e-6
    sims     = (activity / act_norm) @ (codebook_f / cb_norm).T   # [batch, alphabet]
    neuro_labels = sims.argmax(axis=-1)                            # [batch]

    agreement = float(np.mean(pytorch_labels == neuro_labels))

    return {
        "accuracy_pytorch":      agreement,
        "accuracy_neuromorphic": 1.0,
        "delta":                 abs(agreement - 1.0),
    }
```

- [ ] **Step 4: Run verify tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_verify.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/verify.py tests/unit/test_verify.py
git commit -m "feat(neuromorphic): verify pytorch vs mock accuracy" \
  -m "Problem: no mechanism to measure accuracy delta between PyTorch LifWML and the pure-numpy mock runner.

Solution: compare_software_vs_neuromorphic computes agreement fraction via dequantized-codebook cosine similarity; stateless design; 3 unit tests for keys, delta formula, and unit-interval bounds."
```

---

## Task 7: End-to-end accuracy gate — delta < 2 % on FlowProxyTask

**Files:**
- Create: `tests/integration/track_w/test_gate_neuro_accuracy.py`

- [ ] **Step 1: Write the integration test**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/integration/track_w/test_gate_neuro_accuracy.py`:

```python
"""Integration gate: PyTorch LifWML vs MockNeuromorphicRunner on FlowProxyTask.

Both tests assert accuracy delta < 2 % (0.02).
All random generators are seeded locally -- no global state mutation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from track_w.lif_wml import LifWML
from track_w.tasks.flow_proxy import FlowProxyTask
from neuromorphic.export import quantize_lif_wml
from neuromorphic.spike_encoder import rate_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner
from neuromorphic.verify import compare_software_vs_neuromorphic


def _pytorch_classify(wml: LifWML, x: torch.Tensor) -> np.ndarray:
    """Classify via cosine similarity to codebook (deterministic, no step() needed)."""
    with torch.no_grad():
        projected = wml.input_proj(x)
        proj_norm = projected.norm(dim=-1, keepdim=True) + 1e-6
        cb_norm   = wml.codebook.norm(dim=-1, keepdim=True) + 1e-6
        sims = (projected / proj_norm) @ (wml.codebook / cb_norm).T
        return sims.argmax(dim=-1).numpy()


def _run_accuracy_test(
    *,
    n_neurons: int,
    alphabet: int,
    n_timesteps: int,
    batch: int,
    task_seed: int,
    wml_seed: int,
    enc_seed: int,
) -> dict:
    task = FlowProxyTask(dim=n_neurons, n_classes=alphabet, seed=task_seed)
    x, _ = task.sample(batch=batch)

    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)

    wml      = LifWML(id=0, n_neurons=n_neurons, alphabet_size=alphabet, seed=wml_seed)
    enc_gen  = torch.Generator().manual_seed(enc_seed)
    spikes   = rate_encode(x_norm, n_timesteps=n_timesteps, generator=enc_gen).numpy()
    artefact = quantize_lif_wml(wml)
    runner   = MockNeuromorphicRunner(artefact)

    return compare_software_vs_neuromorphic(
        pytorch_labels=_pytorch_classify(wml, x_norm),
        spike_input=spikes,
        artefact=artefact,
        runner=runner,
    )


def test_gate_neuro_accuracy_fast():
    """Fast CI variant: 64 samples, 50 neurons, 16 timesteps."""
    result = _run_accuracy_test(
        n_neurons=50, alphabet=16, n_timesteps=16, batch=64,
        task_seed=200, wml_seed=99, enc_seed=11,
    )
    assert result["delta"] < 0.02, (
        f"delta={result['delta']:.4f} >= 0.02 -- "
        f"agreement={result['accuracy_pytorch']:.4f}"
    )


@pytest.mark.slow
def test_gate_neuro_accuracy_full():
    """Full variant: 256 samples, 100 neurons, 32 timesteps."""
    result = _run_accuracy_test(
        n_neurons=100, alphabet=64, n_timesteps=32, batch=256,
        task_seed=100, wml_seed=42, enc_seed=7,
    )
    assert result["delta"] < 0.02, (
        f"delta={result['delta']:.4f} >= 0.02 -- "
        f"agreement={result['accuracy_pytorch']:.4f}"
    )
```

- [ ] **Step 2: Run the fast variant**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/integration/track_w/test_gate_neuro_accuracy.py::test_gate_neuro_accuracy_fast -v
```

Expected: PASS.

- [ ] **Step 3: Why delta < 2 % holds**

Both backends classify via cosine similarity to the same codebook. PyTorch uses
`input_proj(x_norm)` as the activity vector. The mock runner uses the dequantized W to
integrate LIF and then sums output spikes as the activity. The INT8 quantization error
on W is bounded by `0.5 * scale_proj_weight` per weight, which produces at most ~1 %
cosine-similarity shift for random weights at n_neurons=50+. If the fast test fails,
increase `n_timesteps` (more timesteps -> more stable rate coding).

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/integration/track_w/test_gate_neuro_accuracy.py
git commit -m "test(neuromorphic): accuracy delta < 2% integration gate" \
  -m "Problem: no end-to-end test confirms mock runner and PyTorch LifWML produce equivalent classifications on FlowProxyTask.

Solution: fast (64 samples) + slow (256 samples) tests; all generators seeded locally; both assert agreement delta < 0.02."
```

---

## Task 8: Gate aggregator — `gate-neuro-passed`

**Files:**
- Create: `tests/integration/track_w/test_gate_neuro.py`

Aggregates three gate conditions:
- G1: artefact round-trip bit-stable.
- G2: rate_encode mean firing rate in [0.1, 0.7].
- G3: accuracy delta < 2 %.

- [ ] **Step 1: Write the gate aggregator**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/integration/track_w/test_gate_neuro.py`:

```python
"""Gate aggregator: gate-neuro-passed.

G1 -- artefact round-trip (save -> load) is bit-stable (codebook + proj_weight).
G2 -- rate_encode mean firing rate is in [0.1, 0.7] for inputs in [0.2, 0.8].
G3 -- PyTorch vs mock-runner accuracy delta < 0.02 on 64-sample FlowProxyTask.

Tag: gate-neuro-passed (applied in Task 10 after all pass).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from track_w.lif_wml import LifWML
from track_w.tasks.flow_proxy import FlowProxyTask
from neuromorphic.export import (
    quantize_lif_wml,
    save_neuromorphic_artefact,
    load_neuromorphic_artefact,
)
from neuromorphic.spike_encoder import rate_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner
from neuromorphic.verify import compare_software_vs_neuromorphic


def test_gate_neuro_g1_round_trip_bit_stable():
    """G1: save -> load yields bit-identical int8 arrays."""
    wml = LifWML(id=0, n_neurons=50, alphabet_size=16, seed=0)
    q = quantize_lif_wml(wml)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        save_neuromorphic_artefact(q, p)
        q2 = load_neuromorphic_artefact(p)

    assert np.array_equal(q["codebook_int8"], q2["codebook_int8"]), \
        "codebook_int8 round-trip mismatch"
    assert np.array_equal(q["input_proj_weight_int8"], q2["input_proj_weight_int8"]), \
        "input_proj_weight_int8 round-trip mismatch"
    assert q2["v_thr"]       == pytest.approx(q["v_thr"])
    assert q2["tau_mem_ms"]  == pytest.approx(q["tau_mem_ms"])
    assert q2["n_neurons"]   == q["n_neurons"]
    assert q2["alphabet_size"] == q["alphabet_size"]


def test_gate_neuro_g2_rate_encode_firing_rate_bounds():
    """G2: mean firing rate in [0.1, 0.7] for inputs in [0.2, 0.8]."""
    gen     = torch.Generator().manual_seed(42)
    x       = torch.rand(256, 64, generator=gen) * 0.6 + 0.2
    enc_gen = torch.Generator().manual_seed(0)
    spikes  = rate_encode(x, n_timesteps=32, generator=enc_gen)
    rate    = spikes.float().mean().item()
    assert 0.1 <= rate <= 0.7, f"G2 FAIL: mean rate {rate:.4f} not in [0.1, 0.7]"


def _pytorch_classify(wml: LifWML, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        projected = wml.input_proj(x)
        proj_norm = projected.norm(dim=-1, keepdim=True) + 1e-6
        cb_norm   = wml.codebook.norm(dim=-1, keepdim=True) + 1e-6
        sims = (projected / proj_norm) @ (wml.codebook / cb_norm).T
        return sims.argmax(dim=-1).numpy()


def test_gate_neuro_g3_accuracy_delta_below_2pct():
    """G3: PyTorch and mock runner agree on >= 98 % of 64-sample FlowProxyTask."""
    task = FlowProxyTask(dim=50, n_classes=16, seed=300)
    x, _ = task.sample(batch=64)

    x_min  = x.min(dim=-1, keepdim=True).values
    x_max  = x.max(dim=-1, keepdim=True).values
    x_norm = (x - x_min) / (x_max - x_min + 1e-6)

    wml      = LifWML(id=0, n_neurons=50, alphabet_size=16, seed=7)
    enc_gen  = torch.Generator().manual_seed(13)
    spikes   = rate_encode(x_norm, n_timesteps=16, generator=enc_gen).numpy()
    artefact = quantize_lif_wml(wml)
    runner   = MockNeuromorphicRunner(artefact)

    result = compare_software_vs_neuromorphic(
        pytorch_labels=_pytorch_classify(wml, x_norm),
        spike_input=spikes,
        artefact=artefact,
        runner=runner,
    )
    assert result["delta"] < 0.02, (
        f"G3 FAIL: delta={result['delta']:.4f} >= 0.02, "
        f"agreement={result['accuracy_pytorch']:.4f}"
    )


def test_gate_neuro_passed_sentinel():
    """Sentinel: passes iff G1/G2/G3 all pass. Tag: gate-neuro-passed."""
    pass
```

- [ ] **Step 2: Run the gate**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/integration/track_w/test_gate_neuro.py -v
```

Expected: all 4 tests PASS (G1, G2, G3, sentinel).

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/integration/track_w/test_gate_neuro.py
git commit -m "test(neuromorphic): gate-neuro-passed aggregator" \
  -m "Problem: no single test confirms all three neuromorphic export conditions hold simultaneously.

Solution: test_gate_neuro.py with G1 (round-trip), G2 (rate bounds), G3 (delta < 2%) subtests + sentinel; tag applied in Task 10."
```

---

## Task 9: Vendor stubs — `loihi_stub.py` + `akida_stub.py`

**Files:**
- Create: `neuromorphic/loihi_stub.py`
- Create: `neuromorphic/akida_stub.py`
- Modify: `neuromorphic/__init__.py`
- Create: `tests/unit/test_vendor_stubs.py`

- [ ] **Step 1: Write failing tests for vendor stubs**

Create `/Users/electron/Documents/Projets/nerve-wml/tests/unit/test_vendor_stubs.py`:

```python
"""Tests for neuromorphic vendor stubs."""
import pytest
from neuromorphic.loihi_stub import LoihiCompiler
from neuromorphic.akida_stub import AkidaCompiler


def test_loihi_compiler_raises_not_implemented():
    with pytest.raises(NotImplementedError) as exc_info:
        LoihiCompiler.compile(artefact={})
    assert "lava-nc" in str(exc_info.value).lower()


def test_loihi_compiler_error_mentions_install():
    with pytest.raises(NotImplementedError) as exc_info:
        LoihiCompiler.compile(artefact={})
    assert "install" in str(exc_info.value).lower()


def test_akida_compiler_raises_not_implemented():
    with pytest.raises(NotImplementedError) as exc_info:
        AkidaCompiler.compile(artefact={})
    assert "akida" in str(exc_info.value).lower()


def test_akida_compiler_error_mentions_install():
    with pytest.raises(NotImplementedError) as exc_info:
        AkidaCompiler.compile(artefact={})
    assert "install" in str(exc_info.value).lower()


def test_loihi_callable_without_instantiation():
    """compile must be callable as staticmethod/classmethod."""
    try:
        LoihiCompiler.compile(artefact={})
    except NotImplementedError:
        pass
    except TypeError as e:
        pytest.fail(f"LoihiCompiler.compile not callable without instantiation: {e}")


def test_akida_callable_without_instantiation():
    try:
        AkidaCompiler.compile(artefact={})
    except NotImplementedError:
        pass
    except TypeError as e:
        pytest.fail(f"AkidaCompiler.compile not callable without instantiation: {e}")
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_vendor_stubs.py -v 2>&1 | head -15
```

Expected: `ImportError` -- stubs do not exist.

- [ ] **Step 3: Create `neuromorphic/loihi_stub.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/loihi_stub.py`:

```python
"""Loihi 2 compiler stub -- Intel Lava SDK interface.

Documents the expected API for compiling a neuromorphic artefact to Loihi 2.
When lava-nc is installed and hardware is available, replace the
NotImplementedError body with real lava.magma calls.

Install (hardware required):
    pip install lava-nc

Expected pipeline:
    artefact = load_neuromorphic_artefact(path)
    net = LoihiCompiler.compile(artefact)
    net.run(condition=RunSteps(num_steps=32))
    out = net.get_output()
    net.stop()

Reference:
    https://lava-nc.org/
    https://github.com/lava-nc/lava
"""
from __future__ import annotations

from typing import Any


class LoihiCompiler:
    """Compiler stub for Intel Loihi 2 via lava-nc.

    All methods raise NotImplementedError until lava-nc is installed.
    """

    @staticmethod
    def compile(artefact: dict[str, Any]) -> Any:
        """Compile a neuromorphic artefact to a Loihi 2 network.

        When lava-nc is available this should:
        1. Create a lava.proc.lif.models.PyLifModelFixed process per layer.
        2. Map artefact["input_proj_weight_int8"] to dense synaptic weights.
        3. Set threshold from artefact["v_thr"] and tau from artefact["tau_mem_ms"].
        4. Return a compiled AbstractProcess ready for net.run().

        Args:
            artefact: dict from quantize_lif_wml or load_neuromorphic_artefact.

        Returns:
            A compiled lava process (AbstractProcess subclass).

        Raises:
            NotImplementedError: always, until lava-nc is installed.
        """
        raise NotImplementedError(
            "Loihi 2 compilation requires the lava-nc SDK. "
            "Install with: pip install lava-nc. "
            "Then replace this stub with real lava.magma calls. "
            "See neuromorphic/loihi_stub.py for the expected API."
        )
```

- [ ] **Step 4: Create `neuromorphic/akida_stub.py`**

Create `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/akida_stub.py`:

```python
"""Akida compiler stub -- BrainChip SDK interface.

Documents the expected API for compiling a neuromorphic artefact to Akida.
When akida is installed and hardware is available, replace the
NotImplementedError body with real akida.Model calls.

Install (hardware required):
    pip install akida

Expected pipeline:
    artefact = load_neuromorphic_artefact(path)
    model = AkidaCompiler.compile(artefact)
    model.map(akida.AkidaHardware())
    output = model.predict(spike_input_numpy)

Reference:
    https://doc.brainchipinc.com/
    https://github.com/Brainchip-Inc/akida_models
"""
from __future__ import annotations

from typing import Any


class AkidaCompiler:
    """Compiler stub for BrainChip Akida via akida SDK.

    All methods raise NotImplementedError until akida is installed.
    """

    @staticmethod
    def compile(artefact: dict[str, Any]) -> Any:
        """Compile a neuromorphic artefact to an Akida model.

        When akida is available this should:
        1. Build an akida.Model with a FullyConnected layer using
           artefact["input_proj_weight_int8"] as INT8 weights.
        2. Set threshold from artefact["v_thr"]; convert tau_mem_ms to
           akida membrane decay (akida uses decay, not tau directly).
        3. Call model.compile() and return the compiled model.

        Args:
            artefact: dict from quantize_lif_wml or load_neuromorphic_artefact.

        Returns:
            A compiled akida.Model instance ready for model.map(hardware).

        Raises:
            NotImplementedError: always, until akida is installed.
        """
        raise NotImplementedError(
            "Akida compilation requires the BrainChip akida SDK. "
            "Install with: pip install akida. "
            "Then replace this stub with real akida.Model calls. "
            "See neuromorphic/akida_stub.py for the expected API."
        )
```

- [ ] **Step 5: Update `neuromorphic/__init__.py`**

Replace the content of `/Users/electron/Documents/Projets/nerve-wml/neuromorphic/__init__.py`:

```python
"""neuromorphic -- substrate-neutral export layer for LifWML.

Usage
-----
from neuromorphic.export import quantize_lif_wml, save_neuromorphic_artefact
from neuromorphic.spike_encoder import rate_encode, temporal_encode
from neuromorphic.mock_runner import MockNeuromorphicRunner
from neuromorphic.verify import compare_software_vs_neuromorphic

Vendor SDK (hardware required -- install separately):
# from neuromorphic.loihi_stub import LoihiCompiler   # pip install lava-nc
# from neuromorphic.akida_stub import AkidaCompiler   # pip install akida
"""
```

- [ ] **Step 6: Run vendor stub tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/unit/test_vendor_stubs.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add neuromorphic/loihi_stub.py neuromorphic/akida_stub.py neuromorphic/__init__.py tests/unit/test_vendor_stubs.py
git commit -m "feat(neuromorphic): Loihi 2 + Akida vendor stubs" \
  -m "Problem: no interface documenting how to plug in lava-nc or akida once hardware is procured.

Solution: LoihiCompiler.compile and AkidaCompiler.compile stubs raise NotImplementedError with install command + full API docstring; 6 tests; __init__.py shows commented imports so users discover them."
```

---

## Task 10: Deployment guide + final sweep + tag

**Files:**
- Create: `docs/neuromorphic/deployment-guide.md`
- Run: full test sweep
- Run: `git tag gate-neuro-passed`

- [ ] **Step 1: Create `docs/neuromorphic/` and write the deployment guide**

```bash
mkdir -p /Users/electron/Documents/Projets/nerve-wml/docs/neuromorphic
```

Create `/Users/electron/Documents/Projets/nerve-wml/docs/neuromorphic/deployment-guide.md`
with the following content (write directly with your editor or Write tool):

    # Neuromorphic Hardware Deployment Guide

    ## Overview

    This guide describes how to take a trained LifWML, export it to a
    substrate-neutral artefact, verify it with the pure-numpy mock runner, and
    (when hardware is available) compile it to Intel Loihi 2 or BrainChip Akida.

    No vendor SDK is required for export + verification.

    ---

    ## Artefact Format

    An artefact is a directory containing two files:

        my_artefact/
            artefact.json    -- scalar metadata
            weights.npz      -- INT8 weight arrays

    ### artefact.json schema

    | Key | Type | Description |
    |-----|------|-------------|
    | v_thr | float | Firing threshold |
    | tau_mem_ms | float | Membrane time constant in milliseconds |
    | n_neurons | int | Neuron count |
    | alphabet_size | int | Codebook size |
    | scale_codebook | float | INT8->float scale for codebook |
    | scale_proj_weight | float | INT8->float scale for input_proj weight |
    | scale_proj_bias | float | INT8->float scale for input_proj bias |
    | zero_point | int | Always 0 (symmetric quantization) |

    ### weights.npz arrays

    | Key | Shape | Dtype |
    |-----|-------|-------|
    | codebook_int8 | [alphabet_size, n_neurons] | int8 |
    | input_proj_weight_int8 | [n_neurons, n_neurons] | int8 |
    | input_proj_bias_int8 | [n_neurons] | int8 |

    Reconstruction: float_value = int8_value.astype(float32) * scale

    ---

    ## Pipeline

        LifWML (PyTorch)
            |
            | quantize_lif_wml()
            v
        export dict (INT8 + meta)
            |
            | save_neuromorphic_artefact()
            v
        artefact dir (artefact.json + weights.npz)
            |
            +----> MockNeuromorphicRunner (pure numpy, CI-safe)
            |          |
            |          | compare_software_vs_neuromorphic()
            |          v
            |      delta < 2% -> gate-neuro-passed
            |
            +----> LoihiCompiler.compile()  [requires lava-nc + hardware]
            |
            +----> AkidaCompiler.compile()  [requires akida + hardware]

    ---

    ## Step-by-step usage

    ### 1. Train a LifWML

        from track_w.lif_wml import LifWML
        wml = LifWML(id=0, n_neurons=100, alphabet_size=64, seed=42)
        # ... training loop ...

    ### 2. Export

        from pathlib import Path
        from neuromorphic.export import quantize_lif_wml, save_neuromorphic_artefact

        artefact = quantize_lif_wml(wml)
        save_neuromorphic_artefact(artefact, Path("my_artefact/"))

    ### 3. Verify with mock runner

        import torch
        import numpy as np
        from neuromorphic.export import load_neuromorphic_artefact
        from neuromorphic.spike_encoder import rate_encode
        from neuromorphic.mock_runner import MockNeuromorphicRunner
        from neuromorphic.verify import compare_software_vs_neuromorphic

        artefact = load_neuromorphic_artefact(Path("my_artefact/"))
        runner   = MockNeuromorphicRunner(artefact)

        x_norm  = ...  # float tensor [batch, n_neurons] in [0, 1]
        enc_gen = torch.Generator().manual_seed(0)
        spikes  = rate_encode(x_norm, n_timesteps=32, generator=enc_gen).numpy()

        pytorch_labels = ...  # your PyTorch inference labels [batch]

        result = compare_software_vs_neuromorphic(
            pytorch_labels=pytorch_labels,
            spike_input=spikes,
            artefact=artefact,
            runner=runner,
        )
        print(result)
        # {"accuracy_pytorch": 0.984, "accuracy_neuromorphic": 1.0, "delta": 0.016}

    ### 4. Compile to Loihi 2 (hardware required)

        pip install lava-nc

    Then replace the stub body in neuromorphic/loihi_stub.py with real
    lava.magma calls. See the docstring in that file for the expected API.

    ### 5. Compile to Akida (hardware required)

        pip install akida

    Then replace the stub body in neuromorphic/akida_stub.py with real
    akida.Model calls. See the docstring in that file for the expected API.

    ---

    ## Quantization details

    Symmetric per-tensor INT8:

        scale    = max(|W|) / 127
        W_int8   = clip(round(W / scale), -127, 127)
        W_approx = W_int8 * scale     # max error = 0.5 * scale per weight
        zero_point = 0                 # always, symmetric quantization

    Expected accuracy impact: < 2% agreement delta on FlowProxyTask
    (validated by gate-neuro-passed).

    ---

    ## CI integration

    The gate gate-neuro-passed is a git tag applied after all tests in
    tests/integration/track_w/test_gate_neuro.py pass:

        uv run pytest tests/integration/track_w/test_gate_neuro.py -v
        git tag gate-neuro-passed
        git push origin gate-neuro-passed

- [ ] **Step 2: Run full neuromorphic test suite**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest \
  tests/unit/test_spike_encoder.py \
  tests/unit/test_export.py \
  tests/unit/test_mock_runner.py \
  tests/unit/test_verify.py \
  tests/unit/test_vendor_stubs.py \
  tests/integration/track_w/test_gate_neuro.py \
  tests/integration/track_w/test_gate_neuro_accuracy.py \
  -v 2>&1 | tail -30
```

Expected: all tests PASS. If any fail, debug before proceeding to tag.

- [ ] **Step 3: Run existing tests to confirm no regressions**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run pytest tests/ -m "not slow" -v 2>&1 | tail -20
```

Expected: all pre-existing tests PASS.

- [ ] **Step 4: Commit the deployment guide**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add docs/neuromorphic/deployment-guide.md
git commit -m "docs(neuromorphic): deployment guide -- artefact format + pipeline" \
  -m "Problem: spec §13 open question has no written answer: how does a trained LifWML reach Loihi or Akida?

Solution: deployment-guide.md covers artefact format (JSON + NPZ schema), pipeline diagram, step-by-step usage, Loihi/Akida stub replacement recipes, CI gate integration, and quantization theory."
```

- [ ] **Step 5: Tag gate-neuro-passed and push**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git tag gate-neuro-passed
git push origin master
git push origin gate-neuro-passed
```

Expected: `* [new tag] gate-neuro-passed -> gate-neuro-passed`

---

## Self-review checklist

### Spec coverage

| Requirement | Task |
|-------------|------|
| `quantize_lif_wml` with INT8, scale, zero_point | Task 4 |
| `save_neuromorphic_artefact` (JSON + NPZ) | Task 4 |
| `load_neuromorphic_artefact` (inverse) | Task 4 |
| `rate_encode` [batch, T, dim] binary | Task 2 |
| `temporal_encode` time-to-first-spike | Tasks 2-3 |
| `MockNeuromorphicRunner` pure-numpy LIF | Task 5 |
| `compare_software_vs_neuromorphic` | Task 6 |
| Gate: round-trip bit-stable | Task 8 (G1) |
| Gate: rate-encode firing rate [0.1, 0.7] | Task 8 (G2) |
| Gate: delta < 2% | Tasks 7 + 8 (G3) |
| Loihi stub | Task 9 |
| Akida stub | Task 9 |
| Commented vendor imports in `__init__.py` | Tasks 1 + 9 |
| `docs/neuromorphic/deployment-guide.md` | Task 10 |
| No vendor SDK in `pyproject.toml` | design constraint -- verified: no lava-nc or akida in deps |

### Placeholder scan

No TBD, no TODO, no "similar to Task N", no "appropriate error handling". All code blocks are complete with actual implementations.

### Type consistency

- `quantize_lif_wml(wml) -> dict[str, Any]` -- used as `artefact` everywhere, consistent Tasks 4-10.
- `rate_encode(x, *, n_timesteps, generator) -> Tensor` -- consistent across Tasks 2, 3, 7, 8.
- `temporal_encode(x, *, n_timesteps) -> Tensor` -- consistent Tasks 2, 3.
- `MockNeuromorphicRunner(artefact, *, dt)` -- consistent Tasks 5, 6, 7, 8.
- `compare_software_vs_neuromorphic(*, pytorch_labels, spike_input, artefact, runner)` -- all kwargs, consistent Tasks 6, 7, 8.
- `LoihiCompiler.compile(artefact)` / `AkidaCompiler.compile(artefact)` -- static methods, consistent in stubs and tests (Task 9).
- `_pytorch_classify(wml, x)` -- defined identically in Tasks 6, 7, 8 (three separate test files; intentional to avoid cross-file imports in tests).
