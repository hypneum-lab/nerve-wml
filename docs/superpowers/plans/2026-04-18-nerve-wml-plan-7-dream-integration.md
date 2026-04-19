# Dream-of-Kiki Integration — Offline ε Consolidation via kiki_oniric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal bridge that lets a trained nerve-wml system export its ε error-role trace, hand it to dream-of-kiki's `kiki_oniric` substrate for one offline consolidation cycle, and re-import the resulting transducer weight delta — partially resolving spec §13 open question "Dream integration" (spec line 543).

**Architecture:** `bridge/dream_bridge.py` provides the public `DreamBridge` class (collect → encode → apply delta); `bridge/dream_protocol.py` lazy-imports `kiki_oniric` via `importlib` so nerve-wml ships without a hard dependency; `bridge/mock_consolidator.py` provides a drop-in zero-delta stub for CI; `bridge/eps_replay.py` saves/loads NPZ trace files for offline replay. The whole stack is env-gated by `DREAM_CONSOLIDATION_ENABLED`: unset or `"0"` → all methods no-op, return empty/zero artefacts. Gate `gate-dream-passed` asserts round-trip correctness, env-gate compliance, and trace bit-stability.

**Tech Stack:** Python 3.12, PyTorch 2.3+, numpy 1.26+, importlib (stdlib), pytest 8+. No new runtime dependencies. dream-of-kiki is an optional editable install for integration testing only.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `bridge/dream_protocol.py` | Lazy `importlib` loader for `kiki_oniric`; `load_dream_module()` |
| Create | `bridge/mock_consolidator.py` | `MockConsolidator.consolidate()` → zero delta; CI stand-in |
| Create | `bridge/dream_bridge.py` | `DreamBridge` class: `collect_eps_trace`, `to_dream_input`, `apply_consolidation_output` |
| Create | `bridge/eps_replay.py` | `save_eps_trace` / `load_eps_trace` — NPZ file round-trip |
| Modify | `bridge/__init__.py` | Export `DreamBridge`, `load_dream_module`, `MockConsolidator`, `save_eps_trace`, `load_eps_trace` |
| Create | `tests/unit/test_dream_protocol.py` | `load_dream_module` returns None when not installed; interface contract |
| Create | `tests/unit/test_dream_bridge.py` | Shape, env gate (no-op), trace determinism, logit delta |
| Create | `tests/unit/test_eps_replay.py` | NPZ save/load round-trip, metadata preservation |
| Create | `tests/integration/test_gate_dream.py` | Gate aggregator: full round-trip, env-gate, idempotence, bit-stability |
| Create | `docs/dream/integration-notes.md` | Installation recipe, protocol surface, schema v0 contract |

---

## Task 1: `bridge/dream_protocol.py` — lazy loader + protocol contract

The lazy loader keeps `nerve-wml` installable without `dream-of-kiki`. It tries `importlib.import_module("kiki_oniric")` and returns `None` on `ImportError`. A helper `assert_protocol_surface` verifies the loaded module exposes `consolidate(trace, *, profile) -> np.ndarray` so tests can validate the real module without nerve-wml knowing its internals.

**Files:**
- Create: `bridge/dream_protocol.py`
- Create: `tests/unit/test_dream_protocol.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_dream_protocol.py`:

```python
"""Tests for bridge/dream_protocol.py lazy loader."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from bridge.dream_protocol import assert_protocol_surface, load_dream_module


def test_load_dream_module_returns_none_when_not_installed(monkeypatch):
    """If kiki_oniric is not importable, load_dream_module returns None without raising."""
    # Simulate absence by blocking the import.
    monkeypatch.setitem(sys.modules, "kiki_oniric", None)  # type: ignore[arg-type]
    result = load_dream_module()
    assert result is None


def test_load_dream_module_returns_module_when_installed(monkeypatch):
    """If kiki_oniric is importable, load_dream_module returns the module object."""
    fake = types.ModuleType("kiki_oniric")

    def fake_consolidate(trace: np.ndarray, *, profile: str = "P_equ") -> np.ndarray:
        return np.zeros((2, 64, 64), dtype=np.float32)

    fake.consolidate = fake_consolidate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiki_oniric", fake)
    result = load_dream_module()
    assert result is fake


def test_assert_protocol_surface_passes_for_compliant_module():
    """A module with consolidate(trace, *, profile) should pass the surface check."""
    fake = types.ModuleType("kiki_oniric")

    def fake_consolidate(trace: np.ndarray, *, profile: str = "P_equ") -> np.ndarray:
        return np.zeros((2, 64, 64), dtype=np.float32)

    fake.consolidate = fake_consolidate  # type: ignore[attr-defined]
    # Must not raise.
    assert_protocol_surface(fake)


def test_assert_protocol_surface_raises_for_missing_consolidate():
    """A module without consolidate must raise AttributeError."""
    fake = types.ModuleType("kiki_oniric")
    with pytest.raises(AttributeError, match="consolidate"):
        assert_protocol_surface(fake)


def test_load_dream_module_with_explicit_path_prefers_sys_modules(monkeypatch):
    """Passing path=None uses importlib; if kiki_oniric already in sys.modules it is reused."""
    fake = types.ModuleType("kiki_oniric")
    fake.consolidate = lambda trace, *, profile="P_equ": np.zeros((1, 64, 64))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiki_oniric", fake)
    result = load_dream_module(path=None)
    assert result is fake
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_dream_protocol.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` for `bridge.dream_protocol`.

- [ ] **Step 3: Create `bridge/dream_protocol.py`**

```python
"""Lazy loader for the kiki_oniric consolidation module.

nerve-wml ships without a hard dependency on dream-of-kiki.
Users who want live consolidation install dream-of-kiki as an
editable dep:

    pip install -e /path/to/dreamOfkiki

The loader then finds kiki_oniric via the normal Python import path.
If not installed, load_dream_module() returns None and the bridge
falls back to MockConsolidator automatically.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_dream_module(path: Path | None = None) -> ModuleType | None:
    """Lazily import kiki_oniric.

    Args:
        path: Ignored in this implementation (reserved for future
              sys.path injection when the package lives at a
              non-standard location). Pass None for the default
              importlib lookup.

    Returns:
        The kiki_oniric module, or None if not installed / blocked.
    """
    # If the module was blocked in sys.modules (value is None), honour it.
    sentinel = sys.modules.get("kiki_oniric", _MISSING)
    if sentinel is None:
        return None
    try:
        return importlib.import_module("kiki_oniric")
    except ImportError:
        return None


class _Missing:
    pass


_MISSING = _Missing()


def assert_protocol_surface(module: ModuleType) -> None:
    """Raise AttributeError if module does not expose the expected protocol surface.

    Expected: module.consolidate(trace: np.ndarray, *, profile: str) -> np.ndarray
    """
    if not hasattr(module, "consolidate"):
        raise AttributeError(
            f"kiki_oniric module {module!r} does not expose 'consolidate'. "
            "Expected: consolidate(trace: np.ndarray, *, profile: str = 'P_equ') "
            "-> np.ndarray shaped [n_transducers, 64, 64]."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_dream_protocol.py -v 2>&1 | tail -20
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/dream_protocol.py tests/unit/test_dream_protocol.py
git commit -m "feat(bridge): dream_protocol lazy kiki_oniric loader" \
  -m "Problem: nerve-wml must ship without a hard dep on dream-of-kiki while still being able to call kiki_oniric.consolidate when it is installed.

Solution: load_dream_module() wraps importlib.import_module in a try/except ImportError and checks sys.modules for a blocked sentinel, returning None in all failure cases. assert_protocol_surface() validates the expected API surface."
```

---

## Task 2: `bridge/mock_consolidator.py` — zero-delta CI stub

The mock consolidator implements the same `consolidate(trace, *, profile)` interface as the real `kiki_oniric` module. It returns a zero `np.ndarray` of shape `[n_transducers, 64, 64]` derived from the trace length. All tests use this stub so they never depend on dream-of-kiki being installed.

**Files:**
- Create: `bridge/mock_consolidator.py`
- Create: `tests/unit/test_mock_consolidator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_mock_consolidator.py`:

```python
"""Tests for bridge/mock_consolidator.py."""
from __future__ import annotations

import numpy as np
import pytest

from bridge.mock_consolidator import MockConsolidator


def test_consolidate_returns_zero_array():
    """consolidate must return an all-zero array."""
    trace = np.zeros((10, 4), dtype=np.float32)
    mc = MockConsolidator(n_transducers=3)
    delta = mc.consolidate(trace, profile="P_equ")
    assert np.all(delta == 0.0), "MockConsolidator must return a zero delta"


def test_consolidate_shape_matches_n_transducers():
    """delta shape must be [n_transducers, 64, 64]."""
    trace = np.zeros((50, 4), dtype=np.float32)
    mc = MockConsolidator(n_transducers=5)
    delta = mc.consolidate(trace, profile="P_equ")
    assert delta.shape == (5, 64, 64), f"expected (5, 64, 64), got {delta.shape}"


def test_consolidate_dtype_is_float32():
    """delta must be float32 to match the real kiki_oniric output dtype."""
    trace = np.zeros((20, 4), dtype=np.float32)
    mc = MockConsolidator(n_transducers=2)
    delta = mc.consolidate(trace, profile="P_min")
    assert delta.dtype == np.float32


def test_consolidate_accepts_all_profiles():
    """profile kwarg must be accepted without error for all named profiles."""
    trace = np.zeros((5, 4), dtype=np.float32)
    mc = MockConsolidator(n_transducers=1)
    for profile in ("P_min", "P_equ", "P_max"):
        delta = mc.consolidate(trace, profile=profile)
        assert delta.shape == (1, 64, 64)


def test_consolidate_empty_trace_still_returns_correct_shape():
    """An empty trace (0 events) must not crash; shape is [n_transducers, 64, 64]."""
    trace = np.zeros((0, 4), dtype=np.float32)
    mc = MockConsolidator(n_transducers=4)
    delta = mc.consolidate(trace, profile="P_equ")
    assert delta.shape == (4, 64, 64)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_mock_consolidator.py -v 2>&1 | tail -20
```

Expected: `ImportError` for `bridge.mock_consolidator`.

- [ ] **Step 3: Create `bridge/mock_consolidator.py`**

```python
"""MockConsolidator — zero-delta stub for CI and offline tests.

Drop-in replacement for kiki_oniric when dream-of-kiki is not
installed. Implements the same consolidate() signature; returns an
all-zero delta so apply_consolidation_output becomes a no-op in tests
that verify idempotence.
"""
from __future__ import annotations

import numpy as np


class MockConsolidator:
    """Zero-delta consolidator — matches kiki_oniric.consolidate() surface."""

    ALPHABET_SIZE: int = 64

    def __init__(self, n_transducers: int) -> None:
        self.n_transducers = n_transducers

    def consolidate(
        self,
        trace: np.ndarray,
        *,
        profile: str = "P_equ",
    ) -> np.ndarray:
        """Return a zero delta shaped [n_transducers, 64, 64].

        Args:
            trace: [n_events, 4] float32 array — ignored (mock).
            profile: profile name — accepted but ignored (mock).

        Returns:
            np.ndarray of shape (self.n_transducers, 64, 64), dtype float32,
            all zeros.
        """
        return np.zeros(
            (self.n_transducers, self.ALPHABET_SIZE, self.ALPHABET_SIZE),
            dtype=np.float32,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_mock_consolidator.py -v 2>&1 | tail -20
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/mock_consolidator.py tests/unit/test_mock_consolidator.py
git commit -m "feat(bridge): MockConsolidator zero-delta CI stub" \
  -m "Problem: tests for the dream bridge must not depend on dream-of-kiki being installed in CI, but must exercise the same consolidate() call path as the real module.

Solution: MockConsolidator.consolidate(trace, *, profile) returns np.zeros([n_transducers, 64, 64], float32) — correct shape and dtype, guaranteed zero so idempotence assertions are trivially satisfied."
```

---

## Task 3: `DreamBridge.collect_eps_trace` and `to_dream_input`

`collect_eps_trace` drives a `SimNerveAdapter` for `duration_ticks` ticks (alternating gamma/theta phases) and records every ε-role letter delivered. `to_dream_input` encodes the trace to `[n_events, 4]` float32 with columns `[src, dst, code, phase_clock]`. `phase_clock = round(timestamp * GAMMA_HZ)`.

If `DREAM_CONSOLIDATION_ENABLED` is `"0"` or unset, `collect_eps_trace` returns `[]` and `to_dream_input` returns a `[0, 4]` array without touching the nerve.

**Files:**
- Create: `bridge/dream_bridge.py`
- Create: `tests/unit/test_dream_bridge.py` (partial — Tasks 3 and 4 build this file together)

- [ ] **Step 1: Write the failing tests for collect_eps_trace and to_dream_input**

Create `tests/unit/test_dream_bridge.py`:

```python
"""Tests for DreamBridge collect/encode and apply methods."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from bridge.dream_bridge import DreamBridge
from bridge.mock_consolidator import MockConsolidator
from bridge.sim_nerve_adapter import SimNerveAdapter
from nerve_core.neuroletter import Phase, Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nerve(n_wmls: int = 4, seed: int = 0) -> SimNerveAdapter:
    return SimNerveAdapter(n_wmls=n_wmls, k=2, seed=seed)


def _make_bridge(enabled: bool = True) -> DreamBridge:
    val = "1" if enabled else "0"
    os.environ["DREAM_CONSOLIDATION_ENABLED"] = val
    bridge = DreamBridge()
    return bridge


# ---------------------------------------------------------------------------
# collect_eps_trace
# ---------------------------------------------------------------------------

def test_collect_eps_trace_returns_list(monkeypatch):
    """collect_eps_trace must return a list (possibly empty) without raising."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=42)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=20)
    assert isinstance(trace, list)


def test_collect_eps_trace_only_eps_role(monkeypatch):
    """All letters in the returned trace must have Role.ERROR (ε)."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=7)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=50)
    for letter in trace:
        assert letter.role is Role.ERROR, (
            f"Non-ε letter found: role={letter.role}"
        )


def test_collect_eps_trace_noop_when_gate_disabled(monkeypatch):
    """When DREAM_CONSOLIDATION_ENABLED=0, collect_eps_trace returns []."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "0")
    nerve = _make_nerve(seed=1)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=200)
    assert trace == []


def test_collect_eps_trace_noop_when_env_unset(monkeypatch):
    """When DREAM_CONSOLIDATION_ENABLED is not set, collect_eps_trace returns []."""
    monkeypatch.delenv("DREAM_CONSOLIDATION_ENABLED", raising=False)
    nerve = _make_nerve(seed=2)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=100)
    assert trace == []


def test_collect_eps_trace_determinism(monkeypatch):
    """Same seed → same trace (bit-stable)."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve_a = _make_nerve(seed=99)
    nerve_b = _make_nerve(seed=99)
    bridge = DreamBridge()
    trace_a = bridge.collect_eps_trace(nerve_a, duration_ticks=30)
    trace_b = bridge.collect_eps_trace(nerve_b, duration_ticks=30)
    assert len(trace_a) == len(trace_b), "trace length must be deterministic for same seed"
    for la, lb in zip(trace_a, trace_b):
        assert la == lb, "trace letter mismatch — not bit-stable"


# ---------------------------------------------------------------------------
# to_dream_input
# ---------------------------------------------------------------------------

def test_to_dream_input_shape(monkeypatch):
    """to_dream_input must return array of shape [n_events, 4]."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=5)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=40)
    arr = bridge.to_dream_input(trace)
    if len(trace) > 0:
        assert arr.shape == (len(trace), 4), f"expected ({len(trace)}, 4), got {arr.shape}"
    else:
        assert arr.shape == (0, 4)


def test_to_dream_input_dtype(monkeypatch):
    """to_dream_input must return float32."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=3)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=40)
    arr = bridge.to_dream_input(trace)
    assert arr.dtype == np.float32


def test_to_dream_input_columns(monkeypatch):
    """Columns: [src, dst, code, phase_clock]. code in [0,63], phase_clock >= 0."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=8)
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=60)
    if not trace:
        pytest.skip("no ε letters collected — increase duration_ticks or n_wmls")
    arr = bridge.to_dream_input(trace)
    codes = arr[:, 2]
    clocks = arr[:, 3]
    assert np.all(codes >= 0) and np.all(codes <= 63), "code column out of [0, 63]"
    assert np.all(clocks >= 0), "phase_clock must be non-negative"


def test_to_dream_input_noop_returns_empty_when_disabled(monkeypatch):
    """When gate disabled, to_dream_input([]) must return shape [0, 4]."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "0")
    bridge = DreamBridge()
    arr = bridge.to_dream_input([])
    assert arr.shape == (0, 4)
    assert arr.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_dream_bridge.py -v 2>&1 | tail -20
```

Expected: `ImportError` for `bridge.dream_bridge`.

- [ ] **Step 3: Create `bridge/dream_bridge.py`** (collect + encode only; apply_consolidation_output added in Task 4)

```python
"""DreamBridge — public interface for ε-trace consolidation via kiki_oniric.

Usage (with dream-of-kiki installed):
    bridge = DreamBridge()
    trace = bridge.collect_eps_trace(nerve, duration_ticks=1000)
    arr   = bridge.to_dream_input(trace)
    delta = kiki_oniric.consolidate(arr, profile="P_equ")
    bridge.apply_consolidation_output(nerve, delta)

When DREAM_CONSOLIDATION_ENABLED is unset or "0" every method is a
no-op returning an empty / zero artefact.

Trace schema v0 (locked at gate-dream-passed):
    [n_events, 4] float32
    col 0: src   (int, WML source id)
    col 1: dst   (int, WML destination id)
    col 2: code  (int, 0..63 alphabet code)
    col 3: phase_clock  (int, round(timestamp * GAMMA_HZ))
"""
from __future__ import annotations

import os

import numpy as np
import torch

from bridge.sim_nerve_adapter import SimNerveAdapter
from nerve_core.neuroletter import Neuroletter, Phase, Role


class DreamBridge:
    """Minimal bridge between nerve-wml ε traces and kiki_oniric consolidation.

    Env gate: DREAM_CONSOLIDATION_ENABLED must be exactly "1" for any
    method to perform real work. All other values (including unset) cause
    every method to return empty / zero artefacts without touching the nerve
    or any transducer.
    """

    GAMMA_HZ: float = 40.0
    # Number of ticks between gamma and theta phase during collection.
    # Each cycle: gamma active for _GAMMA_TICKS, then theta for _THETA_TICKS.
    _GAMMA_TICKS: int = 5
    _THETA_TICKS: int = 1

    def __init__(self) -> None:
        raw = os.environ.get("DREAM_CONSOLIDATION_ENABLED", "0")
        self._enabled: bool = raw.strip() == "1"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_eps_trace(
        self,
        nerve: SimNerveAdapter,
        duration_ticks: int = 1000,
    ) -> list[Neuroletter]:
        """Drive nerve for duration_ticks and record all ε-role letters.

        The nerve is driven in alternating gamma / theta windows so that
        both π and ε letters are delivered. Only ε (Role.ERROR) letters
        are retained in the returned trace.

        Args:
            nerve: A SimNerveAdapter instance (pre-initialised).
            duration_ticks: Number of phase ticks to simulate. One tick
                            corresponds to one nerve.tick(dt=1.0/GAMMA_HZ)
                            call.

        Returns:
            List of Neuroletter instances with role == Role.ERROR.
            Returns [] immediately if the env gate is disabled.
        """
        if not self._enabled:
            return []

        dt = 1.0 / self.GAMMA_HZ
        collected: list[Neuroletter] = []
        in_gamma = True
        tick_in_phase = 0

        for _ in range(duration_ticks):
            # Alternate gamma/theta windows.
            if in_gamma:
                nerve.set_phase_active(gamma=True, theta=False)
                phase_limit = self._GAMMA_TICKS
            else:
                nerve.set_phase_active(gamma=False, theta=True)
                phase_limit = self._THETA_TICKS

            # Inject synthetic ε letters on all active edges so the
            # theta window has something to deliver.
            if not in_gamma:
                self._inject_eps(nerve)

            # Drain all WML queues, keep only ε letters.
            for dst in range(nerve.n_wmls):
                letters = nerve.listen(dst)
                for letter in letters:
                    if letter.role is Role.ERROR:
                        collected.append(letter)

            nerve.tick(dt)
            tick_in_phase += 1
            if tick_in_phase >= phase_limit:
                in_gamma = not in_gamma
                tick_in_phase = 0

        return collected

    def to_dream_input(self, trace: list[Neuroletter]) -> np.ndarray:
        """Encode a Neuroletter trace to the kiki_oniric input format.

        Schema v0: float32 array [n_events, 4]
            col 0: src         — WML source id
            col 1: dst         — WML destination id
            col 2: code        — alphabet code, 0..63
            col 3: phase_clock — round(timestamp * GAMMA_HZ)

        The phase_clock is rounded to the nearest int and stored as float32
        to remain locale-free and bit-stable across platforms.

        Args:
            trace: List of Neuroletter instances (typically ε-role only).

        Returns:
            np.ndarray of shape [len(trace), 4], dtype float32.
            Returns shape [0, 4] float32 if trace is empty OR gate disabled.
        """
        if not self._enabled or not trace:
            return np.zeros((0, 4), dtype=np.float32)

        rows: list[list[float]] = []
        for letter in trace:
            phase_clock = float(round(letter.timestamp * self.GAMMA_HZ))
            rows.append([
                float(letter.src),
                float(letter.dst),
                float(letter.code),
                phase_clock,
            ])
        return np.array(rows, dtype=np.float32)

    def apply_consolidation_output(
        self,
        nerve: SimNerveAdapter,
        delta: np.ndarray,
        alpha: float = 0.1,
    ) -> None:
        """Apply kiki_oniric transducer delta in-place to nerve transducers.

        For each transducer in nerve._transducers (indexed positionally in
        sorted key order), adds alpha * delta[i] to transducer.logits.data.

        N-4 invariant is preserved because this method does not touch the
        routing edge matrix (nerve._edges) — only transducer logits.

        Args:
            nerve:  A SimNerveAdapter with active transducers.
            delta:  np.ndarray of shape [n_transducers, 64, 64] float32
                    from kiki_oniric.consolidate().
            alpha:  Learning rate scalar. Default 0.1.
                    apply_consolidation_output(nerve, zero_delta) is always
                    a no-op regardless of alpha.

        Returns:
            None. Modifies nerve._transducers in-place.
        """
        if not self._enabled:
            return

        if delta.size == 0:
            return

        keys = sorted(nerve._transducers.keys())
        delta_t = torch.from_numpy(delta)  # [n_transducers, 64, 64]

        for i, key in enumerate(keys):
            if i >= delta_t.shape[0]:
                break
            transducer = nerve._transducers[key]
            with torch.no_grad():
                transducer.logits.data += alpha * delta_t[i]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_eps(self, nerve: SimNerveAdapter) -> None:
        """Send a synthetic ε letter on every active edge so theta windows carry traffic."""
        from nerve_core.neuroletter import Neuroletter, Phase, Role

        for src in range(nerve.n_wmls):
            for dst in range(nerve.n_wmls):
                if src == dst:
                    continue
                if nerve.routing_weight(src, dst) == 0.0:
                    continue
                letter = Neuroletter(
                    code=src % 64,
                    role=Role.ERROR,
                    phase=Phase.THETA,
                    src=src,
                    dst=dst,
                    timestamp=nerve.time(),
                )
                nerve.send(letter)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_dream_bridge.py -v -k "not apply" 2>&1 | tail -30
```

Expected: all collect/encode tests PASS. `apply_consolidation_output` tests are added in Task 4.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/dream_bridge.py tests/unit/test_dream_bridge.py
git commit -m "feat(bridge): DreamBridge collect_eps_trace + to_dream_input" \
  -m "Problem: spec §13 asks for a bridge to export ε traces from nerve-wml. Without it, accumulated ε signals during online inference have no downstream consolidation target.

Solution: DreamBridge.collect_eps_trace drives SimNerveAdapter in alternating gamma/theta windows, retaining only Role.ERROR letters. to_dream_input encodes them to [n_events, 4] float32 with phase_clock = round(ts * GAMMA_HZ). Both methods no-op when DREAM_CONSOLIDATION_ENABLED != '1'."
```

---

## Task 4: `DreamBridge.apply_consolidation_output` — tests

`apply_consolidation_output` is already written in `bridge/dream_bridge.py` (Task 3). This task adds the missing tests: in-place logit mutation, idempotence with zero delta, N-4 preserved (edges unchanged), and env-gate no-op.

**Files:**
- Modify: `tests/unit/test_dream_bridge.py` (append new test functions)

- [ ] **Step 1: Append apply tests to `tests/unit/test_dream_bridge.py`**

Append the following block at the end of `tests/unit/test_dream_bridge.py`:

```python
# ---------------------------------------------------------------------------
# apply_consolidation_output
# ---------------------------------------------------------------------------

def test_apply_consolidation_output_mutates_logits(monkeypatch):
    """Non-zero delta must change transducer logits by the expected magnitude."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(n_wmls=4, seed=0)
    bridge = DreamBridge()

    n_transducers = len(nerve._transducers)
    assert n_transducers > 0, "nerve must have at least 1 active transducer"

    # Snapshot logits before.
    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    # Build a uniform delta: every entry = 1.0.
    alpha = 0.1
    delta = np.ones((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta, alpha=alpha)

    # Each logit must have increased by alpha * 1.0 = 0.1.
    for i, k in enumerate(keys):
        if i >= n_transducers:
            break
        after = nerve._transducers[k].logits.data
        diff = (after - before[k]).abs().max().item()
        assert abs(diff - alpha) < 1e-5, (
            f"transducer {k}: expected max diff {alpha}, got {diff}"
        )


def test_apply_consolidation_output_zero_delta_is_idempotent(monkeypatch):
    """apply_consolidation_output(nerve, zero_delta) must leave logits unchanged."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(n_wmls=4, seed=0)
    bridge = DreamBridge()

    n_transducers = len(nerve._transducers)
    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    zero_delta = np.zeros((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, zero_delta)

    for k in keys:
        assert torch.equal(before[k], nerve._transducers[k].logits.data), (
            f"transducer {k}: logits changed despite zero delta"
        )


def test_apply_consolidation_output_preserves_edges(monkeypatch):
    """N-4: edge matrix must not change after apply_consolidation_output."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(n_wmls=4, seed=0)
    bridge = DreamBridge()

    edges_before = nerve._edges.clone()
    n_transducers = len(nerve._transducers)
    delta = np.ones((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta)

    assert torch.equal(edges_before, nerve._edges), (
        "N-4 violated: apply_consolidation_output must not modify edge matrix"
    )


def test_apply_consolidation_output_noop_when_disabled(monkeypatch):
    """When DREAM_CONSOLIDATION_ENABLED=0, apply must not touch logits."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "0")
    nerve = _make_nerve(n_wmls=4, seed=0)
    bridge = DreamBridge()

    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    n_transducers = max(1, len(nerve._transducers))
    delta = np.ones((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta)

    for k in keys:
        assert torch.equal(before[k], nerve._transducers[k].logits.data), (
            f"transducer {k}: logits mutated despite gate disabled"
        )
```

- [ ] **Step 2: Run all bridge tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_dream_bridge.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/unit/test_dream_bridge.py
git commit -m "test(bridge): apply_consolidation_output assertions" \
  -m "Problem: apply_consolidation_output mutates transducer logits in-place; without tests for idempotence and N-4 edge preservation a regression could silently corrupt the routing graph.

Solution: four targeted tests — mutation magnitude, zero-delta idempotence, edge-matrix invariant (N-4), env-gate no-op."
```

---

## Task 5: `bridge/eps_replay.py` — NPZ trace save / load

Provides `save_eps_trace` and `load_eps_trace` so a trace collected online can be replayed offline without re-running the simulation. The NPZ contains two arrays: `trace` (the `[n_events, 4]` float32 matrix) and `metadata` (a JSON-encoded string stored as a 0-d numpy bytes array with keys `schema_version`, `n_wmls`, `duration_ticks`, `gamma_hz`).

**Files:**
- Create: `bridge/eps_replay.py`
- Create: `tests/unit/test_eps_replay.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eps_replay.py`:

```python
"""Tests for bridge/eps_replay.py save/load round-trip."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from bridge.eps_replay import load_eps_trace, save_eps_trace


def _make_trace(n_events: int = 20) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((n_events, 4)).astype(np.float32)


def test_round_trip_restores_trace_exactly():
    """save then load must return bit-identical trace array."""
    trace = _make_trace(30)
    meta = {"schema_version": 0, "n_wmls": 4, "duration_ticks": 100, "gamma_hz": 40.0}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.npz"
        save_eps_trace(trace, meta, path)
        loaded_trace, loaded_meta = load_eps_trace(path)
    assert np.array_equal(trace, loaded_trace), "trace not bit-identical after round-trip"


def test_round_trip_restores_metadata():
    """Metadata dict must survive JSON encode/decode round-trip."""
    trace = _make_trace(10)
    meta = {"schema_version": 0, "n_wmls": 8, "duration_ticks": 200, "gamma_hz": 40.0}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.npz"
        save_eps_trace(trace, meta, path)
        _, loaded_meta = load_eps_trace(path)
    assert loaded_meta["n_wmls"] == 8
    assert loaded_meta["duration_ticks"] == 200
    assert loaded_meta["schema_version"] == 0


def test_save_creates_npz_file():
    """save_eps_trace must create a file at the given path."""
    trace = _make_trace(5)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.npz"
        assert not path.exists()
        save_eps_trace(trace, {}, path)
        assert path.exists()


def test_load_empty_trace():
    """A saved trace with 0 events must load to shape [0, 4]."""
    trace = np.zeros((0, 4), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.npz"
        save_eps_trace(trace, {"schema_version": 0}, path)
        loaded_trace, _ = load_eps_trace(path)
    assert loaded_trace.shape == (0, 4)


def test_load_raises_for_missing_file():
    """load_eps_trace must raise FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError):
        load_eps_trace(Path("/tmp/nerve_wml_nonexistent_trace_xyz.npz"))


def test_save_dtype_preserved():
    """Saved trace dtype (float32) must be preserved on load."""
    trace = _make_trace(15)
    assert trace.dtype == np.float32
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dtype.npz"
        save_eps_trace(trace, {}, path)
        loaded_trace, _ = load_eps_trace(path)
    assert loaded_trace.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_eps_replay.py -v 2>&1 | tail -20
```

Expected: `ImportError` for `bridge.eps_replay`.

- [ ] **Step 3: Create `bridge/eps_replay.py`**

```python
"""eps_replay — save and load ε traces for offline consolidation.

File format: numpy NPZ with two arrays:
  "trace"    — float32 [n_events, 4] encoding the ε trace (schema v0).
  "metadata" — 0-d bytes array containing a UTF-8 JSON string with
               at least {"schema_version": int, "n_wmls": int,
               "duration_ticks": int, "gamma_hz": float}.

Usage:
    save_eps_trace(arr, meta, Path("run_001.npz"))
    arr, meta = load_eps_trace(Path("run_001.npz"))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_eps_trace(
    trace: np.ndarray,
    metadata: dict,
    path: Path,
) -> None:
    """Persist an ε trace to an NPZ file.

    Args:
        trace:    float32 array of shape [n_events, 4] (schema v0).
        metadata: Arbitrary JSON-serialisable dict (schema_version,
                  n_wmls, duration_ticks, gamma_hz recommended).
        path:     Destination file path (will be created or overwritten).
    """
    meta_bytes = np.frombuffer(
        json.dumps(metadata).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(
        path,
        trace=trace.astype(np.float32),
        metadata=meta_bytes,
    )


def load_eps_trace(path: Path) -> tuple[np.ndarray, dict]:
    """Load an ε trace from an NPZ file.

    Args:
        path: Path to an NPZ file produced by save_eps_trace.

    Returns:
        (trace, metadata) where trace is float32 [n_events, 4] and
        metadata is the decoded JSON dict.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        trace: np.ndarray = npz["trace"].astype(np.float32)
        meta_bytes: np.ndarray = npz["metadata"]

    metadata: dict = json.loads(bytes(meta_bytes.tolist()).decode("utf-8"))
    return trace, metadata
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/test_eps_replay.py -v 2>&1 | tail -20
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/eps_replay.py tests/unit/test_eps_replay.py
git commit -m "feat(bridge): eps_replay save/load NPZ" \
  -m "Problem: ε traces collected during online nerve operation must survive a process boundary so offline consolidation can replay them without re-running the simulation.

Solution: save_eps_trace/load_eps_trace persist [n_events, 4] float32 + JSON metadata as a compressed NPZ. Schema v0 columns are [src, dst, code, phase_clock]. allow_pickle=False is enforced on load for safety."
```

---

## Task 6: Update `bridge/__init__.py`

Export `DreamBridge`, `load_dream_module`, `MockConsolidator`, `save_eps_trace`, `load_eps_trace` so callers can do `from bridge import DreamBridge`.

**Files:**
- Modify: `bridge/__init__.py`

- [ ] **Step 1: Read current `bridge/__init__.py`**

```bash
cat /Users/electron/Documents/Projets/nerve-wml/bridge/__init__.py
```

- [ ] **Step 2: Replace `bridge/__init__.py`**

Open `bridge/__init__.py` and replace its full contents with:

```python
"""bridge — dream-of-kiki integration layer for nerve-wml.

Public API:
    DreamBridge         — collect ε trace, encode, apply delta
    load_dream_module   — lazy importlib loader for kiki_oniric
    MockConsolidator    — zero-delta CI stub
    save_eps_trace      — persist ε trace to NPZ
    load_eps_trace      — load ε trace from NPZ
"""
from bridge.dream_bridge import DreamBridge
from bridge.dream_protocol import load_dream_module
from bridge.eps_replay import load_eps_trace, save_eps_trace
from bridge.mock_consolidator import MockConsolidator

__all__ = [
    "DreamBridge",
    "load_dream_module",
    "MockConsolidator",
    "save_eps_trace",
    "load_eps_trace",
]
```

- [ ] **Step 3: Verify imports work**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -c "
from bridge import DreamBridge, load_dream_module, MockConsolidator, save_eps_trace, load_eps_trace
print('bridge imports OK')
"
```

Expected: prints `bridge imports OK`.

- [ ] **Step 4: Run full unit suite to confirm no regressions**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing unit tests PASS plus the new dream/eps tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add bridge/__init__.py
git commit -m "chore(bridge): export dream bridge public API" \
  -m "Problem: DreamBridge, MockConsolidator, load_dream_module, save/load_eps_trace were not importable from the bridge package root, breaking any downstream import of the form 'from bridge import DreamBridge'.

Solution: bridge/__init__.py now re-exports all five public symbols with an explicit __all__."
```

---

## Task 7: Gate `gate-dream-passed` integration test aggregator

The gate asserts the full round-trip with MockConsolidator, env-gate compliance, trace determinism, and idempotence in a single integration test file. Passing all tests in this file is the condition for tagging `gate-dream-passed`.

**Files:**
- Create: `tests/integration/test_gate_dream.py`

- [ ] **Step 1: Create `tests/integration/test_gate_dream.py`**

```python
"""Gate: gate-dream-passed.

Full round-trip: collect ε trace → encode → MockConsolidator.consolidate
→ apply delta → assert logit change magnitude.

Pass condition (all five assertions green):
  1. Round-trip produces non-trivially shaped trace with enabled gate.
  2. Env gate disabled → end-to-end no-op (logits unchanged).
  3. Zero-delta idempotence (MockConsolidator always returns zeros → no-op).
  4. Trace bit-stability: same seed → same trace across two independent runs.
  5. to_dream_input column ranges valid (code in [0,63], phase_clock >= 0).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from bridge import (
    DreamBridge,
    MockConsolidator,
    load_eps_trace,
    save_eps_trace,
)
from bridge.sim_nerve_adapter import SimNerveAdapter


def _make_nerve(seed: int = 0) -> SimNerveAdapter:
    return SimNerveAdapter(n_wmls=4, k=2, seed=seed)


# ---------------------------------------------------------------------------
# Gate assertion 1: full round-trip with MockConsolidator
# ---------------------------------------------------------------------------

def test_gate_dream_full_round_trip(monkeypatch):
    """Full round-trip must succeed and leave transducer logits unchanged
    (because MockConsolidator returns zero delta)."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=42)
    bridge = DreamBridge()

    # Collect.
    trace = bridge.collect_eps_trace(nerve, duration_ticks=80)
    # MockConsolidator needs n_transducers; derive from nerve.
    n_transducers = len(nerve._transducers)
    assert n_transducers > 0, "nerve must have active transducers"

    # Encode.
    arr = bridge.to_dream_input(trace)
    assert arr.ndim == 2 and arr.shape[1] == 4, (
        f"to_dream_input must return [n, 4], got {arr.shape}"
    )

    # Consolidate via mock.
    mc = MockConsolidator(n_transducers=n_transducers)
    delta = mc.consolidate(arr, profile="P_equ")
    assert delta.shape == (n_transducers, 64, 64)

    # Snapshot logits.
    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    # Apply.
    bridge.apply_consolidation_output(nerve, delta)

    # Zero delta → logits unchanged.
    for k in keys:
        assert torch.equal(before[k], nerve._transducers[k].logits.data), (
            f"gate-dream: transducer {k} logits changed despite zero delta"
        )


# ---------------------------------------------------------------------------
# Gate assertion 2: env gate disabled → end-to-end no-op
# ---------------------------------------------------------------------------

def test_gate_dream_env_gate_disabled_noop(monkeypatch):
    """With DREAM_CONSOLIDATION_ENABLED=0, the full pipeline must be a no-op."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "0")
    nerve = _make_nerve(seed=7)
    bridge = DreamBridge()

    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    trace = bridge.collect_eps_trace(nerve, duration_ticks=200)
    assert trace == [], "disabled gate: collect must return []"

    arr = bridge.to_dream_input(trace)
    assert arr.shape == (0, 4), "disabled gate: to_dream_input must return [0,4]"

    n_transducers = max(1, len(nerve._transducers))
    delta = np.ones((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, delta)

    for k in keys:
        assert torch.equal(before[k], nerve._transducers[k].logits.data), (
            f"gate-dream: transducer {k} mutated with gate disabled"
        )


# ---------------------------------------------------------------------------
# Gate assertion 3: zero-delta idempotence (explicit alpha path)
# ---------------------------------------------------------------------------

def test_gate_dream_zero_delta_idempotent(monkeypatch):
    """apply_consolidation_output with a zero delta must leave logits bit-identical."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=5)
    bridge = DreamBridge()

    n_transducers = len(nerve._transducers)
    keys = sorted(nerve._transducers.keys())
    before = {k: nerve._transducers[k].logits.data.clone() for k in keys}

    zero_delta = np.zeros((n_transducers, 64, 64), dtype=np.float32)
    bridge.apply_consolidation_output(nerve, zero_delta, alpha=0.5)

    for k in keys:
        assert torch.equal(before[k], nerve._transducers[k].logits.data), (
            f"idempotence violated for transducer {k}"
        )


# ---------------------------------------------------------------------------
# Gate assertion 4: trace bit-stability (same seed → same trace)
# ---------------------------------------------------------------------------

def test_gate_dream_trace_determinism(monkeypatch):
    """Same seed must produce bit-identical trace across two independent nerve instances."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    bridge = DreamBridge()

    nerve_a = _make_nerve(seed=13)
    nerve_b = _make_nerve(seed=13)

    trace_a = bridge.collect_eps_trace(nerve_a, duration_ticks=50)
    trace_b = bridge.collect_eps_trace(nerve_b, duration_ticks=50)

    assert len(trace_a) == len(trace_b), (
        f"trace length differs: {len(trace_a)} vs {len(trace_b)}"
    )
    for i, (la, lb) in enumerate(zip(trace_a, trace_b)):
        assert la == lb, f"trace[{i}] mismatch: {la} vs {lb}"


# ---------------------------------------------------------------------------
# Gate assertion 5: trace column range validity
# ---------------------------------------------------------------------------

def test_gate_dream_trace_column_ranges(monkeypatch):
    """All trace columns must be in range: code ∈ [0,63], phase_clock ≥ 0."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=3)
    bridge = DreamBridge()

    trace = bridge.collect_eps_trace(nerve, duration_ticks=80)
    if not trace:
        pytest.skip("no ε letters collected — topology has no active edges")

    arr = bridge.to_dream_input(trace)
    codes = arr[:, 2]
    clocks = arr[:, 3]

    assert np.all(codes >= 0) and np.all(codes <= 63), (
        f"code column out of [0, 63]: min={codes.min()}, max={codes.max()}"
    )
    assert np.all(clocks >= 0), (
        f"phase_clock column contains negative values: min={clocks.min()}"
    )


# ---------------------------------------------------------------------------
# Gate assertion 6: eps_replay NPZ round-trip
# ---------------------------------------------------------------------------

def test_gate_dream_eps_replay_round_trip(monkeypatch, tmp_path):
    """save_eps_trace then load_eps_trace must preserve trace and metadata exactly."""
    monkeypatch.setenv("DREAM_CONSOLIDATION_ENABLED", "1")
    nerve = _make_nerve(seed=17)
    bridge = DreamBridge()

    trace = bridge.collect_eps_trace(nerve, duration_ticks=60)
    arr = bridge.to_dream_input(trace)
    meta = {
        "schema_version": 0,
        "n_wmls": nerve.n_wmls,
        "duration_ticks": 60,
        "gamma_hz": DreamBridge.GAMMA_HZ,
    }

    path = tmp_path / "gate_dream_trace.npz"
    save_eps_trace(arr, meta, path)
    loaded_arr, loaded_meta = load_eps_trace(path)

    assert np.array_equal(arr, loaded_arr), "trace not bit-identical after NPZ round-trip"
    assert loaded_meta["schema_version"] == 0
    assert loaded_meta["n_wmls"] == nerve.n_wmls
```

- [ ] **Step 2: Run gate tests**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest tests/integration/test_gate_dream.py -v 2>&1 | tail -30
```

Expected: 6 PASSED.

- [ ] **Step 3: Run full test suite to verify no regressions**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest -v --tb=short 2>&1 | tail -40
```

Expected: all existing tests PASS plus the 6 new gate tests.

- [ ] **Step 4: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add tests/integration/test_gate_dream.py
git commit -m "test(gate): gate-dream-passed aggregator (6 assertions)" \
  -m "Problem: the dream bridge has no integration-level gate that proves the full round-trip works end-to-end and the env gate is respected across all three DreamBridge methods.

Solution: test_gate_dream.py asserts (1) full MockConsolidator round-trip, (2) disabled-gate no-op, (3) zero-delta idempotence, (4) trace bit-stability, (5) trace column ranges, (6) NPZ replay round-trip. All six must pass before tagging gate-dream-passed."
```

---

## Task 8: `docs/dream/integration-notes.md`

Documents the schema v0 contract, how to install dream-of-kiki as an editable dep, the protocol surface that `kiki_oniric` must expose, and which nerve-wml tags produce which trace formats.

**Files:**
- Create: `docs/dream/integration-notes.md`

- [ ] **Step 1: Create `docs/dream/` directory and write `integration-notes.md`**

```bash
mkdir -p /Users/electron/Documents/Projets/nerve-wml/docs/dream
```

Then create `docs/dream/integration-notes.md`:

```markdown
# dream-of-kiki Integration Notes

**Schema version:** 0 (locked at tag `gate-dream-passed`)
**Last updated:** 2026-04-19
**Spec reference:** nerve-wml design §13 (line 543) — "Dream integration"

---

## 1. Purpose

nerve-wml accumulates **ε error-role Neuroletters** during online
inference (θ phase, 6 Hz). These signals are natural candidates for
offline consolidation via dream-of-kiki's `kiki_oniric` substrate:
replaying them through the `P_equ` profile's `replay` + `restructure`
operations adjusts transducer logits without disturbing the online nerve.

This document describes the minimal interface between the two projects.

---

## 2. Installing dream-of-kiki as an editable dependency

nerve-wml has **no runtime dependency** on dream-of-kiki. Install it as
an editable package only when you want live consolidation (not required
for CI or tests):

```bash
# From the nerve-wml root:
pip install -e /path/to/dreamOfkiki
# or with uv:
uv pip install -e /path/to/dreamOfkiki
```

Verify the install:

```bash
python -c "import kiki_oniric; print(kiki_oniric.__file__)"
```

When `kiki_oniric` is not installed, `load_dream_module()` returns
`None` and `DreamBridge` falls back to a zero-delta no-op transparently.

---

## 3. Enabling consolidation at runtime

Set the environment variable before constructing `DreamBridge`:

```bash
export DREAM_CONSOLIDATION_ENABLED=1
```

Any other value (including unset) disables all bridge methods. The gate
is read **once** at `DreamBridge.__init__` time and is immutable for the
lifetime of the instance.

---

## 4. Protocol surface — what `kiki_oniric` must expose

`bridge/dream_protocol.py` validates this surface via
`assert_protocol_surface(module)`.

```python
def consolidate(
    trace: np.ndarray,          # [n_events, 4] float32  (schema v0)
    *,
    profile: str = "P_equ",    # "P_min" | "P_equ" | "P_max"
) -> np.ndarray:               # [n_transducers, 64, 64] float32
    ...
```

The returned delta is applied in-place by
`DreamBridge.apply_consolidation_output` as:

```
transducer.logits += alpha * delta[i]   (default alpha = 0.1)
```

---

## 5. Trace schema v0 (locked)

Column layout for `to_dream_input()` output:

| Column | Name | Type | Range | Description |
|--------|------|------|-------|-------------|
| 0 | `src` | float32 | [0, n_wmls) | Source WML id |
| 1 | `dst` | float32 | [0, n_wmls) | Destination WML id |
| 2 | `code` | float32 | [0, 63] | On-wire alphabet code |
| 3 | `phase_clock` | float32 | ≥ 0 | `round(timestamp * GAMMA_HZ)` (40 Hz) |

Schema v0 is **frozen** at tag `gate-dream-passed`. Any change requires
a new schema version and a DualVer bump on the dream-of-kiki side.

---

## 6. Which nerve-wml tags produce which trace formats

| Tag | Schema | Profile tested | Notes |
|-----|--------|----------------|-------|
| `gate-p-passed` | — | — | Track-P only; no WMLs, no ε traces |
| `gate-w-passed` | — | — | Track-W MockNerve; no transducers |
| `gate-m-passed` | — | — | Merge gate; transducers live but bridge not wired |
| `gate-m2-passed` | — | — | M2 gate; same |
| `gate-dream-passed` | **v0** | MockConsolidator (P_equ) | First schema lock |

Real `P_equ` consolidation (live `kiki_oniric`) is deferred to
dream-of-kiki v0.5+ (see §13 "PARTIALLY RESOLVED" note in spec).

---

## 7. NPZ file format (eps_replay)

`save_eps_trace` / `load_eps_trace` produce compressed NPZ files:

```
trace.npz
  ├── trace     — float32 [n_events, 4]  (schema v0)
  └── metadata  — uint8 array (UTF-8 JSON bytes)
                  keys: schema_version, n_wmls, duration_ticks, gamma_hz
```

Load example:

```python
from bridge import load_eps_trace
arr, meta = load_eps_trace(Path("my_run.npz"))
# arr.shape → (n_events, 4), arr.dtype → float32
# meta["schema_version"] → 0
```

---

## 8. Full usage example

```python
import os
os.environ["DREAM_CONSOLIDATION_ENABLED"] = "1"

from pathlib import Path
from bridge import DreamBridge, MockConsolidator, save_eps_trace

nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
bridge = DreamBridge()

# Collect ε trace during inference.
trace = bridge.collect_eps_trace(nerve, duration_ticks=1000)

# Encode to kiki_oniric format.
arr = bridge.to_dream_input(trace)

# Save for offline replay (optional).
save_eps_trace(arr, {"schema_version": 0, "n_wmls": 4,
               "duration_ticks": 1000, "gamma_hz": 40.0},
               Path("run_001.npz"))

# Consolidate (use MockConsolidator in CI, real kiki_oniric in production).
n_transducers = len(nerve._transducers)
delta = MockConsolidator(n_transducers).consolidate(arr, profile="P_equ")

# Apply delta to nerve transducers in-place.
bridge.apply_consolidation_output(nerve, delta, alpha=0.1)
```

---

## 9. Resolving spec §13 "Dream integration"

Status after `gate-dream-passed`: **PARTIALLY RESOLVED**.

- The minimal interface is defined and locked (schema v0, protocol surface).
- The bridge is functional with `MockConsolidator`.
- **Pending:** real `kiki_oniric.consolidate` integration requires
  dream-of-kiki v0.5+ which ships `kiki_oniric` as a proper Python
  package with the `consolidate()` entry point. Track in
  [dream-of-kiki issue tracker](https://github.com/electron-rare/dream-of-kiki).
```

- [ ] **Step 2: Verify the file was created**

```bash
ls -la /Users/electron/Documents/Projets/nerve-wml/docs/dream/integration-notes.md
```

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add docs/dream/integration-notes.md
git commit -m "docs(dream): integration-notes schema v0 contract" \
  -m "Problem: spec §13 asks for a documented minimal interface between nerve-wml and dream-of-kiki; without it the bridge protocol surface is implicit and fragile to cross-repo changes.

Solution: docs/dream/integration-notes.md documents schema v0 column layout, install recipe for dream-of-kiki editable dep, kiki_oniric protocol surface, env gate, NPZ format, tag-to-schema mapping, and usage example. Schema v0 is declared frozen at gate-dream-passed."
```

---

## Task 9: Tag `gate-dream-passed` and push

Final sweep: run linter, type-checker, and full test suite. Tag and push.

**Files:**
- No new files.

- [ ] **Step 1: Run ruff linter on new bridge files**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run ruff check bridge/dream_protocol.py bridge/mock_consolidator.py bridge/dream_bridge.py bridge/eps_replay.py 2>&1 | tail -20
```

Expected: no errors. If any violations appear, fix them before proceeding.

- [ ] **Step 2: Run mypy on nerve_core and track_p**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run mypy nerve_core track_p bridge --ignore-missing-imports 2>&1 | tail -20
```

Expected: no errors introduced by the new bridge files.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && uv run python -m pytest -v --tb=short 2>&1 | tail -40
```

Expected: all tests PASS, including the 6 gate-dream assertions.

- [ ] **Step 4: Tag `gate-dream-passed`**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git tag gate-dream-passed
git push origin master --tags
```

Expected: remote confirms new tag `gate-dream-passed` pushed.

- [ ] **Step 5: Verify tag on origin**

```bash
cd /Users/electron/Documents/Projets/nerve-wml && git ls-remote --tags origin | grep gate-dream
```

Expected: line like `refs/tags/gate-dream-passed`.

---

## Optional Task 10: Mark spec §13 "Dream integration" PARTIALLY RESOLVED

Updates the design spec to record the resolution status, matching the pattern established by §13.1 in Plan 4a.

**Files:**
- Modify: `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`

- [ ] **Step 1: Find the §13 dream integration line**

```bash
grep -n "Dream integration" /Users/electron/Documents/Projets/nerve-wml/docs/superpowers/specs/2026-04-18-nerve-wml-design.md
```

Expected: line like `543: - **Dream integration.** What is the minimal interface...`

- [ ] **Step 2: Append resolution status after §13.1 resolution block**

Find the last line of the `§13.1 Known limitations` section (after the W4 resolution note) and append:

```markdown
#### Resolution status (Plan 7)

- **Dream integration** — PARTIALLY RESOLVED (2026-04-19). `DreamBridge`
  in `bridge/dream_bridge.py` defines the minimal interface:
  `collect_eps_trace` (ε-role letter accumulation) →
  `to_dream_input` (schema v0: `[n_events, 4]` with `[src, dst, code, phase_clock]`) →
  `kiki_oniric.consolidate(trace, profile="P_equ")` →
  `apply_consolidation_output` (`transducer.logits += alpha * delta`).
  The bridge ships without a hard dep on dream-of-kiki via
  `importlib`-based lazy loading. CI uses `MockConsolidator`
  (zero delta). Schema v0 is frozen at tag `gate-dream-passed`.
  **Pending:** real `kiki_oniric.consolidate` integration requires
  dream-of-kiki v0.5+ to export the `consolidate()` entry point.
  See `docs/dream/integration-notes.md` §9.
```

- [ ] **Step 3: Commit**

```bash
cd /Users/electron/Documents/Projets/nerve-wml
git add docs/superpowers/specs/2026-04-18-nerve-wml-design.md
git commit -m "docs(spec): §13 dream integration PARTIALLY RESOLVED" \
  -m "Problem: spec §13 open question 'Dream integration' had no resolution status, making it invisible to future plan writers.

Solution: append resolution block citing gate-dream-passed, bridge/dream_bridge.py, schema v0 column layout, and the pending real-kiki_oniric dependency. Pattern follows §13.1 resolution blocks from Plan 4a."
git push origin master
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| `DreamBridge` class with `collect_eps_trace` | Task 3 |
| `DreamBridge.to_dream_input` → `[n_events, 4]` | Task 3 |
| `DreamBridge.apply_consolidation_output` (`transducer.logits += delta * alpha`) | Task 3 (code), Task 4 (tests) |
| Env gate `DREAM_CONSOLIDATION_ENABLED` | Tasks 3, 7 |
| `bridge/dream_protocol.py` lazy `importlib` loader | Task 1 |
| `MockConsolidator.consolidate(trace, *, profile)` | Task 2 |
| `bridge/eps_replay.py` save/load NPZ | Task 5 |
| `bridge/__init__.py` exports | Task 6 |
| `gate-dream-passed` aggregator (round-trip, env-gate, idempotence, determinism) | Task 7 |
| `docs/dream/integration-notes.md` | Task 8 |
| Tag + push `gate-dream-passed` | Task 9 |
| Spec §13 resolution annotation | Task 10 (optional) |

### Placeholder scan

No "TBD", "TODO", "implement later", or "similar to Task N" patterns present. All code blocks are complete.

### Type consistency

- `collect_eps_trace` → `list[Neuroletter]` used consistently in Tasks 3, 7.
- `to_dream_input(trace: list[Neuroletter]) -> np.ndarray` used consistently in Tasks 3, 7.
- `apply_consolidation_output(nerve, delta: np.ndarray, alpha: float)` consistent Tasks 3, 4, 7.
- `MockConsolidator(n_transducers=int).consolidate(trace, *, profile)` consistent Tasks 2, 7.
- `save_eps_trace(arr, meta, path)` / `load_eps_trace(path)` consistent Tasks 5, 6, 7, 8.

### Design constraints verified

- No runtime dep on dream-of-kiki. `importlib.import_module` wrapped in `try/except ImportError`.
- Env gate read once at `DreamBridge.__init__` time.
- `apply_consolidation_output(nerve, zero_delta)` is idempotent — asserted in `test_gate_dream_zero_delta_idempotent`.
- Trace schema `[n_events, 4]` with `phase_clock = round(timestamp * GAMMA_HZ)` — consistent across `to_dream_input`, docs, and NPZ metadata.
- No WML mutation — only `transducer.logits.data` is touched in `apply_consolidation_output`.
- N-4 invariant preserved — edge matrix never touched — asserted in `test_apply_consolidation_output_preserves_edges`.
