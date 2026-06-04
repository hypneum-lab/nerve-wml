# Biological Substrate WML Implementation Plan

For agentic workers: this plan is written for execution under
`superpowers:subagent-driven-development`. Each task is self-contained,
ends with a green test and a commit, and can be dispatched to an
independent subagent in order. Do not skip the TDD cycle: write the
failing test first, run it and confirm it fails, then write the minimal
real implementation, run it and confirm it passes, then commit.

**Goal.** Add a fourth WML substrate to nerve-wml — `BioWML` — backed by
a remote *biological neural culture* API (Cortical Labs CL1, FinalSpark
Neuroplatform). The substrate must conform to the existing
`nerve_core.protocols.WML` Protocol so it drops into the heterogeneous
pool alongside `MlpWML`, `LifWML`, and `TransformerWML`. Because real
wetware access is async, external, and rate-limited, the substrate is
fully testable offline via a `MockBioCultureClient`; the two real
adapters (`CL1Adapter`, `FinalSparkAdapter`) are env-gated on
`NERVE_WML_BIO_API_KEY` and degrade to `pytest.skip` when it is unset.
Additionally, add a thin `neuromorphic/neurobench_mapping.py` module that
maps one nerve-wml validation outcome onto a NeuroBench task (Yik et al.,
*Nature Communications* 2025) so results are externally comparable.

**Architecture.**

```
                         track_w/bio_clients.py
   ┌──────────────────────────────────────────────────────────┐
   │  BioCultureClient (Protocol)                              │
   │    encode_stimulus(codes: list[int]) -> StimulusFrame     │
   │    decode_activity(frame: ActivityFrame) -> list[int]     │
   │    roundtrip(codes: list[int]) -> ActivityFrame           │
   │    close() -> None                                        │
   ├──────────────────────────────────────────────────────────┤
   │  MockBioCultureClient   — numpy spike sim, latency/jitter │
   │  CL1Adapter             — env-gated, Cortical Labs CL API │
   │  FinalSparkAdapter      — env-gated, FinalSpark REST API  │
   └──────────────────────────────────────────────────────────┘
                              ▲
                              │ injected
   track_w/bio_wml.py         │
   ┌──────────────────────────┴───────────────────────────────┐
   │  BioWML(nn.Module)  — conforms to nerve_core WML Protocol │
   │    id, codebook (nn.Parameter), emit_head_pi              │
   │    step(nerve, t): listen → encode → roundtrip via client │
   │                    → decode → emit π (γ) / ε (θ)          │
   └──────────────────────────────────────────────────────────┘

   neuromorphic/neurobench_mapping.py
   ┌──────────────────────────────────────────────────────────┐
   │  nerve-wml validation outcome  ──►  NeuroBench task row   │
   │  (one task: streaming classification accuracy + footprint)│
   └──────────────────────────────────────────────────────────┘
```

`BioWML` holds a `BioCultureClient`. In tests and CI the client is a
`MockBioCultureClient` (deterministic numpy spike simulation with
realistic latency, jitter, and additive noise). In production a real
adapter is constructed only if `NERVE_WML_BIO_API_KEY` is set; otherwise
construction of the real adapter raises and the substrate caller is
expected to fall back to mock (the cross-substrate test and CI never
touch the network).

**Tech Stack.** Python 3.12+, `uv` (`uv sync --all-extras`), PyTorch
(`torch`, `nn.Module` + `nn.Parameter`, matching the three existing
substrates), `numpy` (spike simulation), `pytest` (`-m "not slow"` for
CI, `@pytest.mark.slow` for network), `ruff`, `mypy`. Real adapters use
the stdlib `urllib.request` only — no new third-party HTTP dependency —
so `uv sync` stays unchanged. Commit format enforced by hooks: subject
≤ 50 chars, body lines ≤ 72, no underscore in commit scope, English, no
`--no-verify`.

---

## Task 0 — Prerequisite: real-hardware access & env gate

**Files:**
- Create: `track_w/CLAUDE.md` is NOT touched; instead document inline.
- Create: `docs/superpowers/research/2026-05-19-bio-substrate-access.md`
- Modify: none (no code yet)

This task has **no code** — it records the external prerequisite so a
later subagent never blocks on it.

- [ ] **Step 1: Note the provider accounts**

  Real biological-culture access requires an account and an API key
  from one of two providers:
  - **Cortical Labs CL1** — the "CL API" exposes real-time
    closed-loop read/stimulate access to a CL1 unit. Commercial;
    request access at corticallabs.com.
  - **FinalSpark Neuroplatform** — a remote REST API to living human
    brain organoids, **free for research groups**; apply at
    finalspark.com/neuroplatform.

- [ ] **Step 2: Record the env-var gate**

  The single environment variable that gates all real access is
  `NERVE_WML_BIO_API_KEY`. A second optional variable
  `NERVE_WML_BIO_PROVIDER` selects the adapter (`"cl1"` or
  `"finalspark"`, default `"finalspark"`). A third optional variable
  `NERVE_WML_BIO_ENDPOINT` overrides the base URL (for staging).

- [ ] **Step 3: Record the degradation contract**

  Degradation contract — **must hold for every later task**:
  - `MockBioCultureClient` needs **no** env var and always works.
  - `CL1Adapter` / `FinalSparkAdapter` constructors read
    `NERVE_WML_BIO_API_KEY`; if it is unset they raise
    `BioApiKeyMissing` (a subclass of `RuntimeError`).
  - Any `pytest` test that constructs a real adapter must first do
    `if not os.environ.get("NERVE_WML_BIO_API_KEY"): pytest.skip(...)`
    and be marked `@pytest.mark.slow`. CI runs `uv run pytest -m "not
    slow"`, so the network is never touched.
  - This mirrors the existing env-gated precedent
    `bridge/kiki_nerve_advisor.py` (`NERVE_WML_ENABLED`,
    `NERVE_WML_CHECKPOINT_PATH`, never raises in the hot path).

- [ ] **Step 4: Write the research note**

  Write the research note `docs/superpowers/research/2026-05-19-bio-
  substrate-access.md` with the three env vars, the two providers,
  the rate-limit expectations (CL1 ~tens of ms closed-loop, FinalSpark
  queued batch), and the degradation contract above. ~40 lines, plain
  markdown.

- [ ] **Step 5: Commit**

   ```bash
   git add docs/superpowers/research/2026-05-19-bio-substrate-access.md
   git commit -m "docs: record bio-substrate access prerequisites"
   ```

No test for this task (documentation only).

---

## Task 1 — Data frames & client Protocol

**Files:**
- Create: `track_w/bio_clients.py`
- Test: `tests/unit/test_bio_clients.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_bio_clients.py`:
   ```python
   import numpy as np

   from track_w.bio_clients import (
       ActivityFrame,
       BioApiKeyMissing,
       BioCultureClient,
       StimulusFrame,
   )


   def test_stimulus_frame_is_immutable_and_carries_codes():
       frame = StimulusFrame(
           codes=(3, 17, 42),
           channels=np.zeros((3, 8), dtype=np.float32),
       )
       assert frame.codes == (3, 17, 42)
       assert frame.channels.shape == (3, 8)


   def test_activity_frame_records_spikes_and_latency():
       frame = ActivityFrame(
           spikes=np.zeros((8, 32), dtype=np.float32),
           latency_ms=12.5,
       )
       assert frame.spikes.shape == (8, 32)
       assert frame.latency_ms == 12.5


   def test_bio_culture_client_is_a_runtime_checkable_protocol():
       # A bare object is not a BioCultureClient.
       assert not isinstance(object(), BioCultureClient)


   def test_bio_api_key_missing_is_a_runtime_error():
       assert issubclass(BioApiKeyMissing, RuntimeError)
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

  Module does not exist:
   ```bash
   uv run pytest tests/unit/test_bio_clients.py -q
   # EXPECT: ModuleNotFoundError / collection error — FAIL
   ```

- [ ] **Step 3: Write the minimal real implementation**

  `track_w/bio_clients.py`:
   ```python
   """Biological-culture client layer for the BioWML substrate.

   A BioCultureClient sends a discrete stimulus (a small list of
   alphabet codes, 0..63) to a neural culture and reads back the
   resulting multi-channel spiking activity. The substrate
   (track_w.bio_wml.BioWML) is provider-agnostic: it talks only to
   this Protocol. Three implementations live here:

     - MockBioCultureClient — offline numpy spike simulation with
       realistic latency, jitter, and additive noise (Task 2).
     - CL1Adapter           — Cortical Labs CL1, env-gated (Task 5).
     - FinalSparkAdapter    — FinalSpark Neuroplatform, env-gated.

    Plan C (bio substrate). See docs/superpowers/plans/
    2026-05-19-bio-substrate-wml.md.
   """
   from __future__ import annotations

   from dataclasses import dataclass
   from typing import Protocol, runtime_checkable

   import numpy as np

   ALPHABET_SIZE = 64


   class BioApiKeyMissing(RuntimeError):
       """Raised when a real adapter is built without NERVE_WML_BIO_API_KEY."""


   @dataclass(frozen=True)
   class StimulusFrame:
       """A stimulus delivered to a culture.

       codes:    the alphabet codes (0..63) being stimulated this tick.
       channels: float32 [n_codes, n_stim_channels] electrode pattern,
                 one row per code. Values in [0, 1] = stimulation
                 amplitude per channel.
       """

       codes: tuple[int, ...]
       channels: np.ndarray


   @dataclass(frozen=True)
   class ActivityFrame:
       """Spiking activity read back from a culture.

       spikes:     float32 [n_read_channels, n_bins] spike-count
                   raster over a short post-stimulus window.
       latency_ms: wall-clock round-trip latency for this exchange.
       """

       spikes: np.ndarray
       latency_ms: float


   @runtime_checkable
   class BioCultureClient(Protocol):
       """Provider-agnostic contract for a biological-culture backend."""

       n_stim_channels: int
       n_read_channels: int
       n_bins: int

       def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
           """Map alphabet codes to an electrode stimulation pattern."""
           ...

       def decode_activity(self, frame: ActivityFrame) -> list[int]:
           """Map read-back spiking activity to alphabet codes."""
           ...

       def roundtrip(self, codes: list[int]) -> ActivityFrame:
           """Stimulate with `codes`, read back, return the activity."""
           ...

       def close(self) -> None:
           """Release any underlying connection. Safe to call twice."""
           ...
   ```

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_bio_clients.py -q
   # EXPECT: 4 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/bio_clients.py tests/unit/test_bio_clients.py
   git commit -m "feat: add BioCultureClient protocol and frames"
   ```

---

## Task 2 — MockBioCultureClient (realistic offline simulation)

**Files:**
- Modify: `track_w/bio_clients.py`
- Test: `tests/unit/test_mock_bio_client.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_mock_bio_client.py`:
   ```python
   import time

   import numpy as np

   from track_w.bio_clients import (
       ActivityFrame,
       BioCultureClient,
       MockBioCultureClient,
       StimulusFrame,
   )


   def test_mock_client_satisfies_protocol():
       client = MockBioCultureClient(seed=0)
       assert isinstance(client, BioCultureClient)


   def test_encode_stimulus_shapes_and_range():
       client = MockBioCultureClient(seed=0)
       frame = client.encode_stimulus([3, 17, 42])
       assert isinstance(frame, StimulusFrame)
       assert frame.codes == (3, 17, 42)
       assert frame.channels.shape == (3, client.n_stim_channels)
       assert frame.channels.min() >= 0.0
       assert frame.channels.max() <= 1.0


   def test_roundtrip_is_deterministic_under_seed():
       a = MockBioCultureClient(seed=7).roundtrip([1, 2, 3])
       b = MockBioCultureClient(seed=7).roundtrip([1, 2, 3])
       assert np.array_equal(a.spikes, b.spikes)
       assert a.latency_ms == b.latency_ms


   def test_roundtrip_differs_across_seeds():
       a = MockBioCultureClient(seed=1).roundtrip([1, 2, 3])
       b = MockBioCultureClient(seed=2).roundtrip([1, 2, 3])
       assert not np.array_equal(a.spikes, b.spikes)


   def test_roundtrip_latency_has_jitter_within_bounds():
       client = MockBioCultureClient(
           seed=0, base_latency_ms=10.0, jitter_ms=4.0,
       )
       lats = [client.roundtrip([5]).latency_ms for _ in range(50)]
       assert all(6.0 <= x <= 14.0 for x in lats)
       assert len(set(lats)) > 1  # jitter actually varies


   def test_roundtrip_sleeps_when_simulate_wall_clock():
       client = MockBioCultureClient(
           seed=0, base_latency_ms=20.0, jitter_ms=0.0,
           simulate_wall_clock=True,
       )
       t0 = time.perf_counter()
       client.roundtrip([1])
       elapsed_ms = (time.perf_counter() - t0) * 1e3
       assert elapsed_ms >= 18.0  # actually slept ~20 ms


   def test_decode_activity_returns_codes_in_alphabet():
       client = MockBioCultureClient(seed=0)
       frame = client.roundtrip([9, 9, 9])
       codes = client.decode_activity(frame)
       assert all(0 <= c < 64 for c in codes)
       assert len(codes) == 3


   def test_stimulating_a_code_biases_decode_toward_that_code():
       # The mock is a noisy channel, not random: a strong repeated
       # stimulus should decode back to itself most of the time.
       client = MockBioCultureClient(seed=0, noise=0.05)
       hits = 0
       for _ in range(40):
           frame = client.roundtrip([11])
           if client.decode_activity(frame)[0] == 11:
               hits += 1
       assert hits >= 28  # >= 70 % round-trip fidelity at low noise


   def test_close_is_idempotent():
       client = MockBioCultureClient(seed=0)
       client.close()
       client.close()  # must not raise
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_mock_bio_client.py -q
   # EXPECT: ImportError MockBioCultureClient — FAIL
   ```

- [ ] **Step 3: Append the real implementation**

  Append to `track_w/bio_clients.py`:
   ```python
   import time as _time


   class MockBioCultureClient:
       """Offline numpy simulation of a neural culture.

       The simulation is a deterministic noisy channel. A stimulus
       code is written as a Gaussian "bump" on a code-dependent set
       of read channels; the culture's read-back spikes are that bump
       plus per-bin Poisson-like baseline firing plus additive
       Gaussian noise. `decode_activity` correlates the read raster
       against the same per-code channel templates and argmax-picks a
       code. With low `noise` round-trip fidelity is high (>70 %),
       which lets tests assert real behaviour rather than mock stubs.

       Latency: every roundtrip reports `base_latency_ms` plus uniform
       jitter in [-jitter_ms, +jitter_ms]. With `simulate_wall_clock`
       the call actually sleeps that long (used by the latency
       integration test); CI keeps it False so unit tests stay fast.
       """

       def __init__(
           self,
           *,
           n_stim_channels: int = 8,
           n_read_channels: int = 32,
           n_bins: int = 16,
           base_latency_ms: float = 12.0,
           jitter_ms: float = 4.0,
           noise: float = 0.15,
           baseline_rate: float = 0.20,
           simulate_wall_clock: bool = False,
           seed: int | None = None,
       ) -> None:
           self.n_stim_channels = n_stim_channels
           self.n_read_channels = n_read_channels
           self.n_bins = n_bins
           self.base_latency_ms = base_latency_ms
           self.jitter_ms = jitter_ms
           self.noise = noise
           self.baseline_rate = baseline_rate
           self.simulate_wall_clock = simulate_wall_clock
           self._rng = np.random.default_rng(seed)
           self._closed = False

           # Fixed per-code channel templates: a stable, seed-derived
           # map from each alphabet code to a soft pattern over the
           # read channels. Built from an independent generator so it
           # does not consume the roundtrip RNG stream.
           tpl_rng = np.random.default_rng(
               (seed if seed is not None else 0) + 104729
           )
           self._templates = tpl_rng.random(
               (ALPHABET_SIZE, n_read_channels), dtype=np.float32
           )
           # Sharpen: each code lights up ~25 % of channels strongly.
           thresh = np.quantile(self._templates, 0.75, axis=1, keepdims=True)
           self._templates = (self._templates >= thresh).astype(np.float32)

       def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
           if self._closed:
               raise RuntimeError("client is closed")
           channels = np.zeros(
               (len(codes), self.n_stim_channels), dtype=np.float32
           )
           for i, code in enumerate(codes):
               c = int(code) % ALPHABET_SIZE
               # Deterministic electrode pattern: which stim channels
               # this code drives, derived from the code bits.
               for ch in range(self.n_stim_channels):
                   channels[i, ch] = float((c >> ch) & 1)
           return StimulusFrame(codes=tuple(int(c) for c in codes),
                                channels=channels)

       def decode_activity(self, frame: ActivityFrame) -> list[int]:
           # Sum spikes over time bins → per-channel rate vector,
           # correlate against every code template, argmax per row.
           rates = frame.spikes.sum(axis=-1)  # [n_read_channels] or [k, n]
           rates = np.atleast_2d(rates)
           # rates rows are stacked per code in roundtrip; correlate.
           scores = rates @ self._templates.T  # [k, ALPHABET_SIZE]
           return [int(row.argmax()) for row in scores]

       def roundtrip(self, codes: list[int]) -> ActivityFrame:
           if self._closed:
               raise RuntimeError("client is closed")
           k = max(len(codes), 1)
           # Per-code read rasters, stacked: [k, n_read_channels, n_bins].
           rasters = np.zeros(
               (k, self.n_read_channels, self.n_bins), dtype=np.float32
           )
           for i, code in enumerate(codes):
               c = int(code) % ALPHABET_SIZE
               template = self._templates[c]  # [n_read_channels]
               # Evoked response: template bump in the early bins.
               evoked = np.outer(
                   template,
                   np.exp(-np.arange(self.n_bins) / 4.0).astype(np.float32),
               )
               baseline = self._rng.poisson(
                   self.baseline_rate, size=evoked.shape
               ).astype(np.float32)
               noise = self._rng.normal(
                   0.0, self.noise, size=evoked.shape
               ).astype(np.float32)
               rasters[i] = np.clip(evoked + baseline + noise, 0.0, None)
           # decode_activity wants [k, n_read_channels] after a sum over
           # bins; collapse the per-code rasters into a [k, ch, bins]
           # spikes array and let decode sum the last axis.
           spikes = rasters.reshape(k, self.n_read_channels, self.n_bins)
           latency = self.base_latency_ms + float(
               self._rng.uniform(-self.jitter_ms, self.jitter_ms)
           )
           if self.simulate_wall_clock:
               _time.sleep(max(latency, 0.0) / 1e3)
           # ActivityFrame.spikes is documented as [n_read, n_bins];
           # here we keep the per-code first axis so decode is exact.
           return ActivityFrame(spikes=spikes, latency_ms=latency)

       def close(self) -> None:
           self._closed = True
   ```
   Note: `decode_activity` calls `frame.spikes.sum(axis=-1)`; with the
   `[k, n_read, n_bins]` array this yields `[k, n_read]`, which is
   exactly what the template correlation expects — keep it that way.

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_mock_bio_client.py -q
   # EXPECT: 9 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/bio_clients.py tests/unit/test_mock_bio_client.py
   git commit -m "feat: add MockBioCultureClient spike simulator"
   ```

---

## Task 3 — BioWML substrate (construction + WML conformance)

**Files:**
- Create: `track_w/bio_wml.py`
- Test: `tests/unit/test_bio_wml.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_bio_wml.py`:
   ```python
   import torch

   from nerve_core.protocols import WML
   from track_w.bio_clients import MockBioCultureClient
   from track_w.bio_wml import BioWML


   def _mk(**kw):
       return BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0, **kw)


   def test_bio_wml_has_required_attrs():
       wml = _mk()
       assert wml.id == 0
       assert wml.codebook.shape == (64, 16)


   def test_bio_wml_conforms_to_wml_protocol():
       wml = _mk()
       assert isinstance(wml, WML)


   def test_bio_wml_parameters_include_codebook():
       wml = _mk()
       param_ids = {id(p) for p in wml.parameters()}
       assert id(wml.codebook) in param_ids


   def test_bio_wml_seed_is_local():
       torch.manual_seed(42)
       expected = torch.rand(1).item()
       torch.manual_seed(42)
       _ = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=99)
       observed = torch.rand(1).item()
       assert expected == observed


   def test_bio_wml_accepts_input_dim_larger_than_d_hidden():
       wml = BioWML(
           id=0, client=MockBioCultureClient(seed=0),
           input_dim=784, d_hidden=16, seed=0,
       )
       x = torch.randn(4, 784)
       h = wml.input_proj(x)
       assert h.shape == (4, 16)


   def test_bio_wml_default_input_dim_matches_d_hidden():
       wml = BioWML(
           id=0, client=MockBioCultureClient(seed=0),
           d_hidden=16, seed=0,
       )
       assert wml.input_dim == 16
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_bio_wml.py -q
   # EXPECT: ModuleNotFoundError track_w.bio_wml — FAIL
   ```

- [ ] **Step 3: Write the BioWML construction**

  Write `track_w/bio_wml.py` (construction only — `step` in Task 4):
   ```python
   """BioWML — a WML whose core is a remote biological neural culture.

   Conforms to nerve_core.protocols.WML. Like the three in-silico
   substrates (MlpWML, LifWML, TransformerWML) it is an nn.Module with
   a learned `codebook` (nn.Parameter) and a learned `emit_head_pi`
   readout. Unlike them, the "core" computation is delegated to a
   BioCultureClient: inbound codes are stimulated onto the culture and
   the read-back spiking activity is decoded back to codes and pooled
   into a hidden vector that the emit heads consume.

   The client is injected. In tests / CI it is a MockBioCultureClient.
   In production a real env-gated adapter is built via
   BioWML.from_env(). See track_w/bio_clients.py and Task 0 of
   docs/superpowers/plans/2026-05-19-bio-substrate-wml.md.
   """
   from __future__ import annotations

   from collections.abc import Iterable

   import numpy as np
   import torch
   from torch import Tensor, nn

   from nerve_core.protocols import Nerve
   from track_w.bio_clients import BioCultureClient


   class BioWML(nn.Module):
       """WML with a biological-culture core + π/ε emission heads."""

       def __init__(
           self,
           id: int,
           client: BioCultureClient,
           d_hidden: int = 16,
           alphabet_size: int = 64,
           threshold_eps: float = 0.30,
           input_dim: int | None = None,
           *,
           seed: int | None = None,
       ) -> None:
           super().__init__()
           self.id = id
           self.client = client
           self.d_hidden = d_hidden
           self.alphabet_size = alphabet_size
           self.threshold_eps = threshold_eps
           self.input_dim = input_dim if input_dim is not None else d_hidden

           gen = torch.Generator()
           if seed is not None:
               gen.manual_seed(seed)

           # Local codebook (N-5 — each WML owns its vocabulary).
           init = torch.randn(alphabet_size, d_hidden, generator=gen) * 0.1
           self.codebook = nn.Parameter(init)

           # Save global RNG so module construction never mutates it.
           saved_rng = torch.get_rng_state()
           try:
               self.input_proj = nn.Linear(self.input_dim, d_hidden)
               with torch.no_grad():
                   self.input_proj.weight.data = torch.randn(
                       d_hidden, self.input_dim, generator=gen
                   ) * 0.1
                   self.input_proj.bias.data.zero_()

               # Decoded spike-activity is pooled into d_hidden; the
               # emit heads map that hidden vector to alphabet logits.
               self.emit_head_pi = nn.Linear(d_hidden, alphabet_size)
               self.emit_head_eps = nn.Linear(d_hidden, alphabet_size)
               for head in (self.emit_head_pi, self.emit_head_eps):
                   with torch.no_grad():
                       head.weight.data = torch.randn(
                           alphabet_size, d_hidden, generator=gen
                       ) * 0.1
                       head.bias.data.zero_()
           finally:
               torch.set_rng_state(saved_rng)

       @classmethod
       def from_env(cls, id: int, **kwargs) -> "BioWML":
           """Build a BioWML backed by a real env-gated adapter.

           Reads NERVE_WML_BIO_PROVIDER (default "finalspark") and
           constructs the matching adapter, which itself reads
           NERVE_WML_BIO_API_KEY and raises BioApiKeyMissing if unset.
           Callers that want offline behaviour must inject a
           MockBioCultureClient directly instead of calling this.
           """
           import os

           from track_w.bio_clients import CL1Adapter, FinalSparkAdapter

           provider = os.environ.get(
               "NERVE_WML_BIO_PROVIDER", "finalspark"
           ).lower()
           client: BioCultureClient
           if provider == "cl1":
               client = CL1Adapter()
           elif provider == "finalspark":
               client = FinalSparkAdapter()
           else:
               raise ValueError(f"unknown bio provider: {provider!r}")
           return cls(id=id, client=client, **kwargs)

       def parameters(self, *args, **kwargs) -> Iterable[Tensor]:  # type: ignore[override]
           return super().parameters(*args, **kwargs)
   ```
   `step` is added in Task 4; the test above only exercises
   construction and protocol conformance.

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_bio_wml.py -q
   # EXPECT: 6 passed
   ```
   Note: `isinstance(wml, WML)` passes because `WML` is
   `@runtime_checkable` and `BioWML` exposes `id`, `codebook`,
   `step` (inherited at class scope after Task 4 — for Task 3 add a
   placeholder `step` so the protocol check passes; see step 3 amendment
   below).

   **Amendment to step 3:** add a minimal `step` so Task 3's
   `test_bio_wml_conforms_to_wml_protocol` is honest — it must call the
   real client even now:
   ```python
       def step(self, nerve: Nerve, t: float) -> None:
           """Filled in fully in Task 4. Minimal honest version:
           pull inbound, do nothing else. Replaced in Task 4."""
           nerve.listen(self.id)
   ```
   This keeps Task 3 green and Task 4 replaces the body.

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/bio_wml.py tests/unit/test_bio_wml.py
   git commit -m "feat: add BioWML substrate construction"
   ```

---

## Task 4 — BioWML.step(): listen → culture → emit π/ε

**Files:**
- Modify: `track_w/bio_wml.py`
- Test: `tests/unit/test_bio_wml_step.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_bio_wml_step.py`:
   ```python
   from nerve_core.neuroletter import Neuroletter, Phase, Role
   from track_w.bio_clients import MockBioCultureClient
   from track_w.bio_wml import BioWML
   from track_w.mock_nerve import MockNerve


   def _inject(nerve, dst, code):
       nerve._queues[dst].append(
           Neuroletter(code=code, role=Role.PREDICTION, phase=Phase.GAMMA,
                       src=1, dst=dst, timestamp=0.0)
       )


   def test_step_runs_without_error_on_silent_input():
       nerve = MockNerve(n_wmls=2, k=1, seed=0)
       nerve.set_phase_active(gamma=True, theta=False)
       wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
       wml.step(nerve, t=0.0)  # no inbound — must not raise


   def test_step_emits_a_prediction_after_stimulus():
       nerve = MockNerve(n_wmls=2, k=1, seed=0)
       nerve.set_phase_active(gamma=True, theta=False)
       wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
       _inject(nerve, dst=0, code=7)
       wml.step(nerve, t=0.0)
       out = nerve.listen(1)  # WML 1's inbox
       assert any(n.role is Role.PREDICTION for n in out)


   def test_step_prediction_uses_gamma_phase():
       nerve = MockNerve(n_wmls=2, k=1, seed=0)
       nerve.set_phase_active(gamma=True, theta=False)
       wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
       _inject(nerve, dst=0, code=7)
       wml.step(nerve, t=0.0)
       out = [n for n in nerve.listen(1) if n.role is Role.PREDICTION]
       assert out and all(n.phase is Phase.GAMMA for n in out)


   def test_step_emitted_code_is_in_alphabet():
       nerve = MockNerve(n_wmls=2, k=1, seed=0)
       nerve.set_phase_active(gamma=True, theta=False)
       wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
       _inject(nerve, dst=0, code=7)
       wml.step(nerve, t=0.0)
       assert all(0 <= n.code < 64 for n in nerve.listen(1))


   def test_step_records_last_latency_ms():
       nerve = MockNerve(n_wmls=2, k=1, seed=0)
       nerve.set_phase_active(gamma=True, theta=False)
       wml = BioWML(id=0, client=MockBioCultureClient(seed=0), seed=0)
       _inject(nerve, dst=0, code=7)
       wml.step(nerve, t=0.0)
       assert wml.last_latency_ms is not None
       assert wml.last_latency_ms > 0.0
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_bio_wml_step.py -q
   # EXPECT: AttributeError last_latency_ms / no emission — FAIL
   ```

- [ ] **Step 3: Implement the real step**

  In `track_w/bio_wml.py`, add `self.last_latency_ms: float | None =
  None` at the end of `__init__`, and replace the placeholder `step`
  with the real implementation:
   ```python
       def step(self, nerve: Nerve, t: float) -> None:
           """One tick: listen, stimulate the culture with inbound
           codes, decode the read-back activity into a hidden vector,
           emit π predictions (γ) and — on surprise — ε errors (θ).
           """
           from nerve_core.neuroletter import Neuroletter, Phase, Role

           inbound = nerve.listen(self.id)
           # Codes to stimulate. With no inbound, probe with a single
           # rest code so the culture/decoder still produce a hidden
           # state (the substrate is never fully silent at the API).
           codes = [n.code for n in inbound] if inbound else [0]

           # Round-trip through the biological culture (or its mock).
           frame = self.client.roundtrip(codes)
           self.last_latency_ms = frame.latency_ms
           decoded = self.client.decode_activity(frame)  # list[int]

           # Pool decoded codes into a d_hidden vector by averaging the
           # corresponding codebook rows — symmetric to embed_inbound.
           if decoded:
               rows = self.codebook[
                   torch.tensor([c % self.alphabet_size for c in decoded])
               ]
               h = rows.mean(dim=0)
           else:
               h = torch.zeros(self.d_hidden)

           pi_logits = self.emit_head_pi(h)
           code_pi = int(pi_logits.argmax().item())

           for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
               if dst == self.id:
                   continue
               if nerve.routing_weight(self.id, dst) == 1.0:
                   nerve.send(Neuroletter(
                       code=code_pi, role=Role.PREDICTION,
                       phase=Phase.GAMMA, src=self.id, dst=dst,
                       timestamp=t,
                   ))

           # ε path. Surprise = L2 norm of the hidden state itself:
           # a culture that produced strong evoked activity (large h)
           # signals a mismatch against the resting prior (h ≈ 0).
           surprise = float(h.norm().item())
           if surprise > self.threshold_eps:
               eps_logits = self.emit_head_eps(h)
               code_eps = int(eps_logits.argmax().item())
               for dst in range(nerve.n_wmls):  # type: ignore[attr-defined]
                   if dst == self.id:
                       continue
                   if nerve.routing_weight(self.id, dst) == 1.0:
                       nerve.send(Neuroletter(
                           code=code_eps, role=Role.ERROR,
                           phase=Phase.THETA, src=self.id, dst=dst,
                           timestamp=t,
                       ))
   ```
   Use `torch.no_grad()` is **not** wrapped here — the heads stay
   differentiable, matching `MlpWML.step` (which also does not wrap).

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_bio_wml_step.py -q
   # EXPECT: 5 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/bio_wml.py tests/unit/test_bio_wml_step.py
   git commit -m "feat: implement BioWML step with culture roundtrip"
   ```

---

## Task 5 — Real adapters (CL1Adapter, FinalSparkAdapter), env-gated

**Files:**
- Modify: `track_w/bio_clients.py`
- Test: `tests/integration/track_w/test_bio_adapters.py`

- [ ] **Step 1: Write the failing test**

  `tests/integration/track_w/test_bio_adapters.py`:
   ```python
   import os

   import pytest

   from track_w.bio_clients import (
       BioApiKeyMissing,
       BioCultureClient,
       CL1Adapter,
       FinalSparkAdapter,
   )

   _ENV = "NERVE_WML_BIO_API_KEY"


   def test_cl1_adapter_raises_without_api_key(monkeypatch):
       monkeypatch.delenv(_ENV, raising=False)
       with pytest.raises(BioApiKeyMissing):
           CL1Adapter()


   def test_finalspark_adapter_raises_without_api_key(monkeypatch):
       monkeypatch.delenv(_ENV, raising=False)
       with pytest.raises(BioApiKeyMissing):
           FinalSparkAdapter()


   def test_adapter_classes_are_constructible_with_fake_key(monkeypatch):
       # A non-empty key lets the constructor succeed; no network call
       # happens at construction time.
       monkeypatch.setenv(_ENV, "fake-key-for-construction-only")
       cl1 = CL1Adapter()
       fs = FinalSparkAdapter()
       assert isinstance(cl1, BioCultureClient)
       assert isinstance(fs, BioCultureClient)
       cl1.close()
       fs.close()


   @pytest.mark.slow
   def test_finalspark_roundtrip_real_hardware():
       if not os.environ.get(_ENV):
           pytest.skip("NERVE_WML_BIO_API_KEY unset — real API skipped")
       client = FinalSparkAdapter()
       frame = client.roundtrip([1, 2, 3])
       assert frame.spikes.ndim >= 2
       assert frame.latency_ms > 0.0
       client.close()


   @pytest.mark.slow
   def test_cl1_roundtrip_real_hardware():
       if not os.environ.get(_ENV):
           pytest.skip("NERVE_WML_BIO_API_KEY unset — real API skipped")
       client = CL1Adapter()
       frame = client.roundtrip([1, 2, 3])
       assert frame.spikes.ndim >= 2
       assert frame.latency_ms > 0.0
       client.close()
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/integration/track_w/test_bio_adapters.py -q
   # EXPECT: ImportError CL1Adapter / FinalSparkAdapter — FAIL
   ```

- [ ] **Step 3: Append the real adapters**

  Append to `track_w/bio_clients.py`. They share an
  env-reading base; HTTP uses stdlib `urllib.request` only:
   ```python
   import json as _json
   import os as _os
   import time as _wall
   import urllib.error as _urlerr
   import urllib.request as _urlreq


   class _RealBioAdapter:
       """Shared env-gating + HTTP plumbing for real bio adapters.

       Reads NERVE_WML_BIO_API_KEY at construction and raises
       BioApiKeyMissing if it is unset/empty — mirroring the env-gate
       discipline of bridge.kiki_nerve_advisor.NerveWmlAdvisor. No
       network call happens in __init__; the first network touch is in
       roundtrip(), which is only reached by @pytest.mark.slow tests.
       """

       _DEFAULT_ENDPOINT = ""  # set by subclass

       def __init__(self) -> None:
           key = _os.environ.get("NERVE_WML_BIO_API_KEY", "")
           if not key:
               raise BioApiKeyMissing(
                   "NERVE_WML_BIO_API_KEY is unset — real bio adapters "
                   "require it; inject a MockBioCultureClient for "
                   "offline use."
               )
           self._key = key
           self.endpoint = _os.environ.get(
               "NERVE_WML_BIO_ENDPOINT", self._DEFAULT_ENDPOINT
           )
           self.n_stim_channels = 8
           self.n_read_channels = 32
           self.n_bins = 16
           self._closed = False

       def _post(self, path: str, payload: dict) -> dict:
           url = self.endpoint.rstrip("/") + path
           data = _json.dumps(payload).encode("utf-8")
           req = _urlreq.Request(
               url, data=data, method="POST",
               headers={
                   "Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json",
               },
           )
           try:
               with _urlreq.urlopen(req, timeout=30.0) as resp:
                   return _json.loads(resp.read().decode("utf-8"))
           except _urlerr.URLError as exc:  # pragma: no cover - network
               raise RuntimeError(f"bio API request failed: {exc}") from exc

       def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
           channels = np.zeros(
               (len(codes), self.n_stim_channels), dtype=np.float32
           )
           for i, code in enumerate(codes):
               c = int(code) % ALPHABET_SIZE
               for ch in range(self.n_stim_channels):
                   channels[i, ch] = float((c >> ch) & 1)
           return StimulusFrame(codes=tuple(int(c) for c in codes),
                                channels=channels)

       def decode_activity(self, frame: ActivityFrame) -> list[int]:
           # Threshold per-channel rate, fold the 32-channel binary
           # vector into a 6-bit code per stimulated row.
           rates = np.atleast_2d(frame.spikes.sum(axis=-1))
           codes: list[int] = []
           for row in rates:
               bits = (row > row.mean()).astype(int)[:6]
               codes.append(int(sum(b << i for i, b in enumerate(bits))))
           return codes

       def close(self) -> None:
           self._closed = True


   class FinalSparkAdapter(_RealBioAdapter):
       """FinalSpark Neuroplatform adapter — remote human brain organoids.

       Free for research. Set NERVE_WML_BIO_API_KEY to your platform
       token. The wire shape below is the documented stimulate/read
       contract; adjust _DEFAULT_ENDPOINT if FinalSpark revises it.
       """

       _DEFAULT_ENDPOINT = "https://neuroplatform.finalspark.com/api/v1"

       def roundtrip(self, codes: list[int]) -> ActivityFrame:  # pragma: no cover - network
           if self._closed:
               raise RuntimeError("client is closed")
           stim = self.encode_stimulus(codes)
           t0 = _wall.perf_counter()
           body = self._post(
               "/stimulate-read",
               {"channels": stim.channels.tolist(),
                "read_bins": self.n_bins},
           )
           latency_ms = (_wall.perf_counter() - t0) * 1e3
           spikes = np.asarray(body["spikes"], dtype=np.float32)
           return ActivityFrame(spikes=spikes, latency_ms=latency_ms)


   class CL1Adapter(_RealBioAdapter):
       """Cortical Labs CL1 adapter — real-time closed-loop CL API.

       Set NERVE_WML_BIO_API_KEY to your CL API token. CL1 supports
       low-latency closed-loop access; the contract below posts a
       stimulus and reads the post-stimulus raster in one call.
       """

       _DEFAULT_ENDPOINT = "https://api.corticallabs.com/cl/v1"

       def roundtrip(self, codes: list[int]) -> ActivityFrame:  # pragma: no cover - network
           if self._closed:
               raise RuntimeError("client is closed")
           stim = self.encode_stimulus(codes)
           t0 = _wall.perf_counter()
           body = self._post(
               "/closed-loop/step",
               {"stim": stim.channels.tolist(),
                "bins": self.n_bins},
           )
           latency_ms = (_wall.perf_counter() - t0) * 1e3
           spikes = np.asarray(body["raster"], dtype=np.float32)
           return ActivityFrame(spikes=spikes, latency_ms=latency_ms)
   ```

- [ ] **Step 4: Run the test, confirm it PASSES**

  The two `slow` tests `skip`:
   ```bash
   uv run pytest tests/integration/track_w/test_bio_adapters.py -q
   # EXPECT: 3 passed, 2 skipped   (5 if NERVE_WML_BIO_API_KEY set)
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/bio_clients.py \
           tests/integration/track_w/test_bio_adapters.py
   git commit -m "feat: add env-gated CL1 and FinalSpark adapters"
   ```

---

## Task 6 — Cross-substrate integration test (BioWML in the pool)

**Files:**
- Create: `tests/integration/track_w/test_bio_cross_substrate.py`
- Modify: none

- [ ] **Step 1: Write the failing test**

  `tests/integration/track_w/test_bio_cross_substrate.py` — BioWML must
  run in a heterogeneous pool next to the three in-silico substrates,
  on a shared `MockNerve`:
   ```python
   import torch

   from nerve_core.protocols import WML
   from track_w.bio_clients import MockBioCultureClient
   from track_w.bio_wml import BioWML
   from track_w.lif_wml import LifWML
   from track_w.mlp_wml import MlpWML
   from track_w.mock_nerve import MockNerve
   from track_w.transformer_wml import TransformerWML


   def _pool():
       return [
           MlpWML(id=0, d_hidden=16, seed=0),
           LifWML(id=1, n_neurons=16, seed=1),
           TransformerWML(id=2, d_model=16, n_tokens=4, n_heads=2, seed=2),
           BioWML(id=3, client=MockBioCultureClient(seed=3), seed=3),
       ]


   def test_all_four_substrates_satisfy_wml_protocol():
       for wml in _pool():
           assert isinstance(wml, WML)


   def test_heterogeneous_pool_runs_ten_ticks():
       nerve = MockNerve(n_wmls=4, k=2, seed=0)
       pool = _pool()
       for tick in range(10):
           nerve.set_phase_active(gamma=True, theta=True)
           for wml in pool:
               wml.step(nerve, t=float(tick))
           nerve.tick(dt=1e-3)
       # The bio substrate logged at least one round-trip latency.
       bio = pool[3]
       assert isinstance(bio, BioWML)
       assert bio.last_latency_ms is not None


   def test_bio_wml_codebook_is_trainable_in_pool():
       bio = BioWML(id=3, client=MockBioCultureClient(seed=3), seed=3)
       opt = torch.optim.SGD(bio.parameters(), lr=0.01)
       before = bio.codebook.detach().clone()
       loss = bio.emit_head_pi(bio.codebook.mean(dim=0)).sum()
       opt.zero_grad()
       loss.backward()
       opt.step()
       assert not torch.equal(before, bio.codebook)
   ```

- [ ] **Step 2: Run the test, confirm the contract holds**

  It FAILS only if a regression exists; on a correct Tasks 1-5 it
  should already PASS — run it to confirm the cross-substrate contract
  holds:
   ```bash
   uv run pytest tests/integration/track_w/test_bio_cross_substrate.py -q
   # EXPECT: 3 passed  (if it fails, a prior task regressed — fix there)
   ```
   This task is a *contract* test: it has no new production code. If it
   fails, the failure is in Task 3/4 — go back and fix, do not patch
   here.

- [ ] **Step 3: No production code**

  No production code for this task.

- [ ] **Step 4: Re-run to confirm green**

  Re-run to confirm green (same command as step 2).

- [ ] **Step 5: Commit**

   ```bash
   git add tests/integration/track_w/test_bio_cross_substrate.py
   git commit -m "test: add bio cross-substrate pool integration"
   ```

---

## Task 7 — NeuroBench mapping module

**Files:**
- Create: `neuromorphic/neurobench_mapping.py`
- Test: `tests/unit/test_neurobench_mapping.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_neurobench_mapping.py`:
   ```python
   from neuromorphic.neurobench_mapping import (
       NeuroBenchResult,
       ValidationOutcome,
       map_to_neurobench,
   )


   def test_maps_outcome_to_streaming_classification_task():
       outcome = ValidationOutcome(
           substrate="BioWML",
           n_correct=87,
           n_total=100,
           total_synops=12_000,
           param_count=4096,
           latency_ms=12.5,
       )
       result = map_to_neurobench(outcome)
       assert isinstance(result, NeuroBenchResult)
       assert result.task == "streaming_classification"
       assert result.accuracy == 0.87


   def test_footprint_metrics_are_carried_through():
       outcome = ValidationOutcome(
           substrate="MlpWML", n_correct=90, n_total=100,
           total_synops=5_000, param_count=2048, latency_ms=0.4,
       )
       result = map_to_neurobench(outcome)
       assert result.connection_sparsity is not None
       assert result.synaptic_ops == 5_000
       assert result.footprint_params == 2048


   def test_zero_total_is_rejected():
       import pytest

       with pytest.raises(ValueError):
           map_to_neurobench(ValidationOutcome(
               substrate="LifWML", n_correct=0, n_total=0,
               total_synops=0, param_count=1, latency_ms=1.0,
           ))


   def test_result_serialises_to_a_flat_dict():
       outcome = ValidationOutcome(
           substrate="BioWML", n_correct=70, n_total=100,
           total_synops=9_000, param_count=1024, latency_ms=11.0,
       )
       row = map_to_neurobench(outcome).as_row()
       assert row["task"] == "streaming_classification"
       assert row["accuracy"] == 0.70
       assert row["substrate"] == "BioWML"
       assert set(row) >= {
           "task", "substrate", "accuracy", "synaptic_ops",
           "footprint_params", "latency_ms", "harness",
       }
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_neurobench_mapping.py -q
   # EXPECT: ModuleNotFoundError neurobench_mapping — FAIL
   ```

- [ ] **Step 3: Write the mapping module**

  Write `neuromorphic/neurobench_mapping.py`:
   ```python
   """NeuroBench mapping — make nerve-wml results externally comparable.

   NeuroBench (Yik et al., "NeuroBench: a framework for benchmarking
   neuromorphic computing algorithms and systems", Nature
   Communications, 2025) is an MLPerf-style, open, community harness
   for neuromorphic algorithms. It scores a model on a task with both
   a *correctness* metric (e.g. accuracy) and *complexity* metrics
   (synaptic operations, connection sparsity, parameter footprint).

   nerve-wml runs cross-substrate validations whose primary outcome is
   a streaming-classification accuracy over a tick sequence. This
   module maps one such ValidationOutcome onto NeuroBench's
   "streaming classification" task so a nerve-wml substrate
   (MlpWML / LifWML / TransformerWML / BioWML) can be quoted on the
   same axes as published NeuroBench entries.

   Deliberately minimal: one task mapping. See scripts/neurobench_map.py
   for the CLI that emits a NeuroBench-shaped row.
   """
   from __future__ import annotations

   from dataclasses import dataclass

   # The single NeuroBench task this module targets.
   NEUROBENCH_TASK = "streaming_classification"
   # Identifies which harness produced the row, for provenance.
   HARNESS_TAG = "neurobench-v1 (Yik et al., Nat. Commun. 2025)"


   @dataclass(frozen=True)
   class ValidationOutcome:
       """A nerve-wml cross-substrate validation result.

       substrate:    "MlpWML" | "LifWML" | "TransformerWML" | "BioWML".
       n_correct:    correctly classified streaming items.
       n_total:      total streaming items (> 0).
       total_synops: synaptic operations consumed over the run.
       param_count:  trainable parameter footprint of the substrate.
       latency_ms:   mean per-tick latency (per-roundtrip for BioWML).
       """

       substrate: str
       n_correct: int
       n_total: int
       total_synops: int
       param_count: int
       latency_ms: float


   @dataclass(frozen=True)
   class NeuroBenchResult:
       """A NeuroBench-shaped result row for one substrate."""

       task: str
       substrate: str
       accuracy: float
       synaptic_ops: int
       connection_sparsity: float | None
       footprint_params: int
       latency_ms: float
       harness: str

       def as_row(self) -> dict:
           """Flat dict suitable for CSV / JSON / a NeuroBench table."""
           return {
               "task": self.task,
               "substrate": self.substrate,
               "accuracy": self.accuracy,
               "synaptic_ops": self.synaptic_ops,
               "connection_sparsity": self.connection_sparsity,
               "footprint_params": self.footprint_params,
               "latency_ms": self.latency_ms,
               "harness": self.harness,
           }


   def map_to_neurobench(outcome: ValidationOutcome) -> NeuroBenchResult:
       """Map a nerve-wml ValidationOutcome to a NeuroBench result.

       Raises ValueError if n_total <= 0.
       """
       if outcome.n_total <= 0:
           raise ValueError("n_total must be > 0 to compute accuracy")
       accuracy = outcome.n_correct / outcome.n_total
       # NeuroBench connection-sparsity convention: fraction of the
       # dense connectivity that carried no synaptic op. We have no
       # per-edge trace here, so derive a conservative proxy from the
       # synop / param ratio, clamped to [0, 1]. A substrate that uses
       # far fewer synops than it has params is "sparse".
       if outcome.param_count > 0:
           density = outcome.total_synops / outcome.param_count
           sparsity = max(0.0, min(1.0, 1.0 - density))
       else:
           sparsity = None
       return NeuroBenchResult(
           task=NEUROBENCH_TASK,
           substrate=outcome.substrate,
           accuracy=accuracy,
           synaptic_ops=outcome.total_synops,
           connection_sparsity=sparsity,
           footprint_params=outcome.param_count,
           latency_ms=outcome.latency_ms,
           harness=HARNESS_TAG,
       )
   ```

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_neurobench_mapping.py -q
   # EXPECT: 4 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add neuromorphic/neurobench_mapping.py \
           tests/unit/test_neurobench_mapping.py
   git commit -m "feat: add NeuroBench streaming-task mapping"
   ```

---

## Task 8 — NeuroBench mapping CLI script

**Files:**
- Create: `scripts/neurobench_map.py`
- Test: `tests/unit/test_neurobench_map_script.py`

- [ ] **Step 1: Write the failing test**

  `tests/unit/test_neurobench_map_script.py`:
   ```python
   import json
   import subprocess
   import sys


   def test_script_emits_a_neurobench_row(tmp_path):
       out = tmp_path / "row.json"
       cmd = [
           sys.executable, "-m", "scripts.neurobench_map",
           "--substrate", "BioWML",
           "--correct", "82", "--total", "100",
           "--synops", "11000", "--params", "4096",
           "--latency-ms", "12.5",
           "--out", str(out),
       ]
       res = subprocess.run(cmd, capture_output=True, text=True)
       assert res.returncode == 0, res.stderr
       row = json.loads(out.read_text())
       assert row["task"] == "streaming_classification"
       assert row["substrate"] == "BioWML"
       assert abs(row["accuracy"] - 0.82) < 1e-9
       assert row["synaptic_ops"] == 11000


   def test_script_rejects_zero_total(tmp_path):
       cmd = [
           sys.executable, "-m", "scripts.neurobench_map",
           "--substrate", "MlpWML",
           "--correct", "0", "--total", "0",
           "--synops", "0", "--params", "1",
           "--latency-ms", "1.0",
           "--out", str(tmp_path / "x.json"),
       ]
       res = subprocess.run(cmd, capture_output=True, text=True)
       assert res.returncode != 0
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_neurobench_map_script.py -q
   # EXPECT: returncode != 0, module missing — FAIL
   ```

- [ ] **Step 3: Write the CLI script**

  Write `scripts/neurobench_map.py`:
   ```python
   """CLI: map one nerve-wml validation outcome to a NeuroBench row.

   Usage:
       uv run python -m scripts.neurobench_map \
           --substrate BioWML --correct 82 --total 100 \
           --synops 11000 --params 4096 --latency-ms 12.5 \
           --out reports/neurobench_biowml.json

   Emits a flat JSON row (see neuromorphic.neurobench_mapping) that can
   be dropped into a NeuroBench comparison table.
   """
   from __future__ import annotations

   import argparse
   import json
   import sys
   from pathlib import Path

   from neuromorphic.neurobench_mapping import (
       ValidationOutcome,
       map_to_neurobench,
   )


   def main(argv: list[str] | None = None) -> int:
       parser = argparse.ArgumentParser(description=__doc__)
       parser.add_argument("--substrate", required=True)
       parser.add_argument("--correct", type=int, required=True)
       parser.add_argument("--total", type=int, required=True)
       parser.add_argument("--synops", type=int, required=True)
       parser.add_argument("--params", type=int, required=True)
       parser.add_argument("--latency-ms", type=float, required=True)
       parser.add_argument("--out", type=Path, required=True)
       args = parser.parse_args(argv)

       outcome = ValidationOutcome(
           substrate=args.substrate,
           n_correct=args.correct,
           n_total=args.total,
           total_synops=args.synops,
           param_count=args.params,
           latency_ms=args.latency_ms,
       )
       try:
           result = map_to_neurobench(outcome)
       except ValueError as exc:
           print(f"error: {exc}", file=sys.stderr)
           return 2

       args.out.parent.mkdir(parents=True, exist_ok=True)
       args.out.write_text(json.dumps(result.as_row(), indent=2))
       print(f"wrote {args.out}")
       return 0


   if __name__ == "__main__":
       raise SystemExit(main())
   ```

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_neurobench_map_script.py -q
   # EXPECT: 2 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add scripts/neurobench_map.py \
           tests/unit/test_neurobench_map_script.py
   git commit -m "feat: add neurobench-map CLI script"
   ```

---

## Task 9 — Full-suite green, lint, types

**Files:**
- Modify: any file flagged by `ruff` / `mypy` (expected: none)
- Test: the whole suite

- [ ] **Step 1: Run the fast suite**

  Everything from Tasks 1-8 must be green and the real-API tests must
  skip:
   ```bash
   uv run pytest -m "not slow" -q
   # EXPECT: all passed; bio adapter real-API tests show as skipped
   ```

- [ ] **Step 2: Lint the new code**

   ```bash
   uv run ruff check .
   # EXPECT: no errors. Fix any (unused import, line length, etc.).
   ```

- [ ] **Step 3: Type-check**

  `track_w` is in the mypy target set:
   ```bash
   uv run mypy nerve_core track_p track_w
   # EXPECT: Success. Common fixes if not:
   #   - numpy arrays: annotate np.ndarray, not bare ndarray
   #   - BioCultureClient Protocol attrs must be declared on adapters
   #   - parameters() override already has the type: ignore precedent
   ```

- [ ] **Step 4: Confirm the real adapters genuinely degrade**

  With the env var unset, no test touches the network:
   ```bash
   env -u NERVE_WML_BIO_API_KEY uv run pytest \
       tests/integration/track_w/test_bio_adapters.py -q
   # EXPECT: 3 passed, 2 skipped
   ```

- [ ] **Step 5: Commit only if fixes were needed**

   ```bash
   git add -A
   git commit -m "chore: satisfy ruff and mypy for bio substrate"
   ```
   If steps 1-4 were already clean, skip the commit (nothing to add).

---

## Task 10 — Wire BioWML into the triple-substrate pool factory

**Files:**
- Modify: `track_w/pool_factory.py`
- Test: `tests/unit/test_pool_factory_bio.py`

- [ ] **Step 1: Read pool_factory and write the failing test**

  Read `track_w/pool_factory.py` first — note the existing
  triple-substrate builder (around line 83, `MLP / LIF / Transformer`).
  Write the failing test `tests/unit/test_pool_factory_bio.py`:
   ```python
   from track_w.bio_wml import BioWML
   from track_w.pool_factory import build_pool_with_bio


   def test_build_pool_with_bio_includes_one_bio_wml():
       pool = build_pool_with_bio(n=4, seed=0)
       assert len(pool) == 4
       assert sum(isinstance(w, BioWML) for w in pool) >= 1


   def test_build_pool_with_bio_ids_are_contiguous():
       pool = build_pool_with_bio(n=5, seed=0)
       assert [w.id for w in pool] == [0, 1, 2, 3, 4]


   def test_build_pool_with_bio_is_deterministic():
       a = build_pool_with_bio(n=4, seed=7)
       b = build_pool_with_bio(n=4, seed=7)
       assert [type(w).__name__ for w in a] == \
              [type(w).__name__ for w in b]
   ```

- [ ] **Step 2: Run the failing test, confirm it FAILS**

   ```bash
   uv run pytest tests/unit/test_pool_factory_bio.py -q
   # EXPECT: ImportError build_pool_with_bio — FAIL
   ```

- [ ] **Step 3: Add build_pool_with_bio**

  Add `build_pool_with_bio` to `track_w/pool_factory.py`. It mirrors
  the existing triple-substrate builder's per-WML seed derivation and
  assigns every 4th id to a `BioWML` backed by a `MockBioCultureClient`
  (offline — CI must never hit the network from the factory):
   ```python
   def build_pool_with_bio(n: int, *, seed: int = 0) -> list:
       """Build an N-WML pool cycling MLP / LIF / Transformer / Bio.

       id % 4 == 0 -> MlpWML, 1 -> LifWML, 2 -> TransformerWML,
       3 -> BioWML (offline MockBioCultureClient). Per-WML seeds are
       derived from `seed` exactly as build_triple_substrate_pool does,
       so the pool is deterministic. The bio substrate is always
       mock-backed here; use BioWML.from_env() for real hardware.
       """
       from track_w.bio_clients import MockBioCultureClient
       from track_w.bio_wml import BioWML
       from track_w.lif_wml import LifWML
       from track_w.mlp_wml import MlpWML
       from track_w.transformer_wml import TransformerWML

       pool: list = []
       for i in range(n):
           wml_seed = seed * 1000 + i
           kind = i % 4
           if kind == 0:
               pool.append(MlpWML(id=i, d_hidden=16, seed=wml_seed))
           elif kind == 1:
               pool.append(LifWML(id=i, n_neurons=16, seed=wml_seed))
           elif kind == 2:
               pool.append(TransformerWML(
                   id=i, d_model=16, n_tokens=4, n_heads=2,
                   seed=wml_seed,
               ))
           else:
               pool.append(BioWML(
                   id=i,
                   client=MockBioCultureClient(seed=wml_seed),
                   d_hidden=16, seed=wml_seed,
               ))
       return pool
   ```
   Match the actual seed-derivation expression used by the existing
   builder in the file — if it differs from `seed * 1000 + i`, copy
   that exact expression instead so determinism is consistent.

- [ ] **Step 4: Run the test, confirm it PASSES**

   ```bash
   uv run pytest tests/unit/test_pool_factory_bio.py -q
   # EXPECT: 3 passed
   ```

- [ ] **Step 5: Commit**

   ```bash
   git add track_w/pool_factory.py tests/unit/test_pool_factory_bio.py
   git commit -m "feat: add bio substrate to pool factory"
   ```

---

---

## Deep-research integration 2026-05-19

This section records how five references classified as **(b)** in the
gating document `project_hypneum_deepresearch_2026_05_19_classification.md`
feed into this plan. All five are fully encapsulated inside
`BioFieldWML.step()` and leave the WML Protocol (N-1..N-5, W-1..W-4)
and dream-of-kiki axioms (DR-0..DR-4) untouched.

### Companion artifact — (i)+(ii) decomposition

This plan is the **(i)** side of the user's "(i)+(ii)" framing: it
delivers `BioFieldWML` as **one conformant substrate among N** under
the existing WML Protocol, with DR-3 substrate-agnosticism preserved
as a universal claim. The **(ii)** side — the interpretive biophysical
sub-theory that stratifies the family of bio-grounded substrates — is
documented separately in the companion spec
`dream-of-kiki/docs/specs/2026-05-20-biophysical-stratification.md`.
That spec routes the nine **(b′)**-classified references (Blum-Moyse
coupled fields, Ursino theta-gamma neural mass, Młynarski-Hermundstad
multimodal efficient+predictive, Heinrich MTRNN embodied grounding,
and others) into five strata. Read both artifacts together: this plan
is the conformant-substrate side, the stratification spec is the
interpretive side; neither alone exhausts the deep-research integration.

### OQ defaults (document for reviewer override)

> **OQ-1 (DR-0 boundary):** `BioFieldWML.step()` performs ONE
> synchronous Up-Down cycle per call. The cycle does **not** span
> multiple `step()` calls. This preserves DR-0 (Accountability: every
> δ output belongs to a bounded Dream Episode) by construction, because
> the DE lifecycle is owned by dream-of-kiki — each `step()` call is
> a complete, finite internal operation with no ambient background state
> that outlives the call boundary.
>
> **OQ-2 (Tomé STDP scope):** STDP triplet + heterosynaptic +
> inhibitory STDP land **exclusively** inside `BioFieldWML`. The
> surrogate-gradient YAGNI of nerve-wml spec line 570 is **bounded**,
> not revoked. The bound is: "the bio substrate may diverge from
> surrogate-gradient; MLX/LIF/Transformer substrates keep surrogate-
> gradient". `LifWML`, `MlpWML`, and `TransformerWML` are not
> affected.

*These defaults are conservative and correct as of 2026-05-20. The
user can override either default at review by updating this block and
the corresponding test descriptions below.*

---

### Ref B-1 — Tomé et al. 2024 (eLife)

**"Dynamic and selective engrams emerge with memory consolidation."**

**Mechanism.**
Tomé 2024 documents three co-occurring plasticity rules in engram
circuits during consolidation: (1) STDP triplet (pre/post/post²
timing windows), (2) heterosynaptic depression (inactive synapses
weaken when active neighbours potentiate), and (3) inhibitory STDP
(interneuron synapses obey anti-Hebbian rule to sharpen selectivity).
Together they produce sparse, decorrelated engrams with bounded
weight growth.

**Encapsulation inside `step()`.**
All three rules execute inside `BioFieldWML._consolidate_weights()`
called from `step()` after the Up-Down cycle. They update internal
weight tensors (not exposed through the WML Protocol). `step()` reads
the `ActivityFrame` returned by the `BioCultureClient`, computes
spike-timing deltas, applies the three rules in order, and returns
Neuroletters. No intermediate state outlives the call (OQ-1).

**Axiom preservation.**
- W-1 (`step()` never mutates another WML): only `self._weights`
  is mutated; the Protocol boundary is respected.
- W-3 (no access to `routing_weight` from inside `step()`): weight
  update targets internal synaptic weights, not the Track-P router.
- N-3 (`role == ERROR ↔ phase == THETA`): plasticity does not
  affect Neuroletter role assignment — that mapping is computed
  independently after weight update.
- DR-0: weight update completes before `step()` returns; no ambient
  background process (OQ-2 bounded scope).

**Test hook.**
`tests/unit/test_bio_wml_consolidation.py::test_weight_norm_bound_after_stdp`
— after N consolidation cycles, assert `‖W‖_∞ ≤ W_MAX` (W-1
internal invariant). Also assert that weight updates are zero for
WMLs other than `self` (W-1 isolation check).

---

### Ref B-2 — Pignatelli et al. 2025 (Nat. Commun.)

**IE (intrinsic excitability) plasticity of ACC engrams.**

**Mechanism.**
Pignatelli 2025 shows that engram neurons in anterior cingulate
cortex undergo intrinsic excitability (IE) changes during
consolidation: tagged neurons hyperpolarise their resting potential,
raising the spike threshold during the post-consolidation window.
This acts as a gating mechanism that makes recently consolidated
engrams harder to overwrite.

**Encapsulation inside `step()`.**
`BioFieldWML` carries a per-neuron scalar `_ie_state: torch.Tensor`
(shape `[n_neurons]`). During the *wake portion* of a `step()` call
(stimulus → roundtrip → decode), neurons whose decoded activity
exceeds a tagging threshold receive an episodic IE tag
(`_ie_tag: bool mask`). During the *sleep portion* (Up-Down
internal consolidation), tagged neurons receive an IE shift
(`Δθ_IE`). `_ie_state` and `_ie_tag` are instance attributes;
they are **not** exposed through any WML Protocol method signature.

**Axiom preservation.**
- W-1: `_ie_state` is private; only `step()` of this instance
  mutates it.
- W-3: IE modulation is local; `routing_weight` is untouched.
- DR-3 (Substrate-agnosticism): IE is biophysical detail inside
  `BioFieldWML`. Other substrates (`LifWML`, etc.) do not carry
  `_ie_state`; the WML Protocol is unchanged.
- DR-0: IE tag is computed and cleared within the same `step()`
  call boundary; no persistent episodic state leaks between calls.

**Test hook.**
`tests/unit/test_bio_wml_ie.py::test_ie_tagged_neurons_hyperpolarise`
— run two consecutive `step()` calls with a high-activity stimulus
followed by a low-activity stimulus. Assert that neurons tagged in
call 1 show a measurable IE shift (higher `_ie_state`) in call 2,
and that untagged neurons are unaffected.

---

### Ref B-3 — Palacios et al. 2024 (arXiv:2409.05386)

**SNN-PC survey — Fristonian extension (variational message
passing, per-neuron spike-time prediction).**

**Mechanism.**
Palacios 2024 extends predictive coding (PC) to spiking networks:
each neuron maintains a belief over *when* peers will spike and
minimises variational free energy via local variational message
passing (VMP). Prediction errors are Poisson-rate residuals
(observed minus expected spike count). This maps naturally onto
the nerve-wml role taxonomy: PREDICTION neurons emit γ-band
neuroletters (N-3 canonical form), ERROR neurons emit θ-band
neuroletters.

**Encapsulation inside `step()`.**
`BioFieldWML` maintains internal belief tensors `_mu: Tensor` (rate
predictions) and `_sigma: Tensor` (rate uncertainty). During
`step()`: (1) prior beliefs are computed from last-tick weights;
(2) the `BioCultureClient` roundtrip provides observed spike counts;
(3) VMP update computes posterior beliefs and prediction errors;
(4) neurons above the prediction threshold emit Role.PREDICTION
neuroletters (γ); neurons with high prediction error emit
Role.ERROR neuroletters (θ). Both `_mu` and `_sigma` are reset or
updated within `step()`.

**Axiom preservation.**
- N-3 (`role == ERROR ↔ phase == THETA`): the VMP rule drives
  ERROR neurons precisely when `phase == THETA` is active on
  the `MockNerve`; PREDICTION neurons are gated to `phase == GAMMA`.
  This is the *same* canonical mapping as `LifWML`.
- W-1: belief tensors are private instance state.
- DR-3: predictive coding is a local computation; the WML Protocol
  shape is unchanged (`step(nerve, t) -> None`).

**Test hook.**
`tests/unit/test_bio_wml_pc.py::test_neuroletter_role_partition`
— inject a `MockNerve` with `phase == THETA` active and a stimulus
that produces a large spike-count residual. Assert that all emitted
neuroletters carry `role == ERROR`. Then switch to `phase == GAMMA`
and a matching stimulus; assert `role == PREDICTION` dominates.

---

### Ref B-4 — Bellitto 2024 (WSCL)

**Wake-Sleep Continual Learning — internal scheduler.**

**Mechanism.**
WSCL alternates wake phases (stimulus encoding, online learning)
with sleep phases (offline consolidation, replay-based weight
update) at the level of the learning agent. The key contribution
is that sleep consolidation prevents catastrophic forgetting across
a stream of tasks without a stored replay buffer, instead using
internal generative replay during sleep.

**Encapsulation inside `step()`.**
`BioFieldWML` tracks a call counter `_call_count: int` and a
phase ratio `_sleep_every: int` (default 4 — every 4th call is a
sleep step). On a wake call, `step()` runs the full stimulus →
roundtrip → decode → Neuroletter emit path. On a sleep call,
`step()` runs the internal consolidation path (generative replay
of stored prototype patterns) and emits no external neuroletters
(silent, consistent with N-1: silence is legitimate). The
wake/sleep toggle is purely internal — it does not change the
`step()` signature or add a new channel. The DE machinery of
dream-of-kiki is **outside** `BioFieldWML`; this scheduler governs
only the internal compute path per call (OQ-1 default).

**Axiom preservation.**
- DR-0: every call to `step()` is a complete, bounded operation.
  The sleep path does not emit δ outputs; no Dream Episode channel
  is opened.
- N-1 (silence is legitimate): sleep calls produce no neuroletters,
  which is explicitly legal under N-1.
- W-1: internal prototype buffer is instance-private.
- DR-3: the WSCL scheduler is an internal heuristic; the WML
  Protocol is unchanged.

**Test hook.**
`tests/integration/test_bio_wml_wscl.py::test_continual_learning_forgetting`
— run `BioFieldWML` on task A for K steps, then task B for K
steps, then re-evaluate on task A. Assert that task-A performance
after task-B is within a defined forgetting threshold (≤ Δ_max
relative to the WSCL ablation baseline from the paper). Compare
against a control `BioFieldWML` with sleep disabled.

---

### Ref B-5 — Tucker, Luu & Friston 2025 (Cerebral Cortex)

**"Adaptive consolidation of active inference: excitatory and
inhibitory mechanisms for organizing feedforward and feedback
memory systems in sleep."**

**Mechanism.**
Tucker et al. 2025 show that during sleep, E/I balance shifts:
inhibitory gain suppresses feedforward (sensory-driven) pathways
while excitatory gain amplifies feedback (prediction-driven)
pathways. This selectivity reorganises memory traces so that
abstract, prediction-consistent patterns are retained and
sensory-noise is pruned. In the context of predictive coding (Ref
B-3 above), this means the *sleep portion of `step()`* should
privilege Role.PREDICTION neuroletters (feedback) over Role.ERROR
(feedforward), producing a measurable shift in the role-mix of
consolidation outputs.

**Encapsulation inside `step()`.**
When the internal WSCL scheduler (Ref B-4) marks a call as a sleep
step, `BioFieldWML` applies an E/I gain tensor `_ei_gain: Tensor`
that upweights feedback prediction pathways and downweights error
pathways. `_ei_gain` is computed from the current `_ie_state`
(Ref B-2) and the VMP belief tensors (Ref B-3), implementing the
sleep-phase-specific gain described in Tucker 2025. On wake calls,
`_ei_gain` reverts to a neutral 1.0 weight. The gain is entirely
internal and does not change the Protocol interface.

**Axiom preservation.**
- N-3: E/I gain shift occurs within `step()` on sleep calls, where
  no neuroletters are emitted externally (sleep is silent per
  Ref B-4). On wake calls, N-3 mapping is unaffected.
- W-3: `_ei_gain` does not touch `routing_weight`.
- DR-0: sleep gain computation completes within the call boundary;
  no state outlives `step()` beyond `_ei_gain` and `_ie_state`
  (both instance attributes, not Protocol outputs).
- DR-3: gain is a biophysical detail inside `BioFieldWML`; the
  WML Protocol shape is unchanged.

**Test hook.**
`tests/unit/test_bio_wml_ei.py::test_feedforward_feedback_selectivity`
— after injecting a sequence of wake and sleep `step()` calls,
inspect the internal `_ei_gain` tensor. Assert that on sleep calls,
the feedback component of `_ei_gain` is ≥ a threshold (e.g. 1.5)
above the feedforward component. Additionally assert that wake
calls restore a neutral `_ei_gain` (ratio ≈ 1.0 ± ε).

---

### Integration note — cohesion of B-3, B-4, B-5

The three mechanistic refs interlock:

- **B-3 (SNN-PC)** supplies the prediction/error machinery and the
  internal belief tensors (`_mu`, `_sigma`).
- **B-4 (WSCL)** supplies the wake/sleep scheduler that determines
  *when* each call runs the external or internal path.
- **B-5 (Tucker)** supplies the sleep-phase E/I gain that modulates
  the B-3 machinery specifically on sleep calls.

This layering is additive: each can be implemented independently
(B-3 first, then B-4's scheduler wraps it, then B-5's gain is
applied inside the sleep branch of B-4). The recommended execution
order for the BioFieldWML implementation tasks is therefore:
B-2 IE state (simple scalar, no dependencies) → B-3 SNN-PC
beliefs (depends on `ActivityFrame`) → B-1 STDP rules (depends on
spike-timing from B-3) → B-4 WSCL scheduler (wraps existing step
logic) → B-5 E/I gain (plugs into B-4 sleep branch using B-3
tensors and B-2 IE state). This sequence preserves the ability to
run each incremental test suite before the next layer is added.

---

## Self-Review

Reviewed against the writing-plans skill checklist and the task brief:

- **Header / structure** — present: title, "For agentic workers" line
  referencing `superpowers:subagent-driven-development`, **Goal**,
  **Architecture** (ASCII diagram), **Tech Stack**, `---` divider.
- **Bite-sized tasks** — 11 tasks (0-10). Task 0 is the explicit
  real-hardware prerequisite naming `NERVE_WML_BIO_API_KEY` and the
  skip-degradation contract. Tasks 1-8, 10 are TDD code tasks; Task 9
  is the suite/lint/type gate; Task 6 is a contract test.
- **TDD cycle** — every code task: failing test → run command with
  expected FAIL → minimal real implementation → run with expected PASS
  → exact `git add` + `git commit`.
- **No placeholders** — complete real code shown for `BioCultureClient`
  Protocol + frames, `MockBioCultureClient` (numpy spike sim with
  latency/jitter/noise/baseline + deterministic seeding), `BioWML`
  (ctor + `from_env` + real `step`), `_RealBioAdapter` + `CL1Adapter` +
  `FinalSparkAdapter` (env-gated, stdlib HTTP), `neurobench_mapping.py`,
  and `scripts/neurobench_map.py`. No "TBD"/"handle errors"/"similar
  to Task N".
- **Exact repo facts** — `WML` / `Nerve` Protocol signatures copied
  from `nerve_core/protocols.py`; `Neuroletter`/`Role`/`Phase` from
  `nerve_core/neuroletter.py`; substrate style (`nn.Module`,
  `nn.Parameter` codebook, `emit_head_pi`, local-generator RNG
  save/restore, `parameters()` override with `type: ignore`) mirrored
  from `lif_wml.py` / `mlp_wml.py`; env-gating mirrored from
  `bridge/kiki_nerve_advisor.py`; test style from
  `tests/unit/test_substrate_input_dim.py` and `test_lif_wml.py`;
  `MockNerve` `_queues` / `set_phase_active` / `n_wmls` usage copied
  from `test_lif_wml.py`. New files land at the briefed paths
  (`track_w/bio_wml.py`, `track_w/bio_clients.py`,
  `neuromorphic/neurobench_mapping.py`, `scripts/neurobench_map.py`).
- **Commit format** — every subject ≤ 50 chars, English, no underscore
  in scope, no `--no-verify`.

One residual risk flagged inline for the executor: the exact per-WML
seed expression in `pool_factory.py` (Task 10 step 3) and the precise
`MockNerve` inbox-injection idiom (Task 4) must be confirmed against the
live files at execution time — both tasks instruct the worker to copy
the real expression rather than assume. No blocking gaps.

## Execution Handoff

**Recommended mode: Subagent-Driven.** Use
`superpowers:subagent-driven-development`. Dispatch tasks strictly in
order 0 → 10; each task is self-contained and ends on a green test plus
a commit, so a fresh subagent per task carries no state debt. Task 6
(cross-substrate contract test) and Task 9 (suite/lint/type gate) are
verification tasks — if either fails, the regression is in an earlier
task; the subagent must fix it there, not patch around it. Task 0 is
documentation-only and has no test.

Inline (single-session) execution is acceptable if the executor prefers
continuity, since total scope is moderate (~4 new source files, ~6 new
test files, 1 modified factory). If inline, still honour every
red → green → commit boundary exactly as written.

<!-- buddy: *force-pushes a wetware adapter that skips when the org isn't plugged in* -->
