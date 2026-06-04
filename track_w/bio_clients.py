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

import json as _json
import os as _os
import time as _time
import urllib.error as _urlerr
import urllib.request as _urlreq
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

ALPHABET_SIZE = 64


class BioApiKeyMissingError(RuntimeError):
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
            baseline = np.asarray(
                self._rng.poisson(self.baseline_rate, size=evoked.shape),
                dtype=np.float32,
            )
            noise = np.asarray(
                self._rng.normal(0.0, self.noise, size=evoked.shape),
                dtype=np.float32,
            )
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


class _RealBioAdapter:
    """Shared env-gating + HTTP plumbing for real bio adapters.

    Reads NERVE_WML_BIO_API_KEY at construction and raises
    BioApiKeyMissingError if it is unset/empty — mirroring the
    env-gate discipline of bridge.kiki_nerve_advisor.NerveWmlAdvisor.
    No network call happens in __init__; the first network touch is
    in roundtrip(), which is only reached by @pytest.mark.slow tests.
    """

    _DEFAULT_ENDPOINT = ""  # set by subclass

    def __init__(self) -> None:
        key = _os.environ.get("NERVE_WML_BIO_API_KEY", "")
        if not key:
            raise BioApiKeyMissingError(
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

    def _post(self, path: str, payload: dict) -> dict:  # pragma: no cover - network
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
        except _urlerr.URLError as exc:
            raise RuntimeError(f"bio API request failed: {exc}") from exc

    def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
        channels = np.zeros(
            (len(codes), self.n_stim_channels), dtype=np.float32
        )
        for i, code in enumerate(codes):
            c = int(code) % ALPHABET_SIZE
            for ch in range(self.n_stim_channels):
                channels[i, ch] = float((c >> ch) & 1)
        return StimulusFrame(
            codes=tuple(int(c) for c in codes), channels=channels
        )

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
        t0 = _time.perf_counter()
        body = self._post(
            "/stimulate-read",
            {"channels": stim.channels.tolist(),
             "read_bins": self.n_bins},
        )
        latency_ms = (_time.perf_counter() - t0) * 1e3
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
        t0 = _time.perf_counter()
        body = self._post(
            "/closed-loop/step",
            {"stim": stim.channels.tolist(),
             "bins": self.n_bins},
        )
        latency_ms = (_time.perf_counter() - t0) * 1e3
        spikes = np.asarray(body["raster"], dtype=np.float32)
        return ActivityFrame(spikes=spikes, latency_ms=latency_ms)
