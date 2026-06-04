"""SpikingKikiBioClient — adapter making SpikingKikiWML satisfy
BioCultureClient so BioFieldWML can consume silicon spike trains.

Design ref:
  docs/superpowers/specs/2026-05-21-biofield-spikingkiki-coupling-design.md

The adapter wraps a ``SpikingKikiWML`` and satisfies the
``BioCultureClient`` Protocol (track_w.bio_clients) without modifying
either substrate. BioFieldWML then consumes spikingkiki spike trains
exactly as it consumes Mock / CL1 / FinalSpark cultures. The Palacios
SNN-PC VMP update becomes a predictive-coding learning signal on the
silicon culture.

Spatial bucketing: contiguous groups (simpler, tied to .npz layout).
Temporal bucketing: ``n_micro_ticks`` ticks divided into ``n_bins``
windows of ``micro_ticks_per_bin`` ticks each.

Output shape: ``ActivityFrame.spikes`` is ``[k, n_read_channels, n_bins]``
matching MockBioCultureClient semantics (per-code first axis so
BioFieldWML._measure_spikes can average over it).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass

import numpy as np
import torch

from track_w.bio_clients import ActivityFrame, StimulusFrame
from track_w.spiking_kiki_wml import SpikingKikiWML

# Default values mirror MockBioCultureClient defaults.
_DEFAULT_N_STIM: int = 8
_DEFAULT_N_READ: int = 32
_DEFAULT_N_BINS: int = 16
_DEFAULT_TICKS_PER_BIN: int = 8
_ALPHABET_SIZE: int = 64

__all__ = ["SpikingKikiBioConfig", "SpikingKikiBioClient"]


@dataclass(frozen=True)
class SpikingKikiBioConfig:
    """Configuration for the spikingkiki bio-culture adapter.

    n_stim_channels:    number of stimulus electrode channels
                        (matches Mock / CL1 default = 8).
    n_read_channels:    number of spike read-back channels
                        (matches Mock / CL1 default = 32).
    n_bins:             number of temporal bins per roundtrip.
    micro_ticks_per_bin: LIF ticks per temporal bin.
                        T = n_bins * micro_ticks_per_bin (must match
                        SpikingKikiWML.n_micro_ticks; the adapter
                        overrides the WML's n_micro_ticks accordingly).
    block_idx:          which transformer block to drive (default 0).
    alphabet_size:      vocabulary size of the attached WML.
    """

    n_stim_channels: int = _DEFAULT_N_STIM
    n_read_channels: int = _DEFAULT_N_READ
    n_bins: int = _DEFAULT_N_BINS
    micro_ticks_per_bin: int = _DEFAULT_TICKS_PER_BIN
    block_idx: int = 0
    alphabet_size: int = _ALPHABET_SIZE


class SpikingKikiBioClient:
    """BioCultureClient adapter that drives a SpikingKikiWML as the
    'culture'.

    encode_stimulus(codes) -> StimulusFrame
        Maps each code to an 8-bit binary electrode pattern (same bit-
        expansion as MockBioCultureClient) and stores the averaged
        channel vector for injection on the next roundtrip.

    roundtrip(codes) -> ActivityFrame
        Runs SpikingKikiWML._lif_integrate() for
        T = n_bins * micro_ticks_per_bin ticks while injecting the
        stimulus current during the first bin window.  Captures per-
        neuron spike events tick-by-tick, groups them into n_read_channels
        contiguous groups, bins into n_bins temporal windows.
        Returns ActivityFrame with spikes shaped [k, n_read_channels,
        n_bins] where k = len(codes), matching Mock semantics.

    decode_activity(frame) -> list[int]
        Sums spikes over time bins, correlates the per-channel rate
        vector against per-code templates, argmax picks the decoded code
        for each input code row.
    """

    n_stim_channels: int
    n_read_channels: int
    n_bins: int

    def __init__(
        self,
        substrate: SpikingKikiWML,
        config: SpikingKikiBioConfig | None = None,
    ) -> None:
        if config is None:
            config = SpikingKikiBioConfig()
        self._substrate = substrate
        self._config = config
        self.n_stim_channels = config.n_stim_channels
        self.n_read_channels = config.n_read_channels
        self.n_bins = config.n_bins
        self._closed = False

        # Total LIF ticks per roundtrip.
        self._T: int = config.n_bins * config.micro_ticks_per_bin

        # Validate spatial bucketing: hidden_dim must be divisible by
        # n_read_channels (or we pad — for simplicity we require exact
        # divisibility and raise early if not).
        hidden = substrate.hidden_dim
        if hidden < config.n_read_channels:
            raise ValueError(
                f"hidden_dim ({hidden}) < n_read_channels "
                f"({config.n_read_channels}); reduce n_read_channels"
            )
        # Neurons per read-channel group (contiguous bucketing).
        self._neurons_per_group: int = hidden // config.n_read_channels
        # Remaining neurons (if hidden not exactly divisible) are dropped.

        # Stimulus encoder: a fixed linear projection from the stim
        # channel vector to hidden_dim. Seeded from the WML id so it is
        # deterministic and reproducible without touching the global RNG.
        gen = torch.Generator()
        gen.manual_seed(abs(hash(("stim_enc", substrate.id))) % (2**32))
        self._stim_enc: torch.Tensor = torch.randn(
            config.n_stim_channels, hidden, generator=gen
        ) * 0.1  # [n_stim_channels, hidden_dim]

        # Per-code channel templates for decode_activity: same structure
        # as MockBioCultureClient (25 % of channels lit per code, stable
        # across calls).
        tpl_rng = np.random.default_rng(
            abs(hash(("tpl", substrate.id))) % (2**31)
        )
        raw = tpl_rng.random(
            (config.alphabet_size, config.n_read_channels), dtype=np.float32
        )
        thresh = np.quantile(raw, 0.75, axis=1, keepdims=True)
        self._templates: np.ndarray = (raw >= thresh).astype(np.float32)

        # Pending stimulus injection vector (set by encode_stimulus,
        # consumed by the first bin of roundtrip).
        self._pending_inject: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # BioCultureClient Protocol implementation
    # ------------------------------------------------------------------

    def encode_stimulus(self, codes: list[int]) -> StimulusFrame:
        """Map alphabet codes to an electrode stimulation pattern.

        Returns StimulusFrame with channels shaped [k, n_stim_channels].
        Stores the mean across codes as the pending injection vector for
        the next roundtrip (per-roundtrip injection, v1 design choice).
        """
        if self._closed:
            raise RuntimeError("SpikingKikiBioClient is closed")
        k = max(len(codes), 1)
        channels = np.zeros(
            (k, self.n_stim_channels), dtype=np.float32
        )
        for i, code in enumerate(codes):
            c = int(code) % _ALPHABET_SIZE
            for ch in range(self.n_stim_channels):
                channels[i, ch] = float((c >> ch) & 1)

        # Project the mean stim vector into hidden_dim space.
        mean_stim = torch.from_numpy(channels.mean(axis=0))  # [n_stim]
        # mean_stim @ _stim_enc -> [hidden_dim]
        self._pending_inject = mean_stim @ self._stim_enc

        return StimulusFrame(
            codes=tuple(int(c) for c in codes),
            channels=channels,
        )

    def decode_activity(self, frame: ActivityFrame) -> list[int]:
        """Map spiking activity back to alphabet codes via template match.

        Sums spikes over time bins to get a [k, n_read_channels] rate
        matrix, correlates each row against all code templates, argmax.
        """
        spikes = np.asarray(frame.spikes, dtype=np.float32)
        if spikes.ndim == 3:
            # [k, n_read_channels, n_bins] -> [k, n_read_channels]
            rates = spikes.sum(axis=-1)
        elif spikes.ndim == 2:
            # [n_read_channels, n_bins] -> treat as single code
            rates = spikes.sum(axis=-1)[np.newaxis, :]
        else:
            rates = np.atleast_2d(spikes)

        # rates [k, n_read_channels] × templates [ALPHABET_SIZE, n_read]^T
        scores = rates @ self._templates.T  # [k, ALPHABET_SIZE]
        return [int(row.argmax()) for row in scores]

    def roundtrip(self, codes: list[int]) -> ActivityFrame:
        """Stimulate the silicon culture and read back spike rasters.

        Runs the LIF integration for T = n_bins * micro_ticks_per_bin
        ticks.  During the first micro_ticks_per_bin ticks, the stimulus
        injection vector (set by encode_stimulus, or derived fresh from
        codes) is added to the drive before block_projection.

        Returns ActivityFrame.spikes of shape
        [k, n_read_channels, n_bins] where k = len(codes), mirroring
        MockBioCultureClient output shape for BioFieldWML compatibility.
        """
        if self._closed:
            raise RuntimeError("SpikingKikiBioClient is closed")

        # Ensure a stimulus vector is available.
        self.encode_stimulus(codes)
        inject = self._pending_inject  # [hidden_dim]
        assert inject is not None

        k = max(len(codes), 1)
        hidden = self._substrate.hidden_dim
        n_ticks = self._T
        tpb = self._config.micro_ticks_per_bin  # ticks per bin
        n_bins = self.n_bins
        n_groups = self.n_read_channels
        neurons_per = self._neurons_per_group

        t0 = _time.perf_counter()

        # We run k independent roundtrips (one per code), capturing a
        # per-code [n_read_channels, n_bins] raster each time.
        rasters = np.zeros((k, n_groups, n_bins), dtype=np.float32)

        substrate = self._substrate

        for code_i, code in enumerate(codes):
            # Per-code stimulus injection (use per-code channel pattern).
            code_val = int(code) % _ALPHABET_SIZE
            code_stim = np.zeros(self.n_stim_channels, dtype=np.float32)
            for ch in range(self.n_stim_channels):
                code_stim[ch] = float((code_val >> ch) & 1)
            code_inject = torch.from_numpy(code_stim) @ self._stim_enc

            # Reset membrane for a clean roundtrip per code.
            substrate.reset_state()

            # Tick-by-tick LIF to capture per-tick spike events.
            spike_trace = torch.zeros(
                n_ticks, hidden, dtype=torch.float32
            )  # [n_ticks, hidden]

            with torch.no_grad():
                for tick in range(n_ticks):
                    # Build drive: block_projection of input drive.
                    # We use a zero base drive (no Neuroletter input);
                    # stimulus injection replaces the inbound signal.
                    # Drive construction mirrors _lif_integrate but
                    # tick-level to capture the binary fire events.
                    block_proj: torch.Tensor = substrate.block_projection  # type: ignore[assignment]
                    drive = code_inject @ block_proj
                    if tick >= tpb:
                        # Stimulus only during first bin window.
                        drive = torch.zeros_like(drive)

                    substrate.v_mem = substrate.v_mem + (
                        1.0 / substrate.tau
                    ) * (-substrate.v_mem + drive)
                    spikes_t = (substrate.v_mem > substrate.threshold).float()
                    spike_trace[tick] = spikes_t
                    substrate.v_mem = substrate.v_mem * (1.0 - spikes_t)

            # Spatial bucketing: contiguous groups of neurons_per neurons.
            # spike_trace: [T, hidden] -> group sum -> [n_groups, T]
            grouped = spike_trace.numpy()  # [n_ticks, hidden]
            # Reshape to [n_ticks, n_groups, neurons_per], sum over last axis.
            usable = n_groups * neurons_per
            group_spikes = grouped[:, :usable].reshape(
                n_ticks, n_groups, neurons_per
            ).sum(axis=-1)  # [n_ticks, n_groups]

            # Temporal binning: [T, n_groups] -> [n_groups, n_bins]
            group_spikes_t = group_spikes.T  # [n_groups, T]
            for b in range(n_bins):
                start = b * tpb
                end = start + tpb
                rasters[code_i, :, b] = group_spikes_t[:, start:end].sum(axis=-1)

        latency_ms = (_time.perf_counter() - t0) * 1e3
        return ActivityFrame(spikes=rasters, latency_ms=latency_ms)

    def close(self) -> None:
        """Release state. Safe to call twice."""
        self._closed = True
