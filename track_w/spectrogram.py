"""Spectrogram → carrier encoder for audio / bio-signal consumers (issue #7).

Takes a raw 1-D waveform tensor and returns a fixed-size carrier vector
ready for ``SensoryWML``. Internally: ``torch.stft`` → magnitude → top-N
frequency bins → temporal mean → linear projection.

Used by ``MlpWML.from_spectrogram(...)`` as a canonical "audio in" path so
``bouba_sens`` (MIT-BIH ECG, Studyforrest audio) and any future consumer
share one implementation instead of re-deriving the FFT pipeline three
times.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectrogramEncoder(nn.Module):
    """Raw waveform → carrier embedding via STFT + magnitude + linear projection.

    Parameters
    ----------
    sample_rate
        Hz of the input waveform. Sets the ``n_fft`` and ``hop_length``
        internally based on ``window_sec`` and ``hop_sec``.
    window_sec
        STFT window length in seconds.
    hop_sec
        STFT hop length in seconds.
    n_bins
        Number of low-frequency bins kept after the FFT (top of the
        spectrum is dropped). Must satisfy ``n_bins <= n_fft // 2 + 1``.
    target_carrier_dim
        Output carrier dimensionality. The internal linear layer maps
        ``n_bins → target_carrier_dim``.
    seed
        Optional seed for the linear projection init. ``None`` keeps the
        global RNG.

    Notes
    -----
    Forward expects a ``(B, T)`` or ``(T,)`` float tensor. The 1-D case
    is auto-unsqueezed to ``(1, T)``. Output shape is always
    ``(B, target_carrier_dim)``.
    """

    def __init__(
        self,
        sample_rate: int,
        window_sec: float = 1.0,
        hop_sec: float = 0.05,
        n_bins: int = 128,
        target_carrier_dim: int = 16,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = max(2, int(round(window_sec * sample_rate)))
        self.hop_length = max(1, int(round(hop_sec * sample_rate)))
        self.n_bins = int(n_bins)
        self.target_carrier_dim = int(target_carrier_dim)

        max_bins = self.n_fft // 2 + 1
        if self.n_bins > max_bins:
            raise ValueError(
                f"n_bins={self.n_bins} exceeds available rfft bins {max_bins} "
                f"for n_fft={self.n_fft}",
            )

        # Hann window (registered as buffer so .to(device) follows it).
        self.register_buffer("_window", torch.hann_window(self.n_fft))

        # Linear projection n_bins → target_carrier_dim, seed-controlled init.
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        self.projection = nn.Linear(self.n_bins, self.target_carrier_dim)
        with torch.no_grad():
            self.projection.weight.data = (
                torch.randn(*self.projection.weight.shape, generator=gen) * 0.1
            )
            self.projection.bias.data.zero_()

    def forward(self, waveform: Tensor) -> Tensor:
        """Encode a raw waveform into a carrier.

        Parameters
        ----------
        waveform
            ``(B, T)`` or ``(T,)`` float tensor of raw samples.

        Returns
        -------
        Tensor
            ``(B, target_carrier_dim)`` carrier features (mean-pooled over
            time frames after magnitude STFT, then linearly projected).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2:
            raise ValueError(
                f"expected (B, T) or (T,) waveform, got shape {tuple(waveform.shape)}",
            )

        # STFT — center=False keeps time alignment with raw samples and
        # avoids implicit padding semantics that would shift bin power.
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            return_complex=True,
            center=False,
        )  # (B, n_freq, n_frames)

        # Magnitude, keep low n_bins (high frequencies dropped),
        # mean over time → (B, n_bins).
        mag = spec.abs()
        low_bins = mag[:, : self.n_bins, :].mean(dim=-1)
        return self.projection(low_bins)
