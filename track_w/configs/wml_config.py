"""WmlConfig — named-size presets for the three substrate types.

Decouples task-input dim (input_dim) from internal substrate sizes
(d_hidden, n_neurons, d_model, n_heads, n_layers, n_tokens) and from
the emission alphabet size. Presets (v1_1, mnist, large) standardize
the configurations used by plans and tests.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WmlConfig:
    # Task input dim (feature width fed to each substrate).
    input_dim:     int = 16
    # Substrate-specific widths.
    d_hidden:      int = 16        # MLP core width
    n_neurons:     int = 16        # LIF population size
    d_model:       int = 16        # Transformer embedding
    n_layers:      int = 2
    n_heads:       int = 2
    n_tokens:      int = 4
    # Protocol alphabet.
    alphabet_size: int = 64

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.d_model % self.n_tokens != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_tokens ({self.n_tokens})"
            )

    @classmethod
    def mnist(cls) -> "WmlConfig":
        return cls(
            input_dim=784, d_hidden=128, n_neurons=128, d_model=128,
            n_layers=2, n_heads=4, n_tokens=8, alphabet_size=256,
        )

    @classmethod
    def large(cls) -> "WmlConfig":
        return cls(
            input_dim=128, d_hidden=256, n_neurons=256, d_model=256,
            n_layers=4, n_heads=8, n_tokens=8, alphabet_size=256,
        )
