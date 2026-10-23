"""Generate the torch GTM golden reference for the N10-C MLX port.

Run in the repo (torch) env: uv run python scripts/n10c_make_gtm_golden.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "n10c_gtm_k4_a16.npz"


def main() -> None:
    mux = GammaThetaMultiplexer(
        GammaThetaConfig(alphabet_size=16, symbols_per_theta=4), seed=0
    )
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (32, 4), generator=gen)
    with torch.no_grad():
        carrier = mux(codes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        constellation=mux.constellation.detach().numpy(),
        codes=codes.numpy().astype(np.int64),
        carrier=carrier.numpy(),
    )
    print(f"wrote {OUT} carrier{tuple(carrier.shape)}")


if __name__ == "__main__":
    main()
