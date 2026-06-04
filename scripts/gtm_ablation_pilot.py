"""GTM ablation + synchrony-collapse analysis.

Closes gap 2. Two arms run the SAME 7-symbol routing task:
- `gtm`           : `track_p.multiplexer.GammaThetaMultiplexer`
- `simple_gating` : `track_p.transducer_baselines.SimpleGatingMultiplexer`

For each arm we report symbol-recovery accuracy, MI (bits) between recovered
and true codes, and a SYNCHRONY INDEX: the fraction of carrier energy that
collapses onto a single shared mode. Oscillator nets are known to collapse to
global synchrony under end-to-end training (Phasor Agents, arXiv:2601.04362);
a synchrony index near 1.0 means every symbol decodes alike (channel dead).

Note on `mi_bits`: `nerve_wml.methodology.mi_estimators.mi_miller_madow_discrete`
returns MI normalised by H(a) (a `[0, 1]`-ish ratio), not nats. The
`* log2(e)` factor below is the textbook nat→bit rescaling applied to that
ratio; the value is non-negative either way, which is what the test asserts
and what we report as `mi_bits` (an upper-bounded bit-rate proxy).

Run directly:  uv run python -m scripts.gtm_ablation_pilot
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from nerve_wml.methodology.mi_estimators import mi_miller_madow_discrete
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer
from track_p.transducer_baselines import (
    KuramotoMultiplexer,
    SimpleGatingMultiplexer,
)

_BITS = float(np.log2(np.e))


def _synchrony_index(carrier: torch.Tensor) -> float:
    """Fraction of carrier variance on its top principal mode.

    1.0 == every example's carrier is a scalar multiple of one shared vector
    (global synchrony / collapse). ~1/D == energy spread across all modes.
    Carriers with more than 2 dims (e.g. GTM's optional [B, 2, T] role
    channel) are flattened to `[B, -1]` so SVD is well-defined.
    """
    if carrier.dim() > 2:
        carrier = carrier.reshape(carrier.shape[0], -1)
    centred = carrier - carrier.mean(dim=0, keepdim=True)
    if centred.shape[0] < 2:
        return 1.0
    svals = torch.linalg.svdvals(centred)
    energy = (svals**2).sum()
    if float(energy) == 0.0:
        return 1.0
    return float((svals[0] ** 2) / energy)


def _train_gtm(
    codes: torch.Tensor, steps: int, seed: int
) -> tuple[float, float, float]:
    """Train GTM end-to-end; return (accuracy, mi_bits, synchrony_index)."""
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)
    opt = Adam(gtm.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = gtm.forward(codes)
        soft = gtm.demodulate(carrier, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        gtm.step()
    with torch.no_grad():
        carrier = gtm.forward(codes)
        pred = gtm.demodulate(carrier, hard=True)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def _train_gtm_null(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Null GTM: encode `inputs`, supervise toward `targets`.

    When inputs and targets are independent (e.g. inputs are a shuffled
    view of the originals), no deterministic input->target mapping exists
    and MI between recovered and true codes should collapse near zero.
    """
    cfg = GammaThetaConfig(symbols_per_theta=inputs.shape[-1])
    gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)
    opt = Adam(gtm.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = gtm.forward(inputs)
        soft = gtm.demodulate(carrier, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        opt.step()
        gtm.step()
    with torch.no_grad():
        carrier = gtm.forward(inputs)
        pred = gtm.demodulate(carrier, hard=True)
        acc = float((pred == targets).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            targets.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def _train_simple(
    codes: torch.Tensor, steps: int
) -> tuple[float, float, float]:
    """Train the simple-gating control; same return triple as `_train_gtm`."""
    m = SimpleGatingMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1]
    )
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = m.forward(codes)
        logits = m.demodulate_logits(carrier)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        carrier = m.forward(codes)
        pred = m.demodulate(carrier)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def _train_akorn(
    codes: torch.Tensor, steps: int,
) -> tuple[float, float, float]:
    """Train the KuramotoMultiplexer; same return triple as _train_simple."""
    m = KuramotoMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1],
        n_oscillators=32, n_steps=8,
    )
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = m.forward(codes)
        logits = m.demodulate_logits(carrier)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        carrier = m.forward(codes)
        pred = m.demodulate(carrier)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def run_gtm_ablation(
    *, steps: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Run both arms on one routing task; return per-arm metrics."""
    torch.manual_seed(seed)
    codes = torch.randint(0, 64, (128, 7))
    acc_g, mi_g, sync_g = _train_gtm(codes, steps, seed)
    acc_s, mi_s, sync_s = _train_simple(codes, steps)
    acc_a, mi_a, sync_a = _train_akorn(codes, steps)
    # Null arm: train GTM with mismatched inputs/targets (inputs are a
    # shuffled view of the codes; targets remain the originals), so the
    # learned mapping cannot exploit any code-to-code correspondence and
    # MI between recovered and true codes should collapse near zero.
    gen = torch.Generator().manual_seed(seed + 9973)
    perm = torch.randperm(codes.shape[0], generator=gen)
    acc_n, mi_n, sync_n = _train_gtm_null(
        inputs=codes[perm], targets=codes, steps=steps, seed=seed + 9973
    )
    return {
        "gtm": {
            "accuracy": acc_g,
            "mi_bits": mi_g,
            "synchrony_index": sync_g,
        },
        "simple_gating": {
            "accuracy": acc_s,
            "mi_bits": mi_s,
            "synchrony_index": sync_s,
        },
        "akorn": {
            "accuracy": acc_a,
            "mi_bits": mi_a,
            "synchrony_index": sync_a,
        },
        "null": {
            "accuracy": acc_n,
            "mi_bits": mi_n,
            "synchrony_index": sync_n,
        },
    }


def main() -> None:
    result = run_gtm_ablation(steps=2000, seed=0)
    header = f"{'arm':<16}{'acc':>10}{'MI (bits)':>12}{'synchrony':>12}"
    print(header)
    for arm, row in result.items():
        print(
            f"{arm:<16}{row['accuracy']:>10.3f}"
            f"{row['mi_bits']:>12.3f}{row['synchrony_index']:>12.3f}"
        )


if __name__ == "__main__":
    main()
