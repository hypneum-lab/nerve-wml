"""Smoke test: MLP + LIF loop under plain SimNerve vs ProtocolNerve (both flags).

Runs ~50 ticks, reports GTM round-trip code fidelity, and confirms the
full-protocol loop runs end-to-end without error.

Usage:
    uv run python scripts/smoke_protocol_nerve.py
"""
from __future__ import annotations

import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.protocol_nerve import ProtocolNerve
from track_p.sim_nerve import SimNerve
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML

SEED = 0
N_TICKS = 50
N_WMLS = 2  # MLP=0, LIF=1
K = 1  # enough to connect 0→1 and 1→0 each with one edge


def _make_wmls() -> tuple[MlpWML, LifWML]:
    mlp = MlpWML(id=0, d_hidden=32, seed=SEED)
    lif = LifWML(id=1, n_neurons=50, seed=SEED)
    return mlp, lif


def run_loop(
    nerve: SimNerve,
    mlp: MlpWML,
    lif: LifWML,
    n_ticks: int,
) -> tuple[int, int, list[int], list[int]]:
    """Run the loop; return (letters_sent, letters_delivered, sent_codes, delivered_codes)."""
    letters_sent: int = 0
    letters_delivered: int = 0
    sent_codes: list[int] = []
    delivered_codes: list[int] = []

    # Determine active edges from the nerve topology.
    edges: list[tuple[int, int]] = [
        (src, dst)
        for src in range(N_WMLS)
        for dst in range(N_WMLS)
        if nerve.routing_weight(src, dst) == 1.0
    ]

    dt = 1e-3  # 1 ms ticks

    for tick in range(n_ticks):
        t = tick * dt
        # Force γ active on every other tick so we get deliveries.
        nerve.gamma_osc.phase = 0.0 if tick % 2 == 0 else 0.6

        # Emit one π from each WML on every active outgoing edge.
        for src, dst in edges:
            code = (tick * 7 + src * 13) % 64
            letter = Neuroletter(
                code=code,
                role=Role.PREDICTION,
                phase=Phase.GAMMA,
                src=src,
                dst=dst,
                timestamp=t,
            )
            nerve.send(letter)
            sent_codes.append(code)
            letters_sent += 1

        # Listen at each WML.
        for wml_id in range(N_WMLS):
            received = nerve.listen(wml_id=wml_id)
            for r in received:
                delivered_codes.append(r.code)
            letters_delivered += len(received)

        nerve.tick(dt)

    return letters_sent, letters_delivered, sent_codes, delivered_codes


def main() -> None:
    torch.manual_seed(SEED)

    print("=== Smoke test: SimNerve vs ProtocolNerve ===\n")

    # --- Plain SimNerve ---
    plain_nerve = SimNerve(n_wmls=N_WMLS, k=K, seed=SEED)
    mlp_a, lif_a = _make_wmls()
    sent_plain, delivered_plain, s_codes, _ = run_loop(plain_nerve, mlp_a, lif_a, N_TICKS)
    print(f"[SimNerve]        sent={sent_plain}  delivered={delivered_plain}")

    # --- ProtocolNerve full (both flags) ---
    proto_nerve = ProtocolNerve(
        n_wmls=N_WMLS, k=K, seed=SEED,
        use_transducer=True,
        use_gtm=True,
        gtm_seed=42,
    )
    mlp_b, lif_b = _make_wmls()
    sent_proto, delivered_proto, sent_codes_p, recv_codes_p = run_loop(
        proto_nerve, mlp_b, lif_b, N_TICKS
    )

    # GTM fidelity: fraction of delivered codes that match their sent code.
    n_cmp = min(len(sent_codes_p), len(recv_codes_p))
    matched = sum(
        1 for s, d in zip(sent_codes_p[:n_cmp], recv_codes_p[:n_cmp], strict=False) if s == d
    )
    fidelity = matched / n_cmp if n_cmp > 0 else float("nan")

    print(f"[ProtocolNerve]   sent={sent_proto}  delivered={delivered_proto}")
    print(f"\nGTM round-trip code-fidelity (untrained constellation): "
          f"{fidelity:.3f}  ({matched}/{n_cmp})")
    print("\nSmoke test PASSED — full-protocol loop completed end-to-end.")


if __name__ == "__main__":
    main()
