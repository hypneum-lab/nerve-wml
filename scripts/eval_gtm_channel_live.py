"""Part 2: re-run transmission THROUGH the trained GTM channel.

Loads the trained ``constellation`` (from ``train_gtm_channel.py``) into a live
``ProtocolNerve`` and measures whether codes survive the real PAC channel,
versus the untrained baseline. Two probes:

1. Live-loop fidelity (MLP+LIF, ~50 ticks): round-trip code fidelity through
   ``demodulate(forward(codes))`` inside ``listen()``. We measure two isolations:
     - full protocol (transducer + GTM): the headline ~1 % untrained number,
       which conflates the untrained transducer scramble with the channel.
     - GTM-only (identity transducer): isolates the CHANNEL's contribution.

2. Cross-substrate survival (A→B): a code emitted by substrate A, transported
   through transducer + trained channel, must arrive intact at substrate B.
   Measured trained vs untrained, with and without an identity transducer.

Usage:
    python scripts/eval_gtm_channel_live.py \
        --constellation runs/gtm_channel/constellation.pt \
        --out runs/gtm_channel/gtm_channel_live.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.protocol_nerve import ProtocolNerve
from track_p.transducer import Transducer

SEED = 0
N_TICKS = 50
N_WMLS = 2  # substrate A = 0, substrate B = 1
K = 1


def _set_identity_transducers(nerve: ProtocolNerve) -> None:
    """Force every per-edge transducer to the identity map (code -> code).

    A large positive logit on the diagonal makes argmax return the input code,
    so the transducer is a no-op and the GTM channel is the only transform on
    the code. This isolates the CHANNEL's contribution from transducer scramble.
    """
    for key in nerve._transducers:  # noqa: SLF001
        t: Transducer = nerve._transducers[key]  # type: ignore[assignment]
        with torch.no_grad():
            t.logits.zero_()
            t.logits.add_(torch.eye(t.alphabet_size) * 30.0)


def run_loop(nerve: ProtocolNerve, n_ticks: int) -> tuple[int, int]:
    """Drive a deterministic A<->B loop with per-letter round-trip matching.

    Each letter is keyed by its unique ``(timestamp, src, dst)`` (the GTM and
    transducer rewrite only ``code``, preserving these), so a delivered letter
    is matched back to exactly the code that was injected — a true per-letter
    fidelity, not a positional zip. Returns ``(matched, total_delivered)``.
    """
    edges = [
        (src, dst)
        for src in range(N_WMLS)
        for dst in range(N_WMLS)
        if nerve.routing_weight(src, dst) == 1.0
    ]
    sent_by_key: dict[tuple[float, int, int], int] = {}
    matched = 0
    total = 0
    dt = 1e-3
    for tick in range(n_ticks):
        t = tick * dt
        nerve.gamma_osc.phase = 0.0 if tick % 2 == 0 else 0.6
        for src, dst in edges:
            code = (tick * 7 + src * 13) % 64
            sent_by_key[(t, src, dst)] = code
            nerve.send(
                Neuroletter(
                    code=code,
                    role=Role.PREDICTION,
                    phase=Phase.GAMMA,
                    src=src,
                    dst=dst,
                    timestamp=t,
                )
            )
        for wml_id in range(N_WMLS):
            for r in nerve.listen(wml_id=wml_id):
                key = (r.timestamp, r.src, r.dst)
                if key in sent_by_key:
                    total += 1
                    if sent_by_key[key] == r.code:
                        matched += 1
        nerve.tick(dt)
    return matched, total


def live_fidelity(
    *,
    constellation: dict | None,
    use_transducer: bool,
    identity_transducer: bool,
) -> tuple[float, int, int]:
    nerve = ProtocolNerve(
        n_wmls=N_WMLS,
        k=K,
        seed=SEED,
        use_transducer=use_transducer,
        use_gtm=True,
        gtm_seed=42,
    )
    assert nerve._gtm is not None  # use_gtm=True guarantees this  # noqa: SLF001
    if constellation is not None:
        nerve._gtm.load_state_dict(constellation)  # noqa: SLF001
    if use_transducer and identity_transducer:
        _set_identity_transducers(nerve)
    matched, total = run_loop(nerve, N_TICKS)
    return (matched / total if total else float("nan"), matched, total)


def cross_substrate_survival(
    *, constellation: dict | None, identity_transducer: bool
) -> dict:
    """End-to-end A->B code survival over the transducer + GTM channel.

    Sends a sweep of all 64 codes from A to B and measures how many arrive
    intact at B's listen. With ``identity_transducer`` the transducer is a
    no-op so the result isolates the channel.
    """
    nerve = ProtocolNerve(
        n_wmls=N_WMLS,
        k=K,
        seed=SEED,
        use_transducer=True,
        use_gtm=True,
        gtm_seed=42,
    )
    assert nerve._gtm is not None  # use_gtm=True guarantees this  # noqa: SLF001
    if constellation is not None:
        nerve._gtm.load_state_dict(constellation)  # noqa: SLF001
    if identity_transducer:
        _set_identity_transducers(nerve)

    # Find an active A->B edge (A=0, B=1). If none, try B as the receiver.
    edges = [
        (src, dst)
        for src in range(N_WMLS)
        for dst in range(N_WMLS)
        if src != dst and nerve.routing_weight(src, dst) == 1.0
    ]
    if not edges:
        return {"survival": float("nan"), "matched": 0, "n": 0, "edge": None}
    src, dst = edges[0]

    nerve.gamma_osc.phase = 0.0  # γ active so listen delivers
    matched = 0
    total = 0
    for code in range(64):
        nerve.send(
            Neuroletter(
                code=code,
                role=Role.PREDICTION,
                phase=Phase.GAMMA,
                src=src,
                dst=dst,
                timestamp=0.0,
            )
        )
        received = nerve.listen(wml_id=dst)
        for r in received:
            total += 1
            if r.code == code:
                matched += 1
    return {
        "survival": matched / total if total else float("nan"),
        "matched": matched,
        "n": total,
        "edge": [src, dst],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--constellation", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/gtm_channel/gtm_channel_live.json")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    trained = torch.load(args.constellation, weights_only=True)

    out: dict = {"constellation_path": args.constellation}

    # --- 1. Live-loop fidelity ---
    out["live_loop"] = {}
    for label, untr, ident in [
        ("full_protocol_transducer_plus_gtm", True, False),
        ("gtm_only_identity_transducer", True, True),
    ]:
        fid_u, m_u, n_u = live_fidelity(
            constellation=None, use_transducer=untr, identity_transducer=ident
        )
        fid_t, m_t, n_t = live_fidelity(
            constellation=trained, use_transducer=untr, identity_transducer=ident
        )
        out["live_loop"][label] = {
            "untrained": {"fidelity": fid_u, "matched": m_u, "n": n_u},
            "trained": {"fidelity": fid_t, "matched": m_t, "n": n_t},
        }

    # --- 2. Cross-substrate survival A->B ---
    out["cross_substrate"] = {}
    for label, ident in [
        ("transducer_plus_gtm", False),
        ("gtm_only_identity_transducer", True),
    ]:
        surv_u = cross_substrate_survival(
            constellation=None, identity_transducer=ident
        )
        surv_t = cross_substrate_survival(
            constellation=trained, identity_transducer=ident
        )
        out["cross_substrate"][label] = {"untrained": surv_u, "trained": surv_t}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("=== Part 2: live transmission through trained GTM channel ===\n")
    print("Live-loop round-trip code fidelity:")
    for label, d in out["live_loop"].items():
        print(
            f"  {label}: untrained={d['untrained']['fidelity']:.3f}"
            f"  trained={d['trained']['fidelity']:.3f}"
        )
    print("\nCross-substrate A->B survival (64-code sweep):")
    for label, d in out["cross_substrate"].items():
        print(
            f"  {label}: untrained={d['untrained']['survival']:.3f}"
            f"  trained={d['trained']['survival']:.3f}"
        )
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
