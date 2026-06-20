"""Unit tests for track_p.protocol_nerve.ProtocolNerve.

Invariants tested:
- Parity: ProtocolNerve(use_transducer=False, use_gtm=False) is no-op vs SimNerve.
- Transducer path: codes rewritten per-edge; N-3 holds; routing drops inactive edges.
- GTM channel: codes from listen() are valid [0,64); round-trip runs without error.
- N-1..N-5 invariants hold for ProtocolNerve.
"""
from __future__ import annotations

import pytest

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.protocol_nerve import ProtocolNerve
from track_p.sim_nerve import SimNerve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _letter(
    src: int,
    dst: int,
    role: Role = Role.PREDICTION,
    phase: Phase = Phase.GAMMA,
    code: int = 5,
    t: float = 0.0,
) -> Neuroletter:
    return Neuroletter(code=code, role=role, phase=phase, src=src, dst=dst, timestamp=t)


def _active_edge(nerve: ProtocolNerve) -> tuple[int, int]:
    """Return the first active (src, dst) edge, or raise if none found."""
    n = nerve.n_wmls
    for src in range(n):
        for dst in range(n):
            if nerve.routing_weight(src, dst) == 1.0:
                return src, dst
    raise RuntimeError("No active edge found in the nerve topology.")


# ---------------------------------------------------------------------------
# Parity: default flags are a no-op vs SimNerve
# ---------------------------------------------------------------------------


def test_parity_default_no_transducer_no_gtm():
    """ProtocolNerve(use_transducer=False, use_gtm=False) matches SimNerve exactly."""
    # Use the same seed so topology is identical.
    seed = 7
    plain = SimNerve(n_wmls=4, k=2, seed=seed)
    proto = ProtocolNerve(n_wmls=4, k=2, seed=seed)

    # Verify routing matrices are identical.
    for src in range(4):
        for dst in range(4):
            assert plain.routing_weight(src, dst) == proto.routing_weight(src, dst), (
                f"Routing mismatch at ({src}, {dst})"
            )

    # Find an active edge.
    src, dst = _active_edge(proto)

    # Send the same letter to both nerves — both in γ-active phase at t=0.
    letter = _letter(src, dst, code=17)
    plain.send(letter)
    proto.send(letter)

    delivered_plain = plain.listen(wml_id=dst)
    delivered_proto = proto.listen(wml_id=dst)

    assert len(delivered_proto) == len(delivered_plain)
    for p, q in zip(delivered_proto, delivered_plain, strict=True):
        assert p.code == q.code
        assert p.role == q.role
        assert p.phase == q.phase
        assert p.src == q.src
        assert p.dst == q.dst


def test_parity_held_letters_same_as_sim_nerve():
    """θ letters are held at t=0 by ProtocolNerve the same way as SimNerve."""
    seed = 3
    plain = SimNerve(n_wmls=4, k=2, seed=seed)
    proto = ProtocolNerve(n_wmls=4, k=2, seed=seed)

    src, dst = _active_edge(proto)
    # Also need an edge for ε — use same src/dst if role permits (N-3 allows it for ERROR/THETA).
    theta_letter = _letter(src, dst, role=Role.ERROR, phase=Phase.THETA, code=9)

    plain.send(theta_letter)
    proto.send(theta_letter)

    # At t=0: γ active, θ inactive → both hold the ERROR letter.
    delivered_plain = plain.listen(wml_id=dst)
    delivered_proto = proto.listen(wml_id=dst)
    assert len(delivered_plain) == 0
    assert len(delivered_proto) == 0


# ---------------------------------------------------------------------------
# Transducer path
# ---------------------------------------------------------------------------


def test_transducer_rewrites_code():
    """With use_transducer=True the received code may differ from the sent code."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=1, use_transducer=True)
    src, dst = _active_edge(nerve)

    # Activate γ so the letter is immediately deliverable.
    nerve.gamma_osc.phase = 0.0  # γ active in [0, 0.5)

    original_code = 5
    nerve.send(_letter(src, dst, code=original_code))
    delivered = nerve.listen(wml_id=dst)
    assert len(delivered) == 1
    # The transducer rewrites code; the rewritten value must be in [0, 64).
    assert 0 <= delivered[0].code < 64


def test_transducer_preserves_role_phase_src_dst():
    """Transducer only touches code; N-3 metadata survives."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=2, use_transducer=True)
    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    letter = _letter(src, dst, role=Role.PREDICTION, phase=Phase.GAMMA, code=33)
    nerve.send(letter)
    received = nerve.listen(wml_id=dst)
    assert len(received) == 1
    r = received[0]
    assert r.role is Role.PREDICTION
    assert r.phase is Phase.GAMMA
    assert r.src == src
    assert r.dst == dst
    # N-3 must hold for the delivered letter.
    assert_n3_role_phase_consistent(r)


def test_transducer_n3_still_holds():
    """N-3 is asserted on send() even with transducer enabled."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=5, use_transducer=True)
    src, dst = _active_edge(nerve)
    # Craft a role/phase mismatch — must raise AssertionError.
    bad_letter = Neuroletter(
        code=1, role=Role.PREDICTION, phase=Phase.THETA, src=src, dst=dst, timestamp=0.0
    )
    with pytest.raises(AssertionError):
        nerve.send(bad_letter)


def test_transducer_drops_inactive_edge():
    """Routing still drops letters on inactive edges even with transducer on."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=6, use_transducer=True)
    # Find an INactive edge.
    n = nerve.n_wmls
    inactive: tuple[int, int] | None = None
    for src in range(n):
        for dst in range(n):
            if nerve.routing_weight(src, dst) == 0.0:
                inactive = (src, dst)
                break
        if inactive:
            break
    if inactive is None:
        pytest.skip("No inactive edge in this topology (all edges active).")
    src, dst = inactive
    nerve.gamma_osc.phase = 0.0
    nerve.send(_letter(src, dst, code=7))
    # Nothing should arrive.
    assert nerve.listen(wml_id=dst) == []


# ---------------------------------------------------------------------------
# GTM channel
# ---------------------------------------------------------------------------


def test_gtm_round_trip_valid_codes():
    """GTM round-trip returns codes in [0, 64) without raising."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0, use_gtm=True, gtm_seed=42)
    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    # Send several letters.
    codes_sent = [5, 17, 42, 3, 63]
    for i, code in enumerate(codes_sent):
        nerve.send(_letter(src, dst, code=code, t=float(i)))

    delivered = nerve.listen(wml_id=dst)
    # We sent 5 letters — expect up to 5 back (some may be chunked into ≤7).
    assert len(delivered) == len(codes_sent)
    for letter in delivered:
        assert 0 <= letter.code < 64, f"GTM produced out-of-range code: {letter.code}"


def test_gtm_round_trip_code_fidelity_reported():
    """Measure and report GTM round-trip code fidelity (may be low; must be ≥ 0)."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0, use_gtm=True, gtm_seed=42)
    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    n_send = 7  # Fill exactly one GTM chunk (symbols_per_theta = 7).
    codes_sent = [(i * 9) % 64 for i in range(n_send)]
    for i, code in enumerate(codes_sent):
        nerve.send(_letter(src, dst, code=code, t=float(i)))

    delivered = nerve.listen(wml_id=dst)
    assert len(delivered) == n_send

    # Fidelity: fraction of codes that survived the round-trip unchanged.
    recv_codes = [letter.code for letter in delivered]
    matched = sum(1 for s, d in zip(codes_sent, recv_codes, strict=True) if s == d)
    fidelity = matched / n_send
    # With an untrained constellation, fidelity may be 0.
    # We only assert it's a valid fraction (no error; not negative).
    assert 0.0 <= fidelity <= 1.0
    # Report (visible with -s flag).
    print(f"\nGTM round-trip code fidelity (untrained): {fidelity:.3f} ({matched}/{n_send})")


def test_gtm_preserves_metadata():
    """GTM only rewrites code; role/phase/src/dst/timestamp are kept."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0, use_gtm=True, gtm_seed=99)
    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    ts = 3.14
    nerve.send(_letter(src, dst, role=Role.PREDICTION, phase=Phase.GAMMA, code=11, t=ts))
    delivered = nerve.listen(wml_id=dst)
    assert len(delivered) == 1
    r = delivered[0]
    assert r.role is Role.PREDICTION
    assert r.phase is Phase.GAMMA
    assert r.src == src
    assert r.dst == dst
    assert r.timestamp == ts


def test_gtm_chunking_more_than_k():
    """Letters > symbols_per_theta are chunked; all return valid codes."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0, use_gtm=True, gtm_seed=10)
    assert nerve._gtm is not None
    k = nerve._gtm.cfg.symbols_per_theta  # 7 by default

    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    # Send k+3 letters to force chunking.
    n_send = k + 3
    for i in range(n_send):
        nerve.send(_letter(src, dst, code=(i * 7) % 64, t=float(i)))

    delivered = nerve.listen(wml_id=dst)
    assert len(delivered) == n_send
    for letter in delivered:
        assert 0 <= letter.code < 64


# ---------------------------------------------------------------------------
# Combined: both transducer + GTM
# ---------------------------------------------------------------------------


def test_full_protocol_no_error():
    """Both flags enabled: letters flow end-to-end without error."""
    nerve = ProtocolNerve(
        n_wmls=4, k=2, seed=0,
        use_transducer=True,
        use_gtm=True,
        gtm_seed=7,
    )
    src, dst = _active_edge(nerve)
    nerve.gamma_osc.phase = 0.0

    for i in range(5):
        nerve.send(_letter(src, dst, code=i * 13 % 64, t=float(i)))

    delivered = nerve.listen(wml_id=dst)
    assert len(delivered) == 5
    for letter in delivered:
        assert 0 <= letter.code < 64
        assert_n3_role_phase_consistent(letter)


# ---------------------------------------------------------------------------
# parameters() coverage
# ---------------------------------------------------------------------------


def test_parameters_no_flags():
    """Default flags: only router logits in parameters()."""
    nerve = ProtocolNerve(n_wmls=3, k=1, seed=0)
    params = list(nerve.parameters())
    # Only router.logits.
    assert len(params) == 1


def test_parameters_with_transducer():
    """With use_transducer=True, transducer logits appear in parameters()."""
    nerve = ProtocolNerve(n_wmls=3, k=1, seed=0, use_transducer=True)
    params = list(nerve.parameters())
    # router.logits + transducer logits per active edge.
    n_active = sum(
        1 for src in range(3) for dst in range(3) if nerve.routing_weight(src, dst) == 1.0
    )
    assert len(params) == 1 + n_active  # 1 router + 1 Transducer.logits per edge


def test_parameters_with_gtm():
    """With use_gtm=True, GTM constellation appears in parameters()."""
    nerve = ProtocolNerve(n_wmls=3, k=1, seed=0, use_gtm=True)
    params = list(nerve.parameters())
    # router.logits + GTM constellation (the only GTM parameter).
    assert len(params) == 2


# ---------------------------------------------------------------------------
# N-1 invariant: silence is legal
# ---------------------------------------------------------------------------


def test_n1_silence_legal():
    """listen() on an empty queue returns [] without error."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0)
    result = nerve.listen(wml_id=0)
    assert result == []


# ---------------------------------------------------------------------------
# N-4 invariant: routing weights are {0, 1} after pruning
# ---------------------------------------------------------------------------


def test_n4_routing_weights_binary():
    """Routing weights are strictly 0 or 1."""
    nerve = ProtocolNerve(n_wmls=4, k=2, seed=0)
    for src in range(4):
        for dst in range(4):
            w = nerve.routing_weight(src, dst)
            assert w in (0.0, 1.0), f"N-4 violated at ({src},{dst}): {w}"
