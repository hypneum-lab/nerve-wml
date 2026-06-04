"""Unit tests for BioFieldWML — Phase 1 wake/sleep scheduler (Bellitto WSCL).

TDD RED → GREEN cycle per task instruction.

Convention under test:
  A call is a SLEEP call if _call_count % _sleep_every == 0 after increment.
  All other calls are WAKE calls.

  With _sleep_every = N, sleep falls on calls N, 2N, 3N, ...
  (1-indexed: the Nth call within each cycle of N calls).

Invariants verified:
  N-1: sleep calls emit nothing (silence is legitimate — no neuroletters
       delivered to the nerve).
  W-1: step() does not mutate other WMLs — all output via nerve.send().
"""
from __future__ import annotations

import pytest

from track_w.bio_field_wml import BioFieldWML
from track_w.mock_nerve import MockNerve

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_nerve(n_wmls: int = 2) -> MockNerve:
    """Minimal nerve: 2 WMLs, fully connected (k=n_wmls-1)."""
    return MockNerve(n_wmls=n_wmls, k=n_wmls - 1, seed=0)


# ---------------------------------------------------------------------------
# Test 1 — wake call emits at least one neuroletter (N-1 does not forbid this)
# ---------------------------------------------------------------------------

def test_wake_call_emits_neuroletter() -> None:
    """First call (call_count=1) with _sleep_every=5 is a wake call.

    A wake call MUST emit at least one Neuroletter on the nerve so that
    the substrate is observable by downstream WMLs.  This validates that
    the placeholder dynamics produce non-empty output.
    """
    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(id=0, sleep_every=5, seed=0)

    wml.step(nerve, t=0.0)

    # Collect all letters sent to WML 1 (the only possible destination).
    received = nerve.listen(wml_id=1)
    assert len(received) >= 1, (
        "Wake call (call_count=1, sleep_every=5) must emit at least one "
        "neuroletter; got empty list."
    )


# ---------------------------------------------------------------------------
# Test 2 — sleep call is silent (N-1 invariant)
# ---------------------------------------------------------------------------

def test_sleep_call_is_silent() -> None:
    """The _sleep_every-th call is the sleep call and must emit nothing.

    With sleep_every=3, the 3rd call (call_count % 3 == 0) is a sleep
    call.  N-1 states that len(listen(wml_id)) == 0 is valid; on a sleep
    call it must be *exactly* 0 — BioFieldWML must not emit.
    """
    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(id=0, sleep_every=3, seed=0)

    # Calls 1 and 2 are wake calls — drain their output so the queue is clean.
    wml.step(nerve, t=0.0)
    _ = nerve.listen(wml_id=1)  # drain
    wml.step(nerve, t=1.0)
    _ = nerve.listen(wml_id=1)  # drain

    # Call 3 is the sleep call (3 % 3 == 0).
    wml.step(nerve, t=2.0)
    received = nerve.listen(wml_id=1)
    assert received == [], (
        "Sleep call (call_count=3, sleep_every=3) must not emit any "
        "neuroletters (N-1: silence is legitimate on consolidation steps)."
    )


# ---------------------------------------------------------------------------
# Test 3 — scheduler alternates correctly over 8 calls
# ---------------------------------------------------------------------------

def test_scheduler_alternates_correctly() -> None:
    """With sleep_every=4, calls 4 and 8 are sleep; all others are wake.

    Expected pattern over 8 calls (call_count after increment):
      1 → wake   (1 % 4 = 1)
      2 → wake   (2 % 4 = 2)
      3 → wake   (3 % 4 = 3)
      4 → sleep  (4 % 4 = 0)
      5 → wake   (5 % 4 = 1)
      6 → wake   (6 % 4 = 2)
      7 → wake   (7 % 4 = 3)
      8 → sleep  (8 % 4 = 0)

    Convention: a call is a sleep call iff _call_count % _sleep_every == 0
    after the counter has been incremented at the start of step().
    """
    nerve = _make_nerve(n_wmls=2)
    wml = BioFieldWML(id=0, sleep_every=4, seed=0)

    wake_calls  = [1, 2, 3, 5, 6, 7]
    sleep_calls = [4, 8]

    emitted: dict[int, int] = {}
    for call_idx in range(1, 9):
        wml.step(nerve, t=float(call_idx))
        letters = nerve.listen(wml_id=1)
        emitted[call_idx] = len(letters)

    for call_idx in wake_calls:
        assert emitted[call_idx] >= 1, (
            f"Call {call_idx} should be a wake call (emits ≥1 neuroletter) "
            f"but got {emitted[call_idx]}."
        )

    for call_idx in sleep_calls:
        assert emitted[call_idx] == 0, (
            f"Call {call_idx} should be a sleep call (emits 0 neuroletters, "
            f"N-1) but got {emitted[call_idx]}."
        )


# ---------------------------------------------------------------------------
# Test — sleep_every must be >= 1 (issue #21 Phase 1 hardening)
# ---------------------------------------------------------------------------

def test_sleep_every_must_be_positive() -> None:
    """Constructor rejects sleep_every < 1.

    The wake/sleep cadence is defined via _call_count % _sleep_every.
    sleep_every == 0 would raise ZeroDivisionError inside step(), and
    negative values collapse the phase classification.  Guard at __init__
    so the error surfaces at construction (fail-fast), not on first step.
    """
    with pytest.raises(ValueError, match="sleep_every must be >= 1"):
        BioFieldWML(id=0, sleep_every=0, seed=0)

    with pytest.raises(ValueError, match="sleep_every must be >= 1"):
        BioFieldWML(id=0, sleep_every=-1, seed=0)
