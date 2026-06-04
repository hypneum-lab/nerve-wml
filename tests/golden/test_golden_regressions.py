"""Golden regression tests — verify NPZ artefacts match code output.

L4 contract: every CI run must produce numerically equivalent outputs
from the frozen simulation state. These tests reload
cycle_trace_4wmls_seed0.npz and transducers_merged.npz, comparing
against current code's output.

If a test fails, DO NOT casually regenerate the NPZ — investigate whether:
  1. freeze_golden.py::_emit_cycle was modified (check git diff)
  2. SimNerveAdapter.__init__ seeding has drifted
  3. RNG state alignment (torch.manual_seed timing relative to nerve construction)

If change is spec-sanctioned, re-run scripts/freeze_golden.py and commit new NPZs.

Tolerance policy
----------------
``test_cycle_trace_bit_stable`` is bit-exact (integer codes, no
platform variance possible).

``test_transducer_logits_bit_stable`` was originally written with
``rtol=atol=1e-14`` (effectively bit-exact). That tolerance hit the
Linux/Py3.12 CI runner where torch's CPU randn output differs in the
last few mantissa bits from macOS/Py3.14 — a known cross-platform RNG
quirk that does not violate the golden contract (the *distribution*
identity holds, only the bit-pattern of the initial sample drifts).
N3 Task 9 (2026-05-10) loosened to ``rtol=atol=1e-6`` which still
catches any genuine algorithmic regression while tolerating the
documented platform-RNG drift. See test docstring for details.
"""
from __future__ import annotations

import numpy as np
import torch

from bridge.sim_nerve_adapter import SimNerveAdapter
from scripts.freeze_golden import _emit_cycle


def test_cycle_trace_bit_stable() -> None:
    """Cycle trace codes must match frozen NPZ exactly (np.int32 arrays)."""
    torch.manual_seed(0)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    expected = np.load("tests/golden/cycle_trace_4wmls_seed0.npz")["codes"]
    actual = _emit_cycle(nerve, n_cycles=1000)
    np.testing.assert_array_equal(actual, expected)


def test_transducer_logits_bit_stable() -> None:
    """Transducer logits must match frozen NPZ within numerical tolerance.

    Tolerance is ``rtol=atol=1e-6``, not bit-exact. Rationale: the NPZ
    was frozen on macOS/Py3.14 with one torch CPU RNG implementation;
    the Linux/Py3.12 CI runner uses a different torch build whose
    ``torch.randn`` last-mantissa-bits diverge slightly. The 1e-6
    tolerance is tight enough to catch any real regression in
    SimNerveAdapter init (a real regression would shift logits by O(1)
    or change shapes) but loose enough to tolerate this documented
    platform RNG drift. The integer-code golden test above remains
    bit-exact and would catch any genuine determinism break.
    """
    torch.manual_seed(0)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    expected = np.load("tests/golden/transducers_merged.npz")

    for key, transducer in nerve._transducers.items():
        assert key in expected.files, f"unexpected transducer key {key}"
        actual_logits = transducer.logits.detach().cpu().numpy()
        expected_logits = expected[key]
        np.testing.assert_allclose(
            actual_logits, expected_logits, rtol=1e-6, atol=1e-6
        )
