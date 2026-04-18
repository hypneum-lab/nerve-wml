"""Golden regression tests — verify NPZ artefacts match code output (bit-stable).

L4 contract: every CI run must produce identical numerical outputs from the
frozen simulation state. These tests reload cycle_trace_4wmls_seed0.npz and
transducers_merged.npz, comparing against current code's output.

If a test fails, DO NOT casually regenerate the NPZ — investigate whether:
  1. freeze_golden.py::_emit_cycle was modified (check git diff)
  2. SimNerveAdapter.__init__ seeding has drifted
  3. RNG state alignment (torch.manual_seed timing relative to nerve construction)

If change is spec-sanctioned, re-run scripts/freeze_golden.py and commit new NPZs.
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
    """Transducer logits must match frozen NPZ within float64 tolerance."""
    torch.manual_seed(0)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)
    expected = np.load("tests/golden/transducers_merged.npz")

    for key, transducer in nerve._transducers.items():
        assert key in expected.files, f"unexpected transducer key {key}"
        actual_logits = transducer.logits.detach().cpu().numpy()
        expected_logits = expected[key]
        np.testing.assert_allclose(
            actual_logits, expected_logits, rtol=1e-14, atol=1e-14
        )
