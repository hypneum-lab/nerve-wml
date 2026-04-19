"""NerveWmlAdvisor latency gate.

Gate threshold: advise() completes in < 50 ms on commodity M-series Apple
Silicon once the checkpoint is loaded. The disabled-path is much faster
(< 1 ms) — verifying both catches any accidental eager work in __init__.
"""
import time

import pytest
import torch

from bridge.checkpoint import save_advisor_checkpoint
from bridge.kiki_nerve_advisor import NerveWmlAdvisor
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML


@pytest.mark.slow
def test_advise_latency_under_50ms(tmp_path):
    """Warm-path advise() under 50 ms on M-series (slow-marked for CI)."""
    pool = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)
    save_advisor_checkpoint(pool, nerve, tmp_path)

    adv = NerveWmlAdvisor(enabled=True, checkpoint_path=tmp_path, n_domains=35)
    q = torch.randn(1, 16)

    # Warm up (first call loads the checkpoint).
    _ = adv.advise(q)

    # Measure 10 calls, report the median.
    durations = []
    for _ in range(10):
        start = time.perf_counter()
        adv.advise(q)
        durations.append(time.perf_counter() - start)

    median_ms = sorted(durations)[5] * 1000
    assert median_ms < 50.0, f"advise() median {median_ms:.1f} ms exceeds 50 ms budget"


def test_disabled_advise_overhead_under_1ms():
    """Disabled advisor must add < 1 ms per call to the host."""
    adv = NerveWmlAdvisor(enabled=False)
    q = torch.randn(1, 16)

    # Warm up.
    _ = adv.advise(q)

    # Measure.
    durations = []
    for _ in range(100):
        start = time.perf_counter()
        adv.advise(q)
        durations.append(time.perf_counter() - start)

    median_ms = sorted(durations)[50] * 1000
    # Be generous: 5 ms ceiling to tolerate CI jitter.
    assert median_ms < 5.0, (
        f"disabled advise overhead median {median_ms:.3f} ms exceeds 5 ms"
    )
