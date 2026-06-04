"""Integration: HardSplitTask baseline — no mitigation forgets >= 50 %."""
import torch

from scripts.track_w_pilot import run_w4_compare
from track_w.tasks.hard_split import HardSplitTask


def test_hard_split_none_forgets_at_least_50pct():
    """Prove the task is genuinely hard: forgetting >= 0.50 without mitigation.

    Threshold 0.50 traces to docs/superpowers/research/2026-05-30-w4-ewc-comparison.json
    key "baseline_none_forgetting_threshold".
    """
    torch.manual_seed(0)
    task = HardSplitTask(seed=0)
    result = run_w4_compare(method="none", task=task, steps=400, seed=0)
    assert result["acc0_before"] > 0.30, (
        f"Task0 baseline too low ({result['acc0_before']:.3f}), "
        "adjust HardSplitTask difficulty"
    )
    assert result["forgetting"] >= 0.50, (
        f"Expected forgetting >= 0.50 but got {result['forgetting']:.3f}. "
        "The task is not hard enough — increase HardFlowProxyTask noise or overlap."
    )
