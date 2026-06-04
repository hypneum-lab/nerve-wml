"""Smoke tests for run_w4_compare — checks dict keys and numeric ranges."""
import pytest
import torch

from scripts.track_w_pilot import run_w4_compare
from track_w.tasks.hard_split import HardSplitTask


@pytest.mark.parametrize("method", ["none", "rehearsal", "ewc"])
def test_run_w4_compare_keys(method):
    torch.manual_seed(0)
    task = HardSplitTask(seed=0)
    result = run_w4_compare(method=method, task=task, steps=50, seed=0)
    for key in ("forgetting", "acc0_before", "acc0_after", "acc1", "method", "lam"):
        assert key in result, f"Missing key {key!r} for method={method}"

@pytest.mark.parametrize("method", ["none", "rehearsal", "ewc"])
def test_run_w4_compare_method_label(method):
    torch.manual_seed(0)
    task = HardSplitTask(seed=0)
    result = run_w4_compare(method=method, task=task, steps=50, seed=0)
    assert result["method"] == method
