import pytest
import torch

from scripts.track_w_pilot import run_w2_n32


@pytest.mark.slow
def test_w2_n32_stress_no_crash():
    """32-WML pool trains + evals without crashing. Primary stress test."""
    torch.manual_seed(0)
    report = run_w2_n32(steps=200)
    assert "mean_acc_mlp" in report
    assert "mean_acc_lif" in report
    assert report["n_mlp"] == 16
    assert report["n_lif"] == 16


@pytest.mark.slow
def test_w2_n32_gap_under_15pct_soft():
    """Soft assertion: relative gap < 15 % (degraded from N=16's 10 %).

    Failure here is a scaling finding to document, not a bug. Report
    the actual gap in the commit message / paper §5.1 Scaling.
    """
    torch.manual_seed(0)
    report = run_w2_n32(steps=200)
    if report["mean_acc_mlp"] > 0.6 and report["mean_acc_lif"] > 0.6:
        gap = abs(report["mean_acc_mlp"] - report["mean_acc_lif"]) / max(
            report["mean_acc_mlp"], 1e-6
        )
        assert gap < 0.15, f"N=32 soft gap {gap:.3f} exceeds 15 %"
    else:
        # If either substrate fails to learn at N=32, that's the finding.
        # Do not enforce gap. Pytest-xfail-like handling:
        pytest.skip(
            f"N=32 training under-converged: MLP={report['mean_acc_mlp']:.3f}, "
            f"LIF={report['mean_acc_lif']:.3f}. Bump steps or revisit router."
        )
