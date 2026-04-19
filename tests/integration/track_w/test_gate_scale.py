import pytest
import torch

from scripts.track_w_pilot import run_gate_scale


@pytest.mark.slow
def test_gate_scale_all_criteria_pass():
    """Gate-Scale: W1-N16 + W2-N16 + W4-N16 + W2-N32 all pass their thresholds."""
    torch.manual_seed(0)
    report = run_gate_scale()

    # W1-N16
    assert report["w1_n16_accuracy"] > 0.6

    # W2-N16
    assert report["w2_n16_acc_mlp"] > 0.6
    assert report["w2_n16_acc_lif"] > 0.6
    assert report["w2_n16_gap"] < 0.10

    # W4-N16
    assert report["w4_n16_forgetting"] < 0.20

    # W2-N32 (soft — just no crash required, no threshold assertion)
    assert report["w2_n32_n_mlp"] == 16
    assert report["w2_n32_n_lif"] == 16

    # Overall
    assert report["all_passed"] is True
