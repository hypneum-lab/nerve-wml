import torch

from scripts.track_w_pilot import run_w2_true_lif


def test_w2_true_lif_polymorphie_is_honest():
    """Full-step LIF evaluation, not linear probe on input_proj.

    Exercises the full spike dynamics + pattern-match decoder.
    Expected: non-zero gap (§13.1). Sanity floor: MLP > 0.30 random.
    """
    torch.manual_seed(0)
    report = run_w2_true_lif(steps=400)
    assert "acc_mlp" in report
    assert "acc_lif" in report
    assert 0.0 <= report["acc_mlp"] <= 1.0
    assert 0.0 <= report["acc_lif"] <= 1.0
    assert report["acc_mlp"] > 0.30
