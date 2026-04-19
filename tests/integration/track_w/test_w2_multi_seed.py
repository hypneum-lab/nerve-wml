import torch

from scripts.track_w_pilot import run_w2_multi_seed


def test_w2_multi_seed_returns_per_seed_accuracies():
    """Multi-seed helper returns MLP and LIF accuracy lists, one per seed."""
    torch.manual_seed(0)
    report = run_w2_multi_seed(seeds=[0, 1, 2], steps=200)
    assert "acc_mlp" in report
    assert "acc_lif" in report
    assert len(report["acc_mlp"]) == 3
    assert len(report["acc_lif"]) == 3
    # Each value in [0, 1].
    for v in report["acc_mlp"] + report["acc_lif"]:
        assert 0.0 <= v <= 1.0


def test_w2_multi_seed_is_deterministic_per_seed():
    """Calling with the same seeds twice yields the same accuracies."""
    torch.manual_seed(0)
    a = run_w2_multi_seed(seeds=[0, 1], steps=100)
    torch.manual_seed(0)
    b = run_w2_multi_seed(seeds=[0, 1], steps=100)
    assert a["acc_mlp"] == b["acc_mlp"]
    assert a["acc_lif"] == b["acc_lif"]
