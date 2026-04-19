import torch

from scripts.track_p_pilot import run_p1_dead_vs_steps


def test_p1_dead_vs_steps_returns_per_variant_curves():
    """Helper returns three lists of (step, dead_frac) tuples."""
    torch.manual_seed(0)
    report = run_p1_dead_vs_steps(max_steps=2000, checkpoint_every=500)
    assert "mog_init" in report
    assert "random_no_rot" in report
    assert "random_with_rot" in report
    # 4 checkpoints (500, 1000, 1500, 2000).
    for variant in ("mog_init", "random_no_rot", "random_with_rot"):
        assert len(report[variant]) == 4
        for step, dead in report[variant]:
            assert step > 0
            assert 0.0 <= dead <= 1.0


def test_p1_dead_vs_steps_rotation_beats_no_rotation_eventually():
    """At max_steps, random_with_rot must have lower dead-fraction than random_no_rot."""
    torch.manual_seed(0)
    report = run_p1_dead_vs_steps(max_steps=2000, checkpoint_every=500)
    _, dead_no_rot = report["random_no_rot"][-1]
    _, dead_with_rot = report["random_with_rot"][-1]
    assert dead_with_rot <= dead_no_rot
