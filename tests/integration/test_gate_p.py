import torch

from scripts.track_p_pilot import run_gate_p


def test_gate_p_all_criteria_pass():
    torch.manual_seed(0)
    report = run_gate_p()
    assert report["p1_dead_code_fraction"]  < 0.10
    assert report["p1_perplexity"]          >= 32
    assert report["p2_retention"]           > 0.95
    assert report["p3_collision_count"]     == 0
    assert report["p4_connected"]           is True
    assert (report["p4_k_per_wml"]          == 2).all()
    assert report["all_passed"]             is True



def test_p1_random_init_baseline_runs():
    """Sanity test: run_p1_random_init completes and reports a dead-code fraction.

    This is a scientific-honesty baseline, not a gate. Dead-code fraction
    is expected to be HIGHER than gate P1's MOG-init value — documented
    in spec §13 as a known limitation of fully-random VQ init.
    """
    from scripts.track_p_pilot import run_p1_random_init
    torch.manual_seed(0)
    cb, dead = run_p1_random_init(steps=2000)  # short for CI speed
    assert 0.0 <= dead <= 1.0
