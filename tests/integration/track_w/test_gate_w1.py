import torch

from scripts.track_w_pilot import run_w1


def test_w1_two_mlp_wmls_converge():
    torch.manual_seed(0)
    accuracy = run_w1(steps=400)
    # Gate W1: task solved, accuracy well above random (0.25 for 4 classes).
    assert accuracy > 0.6
