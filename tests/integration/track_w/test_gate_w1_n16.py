import torch

from scripts.track_w_pilot import run_w1_n16


def test_w1_n16_converges():
    """16-WML all-MLP pool reaches accuracy > 0.6 on FlowProxyTask."""
    torch.manual_seed(0)
    accuracy = run_w1_n16(steps=400)
    assert accuracy > 0.6, f"W1-N16 accuracy {accuracy:.3f} below 0.6 threshold"
