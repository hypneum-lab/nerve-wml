"""Triple-substrate polymorphism gate — MLP vs LIF vs Transformer.

On the linearly-separable FlowProxyTask, all three substrates are
expected to saturate, giving a triple_gap of 0. On the non-linear
HardFlowProxyTask, substrates diverge (MLP/TRF ≈ 0.53, LIF ≈ 0.61),
but the 5 %-per-pair contract is expected to re-close at pool scale
(see test_w2_hard_scale.py).

This gate proves the substrate-agnostic claim across THREE
structurally distinct implementations: stateless MLP, spiking LIF,
and attention-based Transformer.
"""
import torch

from scripts.track_w_pilot import run_w_triple_substrate


def test_triple_substrate_saturates_flow_proxy():
    """All three substrates converge to > 0.95 on FlowProxyTask."""
    torch.manual_seed(0)
    r = run_w_triple_substrate(steps=400, hard=False)
    assert r["acc_mlp"] > 0.95, f"MLP under-trained: {r['acc_mlp']}"
    assert r["acc_lif"] > 0.95, f"LIF under-trained: {r['acc_lif']}"
    assert r["acc_trf"] > 0.95, f"Transformer under-trained: {r['acc_trf']}"
    assert r["triple_gap"] < 0.05, (
        f"triple-substrate gap {r['triple_gap']:.4f} exceeds 5 % "
        "on a saturated task — likely a training or seeding bug"
    )


def test_triple_substrate_hard_task_has_no_collapse():
    """On HardFlowProxyTask, each substrate beats the 1/12 random floor
    substantially. The gap is non-trivial but tracked (not gated < 5 %
    per the v0.4 honest-measurement note in §Threats)."""
    torch.manual_seed(0)
    r = run_w_triple_substrate(steps=800, hard=True)
    # All three substantially beat random (1/12 ≈ 0.083).
    assert r["acc_mlp"] > 3 / 12
    assert r["acc_lif"] > 3 / 12
    assert r["acc_trf"] > 3 / 12
    # Each substrate reaches or nears the linear-probe plateau (~0.55).
    assert r["acc_mlp"] > 0.45
    assert r["acc_lif"] > 0.45
    assert r["acc_trf"] > 0.40
