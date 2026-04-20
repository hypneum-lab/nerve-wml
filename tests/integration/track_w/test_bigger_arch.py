"""Bigger-architecture scaling tests (v1.2 track 2)."""
import pytest

from scripts.run_bigger_arch import (
    run_hard_bigger_arch_multiseed,
    run_mi_bigger_arch,
)


@pytest.mark.slow
def test_bigger_arch_gap_is_measurable_and_bounded():
    """Honest finding: at d_hidden=128 the substrate asymmetry is
    LARGER, not smaller, than at d_hidden=16 — LIF's spike
    expressivity amplifies with more neurons on XOR-style tasks.
    This refutes the naive extrapolation "bigger arch = tighter
    plateau"; arch scale and pool scale are orthogonal dimensions.

    Bounds: gap is non-trivial (> 10 %, not within seed variance)
    and bounded (< 50 %, both substrates still beat random).
    Direction is LIF > MLP, consistent with v1.1's 15/15 finding.
    """
    r = run_hard_bigger_arch_multiseed(seeds=[0, 1, 2], steps=300)
    assert r["median_gap"] > 0.10, (
        f"expected amplification at bigger arch, "
        f"got median gap {r['median_gap']:.3f}"
    )
    assert r["max_gap"] < 0.50, (
        f"gap {r['max_gap']:.3f} exceeds 50 % — substrate collapse"
    )


@pytest.mark.slow
def test_bigger_arch_mi_holds():
    """MI/H at d_hidden=128 stays > 50 % — Claim B survives scale."""
    results = run_mi_bigger_arch(seeds=[0, 1, 2], steps=300, batch=1024)
    ratios = [r["mi_over_h"] for r in results]
    assert sum(ratios) / len(ratios) > 0.50
