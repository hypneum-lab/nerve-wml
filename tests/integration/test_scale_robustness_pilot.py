import pytest

from scripts.scale_robustness_pilot import run_scale_robustness


@pytest.mark.slow
def test_run_scale_robustness_returns_rows_per_size():
    out = run_scale_robustness(
        sizes=(64, 128, 256), seed=0
    )
    rows = out["rows"]
    assert [r["n"] for r in rows] == [64, 128, 256]
    for r in rows:
        assert "hsic" in r and "cknna" in r
        assert 0.0 <= r["cknna"] <= 1.0


@pytest.mark.slow
def test_scale_robustness_hsic_finite():
    out = run_scale_robustness(sizes=(64, 128), seed=1)
    rows = out["rows"]
    import math

    assert all(math.isfinite(r["hsic"]) for r in rows)


@pytest.mark.slow
def test_null_rows_below_real():
    out = run_scale_robustness(sizes=(64, 128), seed=0)
    assert "rows" in out and "null_rows" in out
    real_cknna = [r["cknna"] for r in out["rows"]]
    null_cknna = [r["cknna"] for r in out["null_rows"]]
    assert max(null_cknna) < max(real_cknna)


@pytest.mark.slow
def test_run_scale_robustness_handles_large_n():
    """Validate that CKNNA remains stable and meaningful at large N (1024-2048)."""
    out = run_scale_robustness(sizes=(512, 1024, 2048), seed=0)
    assert [r["n"] for r in out["rows"]] == [512.0, 1024.0, 2048.0]

    # CKNNA bounded in [0, 1] at all sizes.
    for r in out["rows"]:
        assert 0.0 <= r["cknna"] <= 1.0

    # Real vs null still distinguishable at N=2048.
    real = out["rows"][-1]["cknna"]
    null = out["null_rows"][-1]["cknna"]
    assert real > null, (
        f"Real CKNNA at N=2048 ({real:.4f}) must exceed null "
        f"({null:.4f}) to reject PRH scale degradation critique"
    )
