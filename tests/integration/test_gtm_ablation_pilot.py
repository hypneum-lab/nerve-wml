import pytest

from scripts.gtm_ablation_pilot import run_gtm_ablation


@pytest.mark.slow
def test_run_gtm_ablation_reports_both_arms():
    result = run_gtm_ablation(steps=300, seed=0)
    assert {"gtm", "simple_gating", "null"} <= set(result)
    for arm, row in result.items():
        # Symbol-recovery accuracy is a fraction in [0, 1].
        assert 0.0 <= row["accuracy"] <= 1.0, arm
        # MI between recovered and true codes, non-negative bits.
        assert row["mi_bits"] >= 0.0, arm
        # Synchrony index in [0, 1]; 1.0 == full collapse.
        assert 0.0 <= row["synchrony_index"] <= 1.0, arm


@pytest.mark.slow
def test_gtm_does_not_fully_collapse():
    result = run_gtm_ablation(steps=300, seed=0)
    # If GTM band-multiplexing works, it must not collapse to global sync.
    assert result["gtm"]["synchrony_index"] < 0.95


@pytest.mark.slow
def test_null_arm_below_gtm():
    res = run_gtm_ablation(steps=300, seed=0)
    assert "null" in res
    assert res["null"]["mi_bits"] < res["gtm"]["mi_bits"] - 0.3


@pytest.mark.slow
def test_gtm_ablation_includes_akorn_arm() -> None:
    from scripts.gtm_ablation_pilot import run_gtm_ablation
    res = run_gtm_ablation(steps=200, seed=0)
    assert "akorn" in res
    for key in ("accuracy", "mi_bits", "synchrony_index"):
        assert key in res["akorn"]
