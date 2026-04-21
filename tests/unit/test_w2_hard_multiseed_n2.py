from scripts.track_w_pilot import run_w2_hard_multiseed


def test_run_w2_hard_multiseed_returns_5_seed_stats() -> None:
    result = run_w2_hard_multiseed(seeds=list(range(5)), steps=200)
    assert isinstance(result, dict)
    assert "seeds" in result
    assert result["seeds"] == [0, 1, 2, 3, 4]
    assert "gaps" in result
    assert len(result["gaps"]) == 5
    assert "median_gap" in result
    assert "p25_gap" in result
    assert "p75_gap" in result
    assert "mean_acc_mlp" in result
    assert "mean_acc_lif" in result


def test_run_w2_hard_multiseed_direction_stability() -> None:
    result = run_w2_hard_multiseed(seeds=list(range(5)), steps=200)
    lif_ge_mlp = sum(
        1 for i in range(5)
        if result["accs_lif"][i] >= result["accs_mlp"][i]
    )
    assert lif_ge_mlp >= 3, f"expected LIF>=MLP in >=3 of 5 seeds, got {lif_ge_mlp}"
