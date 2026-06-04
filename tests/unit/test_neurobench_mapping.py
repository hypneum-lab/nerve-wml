from neuromorphic.neurobench_mapping import (
    NeuroBenchResult,
    ValidationOutcome,
    map_to_neurobench,
)


def test_maps_outcome_to_streaming_classification_task():
    outcome = ValidationOutcome(
        substrate="BioWML",
        n_correct=87,
        n_total=100,
        total_synops=12_000,
        param_count=4096,
        latency_ms=12.5,
    )
    result = map_to_neurobench(outcome)
    assert isinstance(result, NeuroBenchResult)
    assert result.task == "streaming_classification"
    assert result.accuracy == 0.87


def test_footprint_metrics_are_carried_through():
    outcome = ValidationOutcome(
        substrate="MlpWML", n_correct=90, n_total=100,
        total_synops=5_000, param_count=2048, latency_ms=0.4,
    )
    result = map_to_neurobench(outcome)
    assert result.connection_sparsity is not None
    assert result.synaptic_ops == 5_000
    assert result.footprint_params == 2048


def test_zero_total_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        map_to_neurobench(ValidationOutcome(
            substrate="LifWML", n_correct=0, n_total=0,
            total_synops=0, param_count=1, latency_ms=1.0,
        ))


def test_result_serialises_to_a_flat_dict():
    outcome = ValidationOutcome(
        substrate="BioWML", n_correct=70, n_total=100,
        total_synops=9_000, param_count=1024, latency_ms=11.0,
    )
    row = map_to_neurobench(outcome).as_row()
    assert row["task"] == "streaming_classification"
    assert row["accuracy"] == 0.70
    assert row["substrate"] == "BioWML"
    assert set(row) >= {
        "task", "substrate", "accuracy", "synaptic_ops",
        "footprint_params", "latency_ms", "harness",
    }
