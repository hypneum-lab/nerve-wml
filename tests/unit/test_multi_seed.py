"""Unit tests for the multi-seed runner wrapper."""
from __future__ import annotations

import pytest

from scripts.multi_seed import run_multi_seed, summarize


def _mock_runner(*, seed: int) -> dict[str, dict[str, float]]:
    # Deterministic linear-in-seed values so we can verify aggregation.
    return {
        "learned":   {"mi_bits": 1.0 + 0.1 * seed, "acc": 0.9},
        "procrustes": {"mi_bits": 0.5 + 0.05 * seed, "acc": 0.7},
    }


def test_shape_preserved() -> None:
    agg = run_multi_seed(_mock_runner, seeds=(0, 1, 2))
    assert set(agg) == {"learned", "procrustes"}
    assert set(agg["learned"]) == {"mi_bits", "acc"}


def test_values_list_and_stats() -> None:
    agg = run_multi_seed(_mock_runner, seeds=(0, 1, 2))
    leaf = agg["learned"]["mi_bits"]
    assert leaf["values"] == [1.0, 1.1, 1.2]
    assert leaf["mean"] == pytest.approx(1.1)
    assert leaf["std"] == pytest.approx(0.0816496580927726, rel=1e-3)


def test_single_seed_zero_std() -> None:
    agg = run_multi_seed(_mock_runner, seeds=(7,))
    assert agg["learned"]["mi_bits"]["std"] == 0.0


def test_kwargs_forwarded() -> None:
    def runner(*, seed: int, scale: float) -> dict[str, dict[str, float]]:
        return {"a": {"x": scale * seed}}

    agg = run_multi_seed(runner, seeds=(1, 2), scale=10.0)
    assert agg["a"]["x"]["values"] == [10.0, 20.0]


def test_summarize_returns_string() -> None:
    agg = run_multi_seed(_mock_runner, seeds=(0, 1))
    out = summarize(agg)
    assert "learned" in out
    assert "mi_bits" in out
    assert isinstance(out, str)
