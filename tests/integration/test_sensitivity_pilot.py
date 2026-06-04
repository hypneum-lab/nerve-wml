"""Integration test for the sensitivity sweep pilot."""
from __future__ import annotations

import pytest

from scripts.sensitivity_pilot import run_sensitivity_sweeps


@pytest.mark.slow
def test_sensitivity_shape_and_variation() -> None:
    out = run_sensitivity_sweeps(seeds=(0, 1, 2))
    assert set(out) == {"k_cknna", "lambda_cycle", "n_anchors", "steps"}
    for _axis_name, axis_rows in out.items():
        assert isinstance(axis_rows, list)
        assert len(axis_rows) >= 2
        for row in axis_rows:
            assert "param" in row
            assert "value" in row
            assert "mean" in row and "std" in row
    # At least one axis exhibits non-trivial mean variation.
    def spread(axis_rows: list[dict]) -> float:
        means = [r["mean"] for r in axis_rows]
        return max(means) - min(means)
    assert any(spread(rows) > 1e-4 for rows in out.values())
