"""Analytic floor of the Test (6) mutual-kNN kernel (issue #110)."""
from __future__ import annotations

import pytest
import torch

from scripts.platonic_rh_alignment import mutual_knn, mutual_knn_floor


def test_mknn_floor_formula() -> None:
    assert mutual_knn_floor(1024, 10) == pytest.approx(10 / 1023)
    assert mutual_knn_floor(2, 1) == 1.0


@pytest.mark.parametrize(("n", "k"), [(1, 1), (10, 0), (10, 10)])
def test_mknn_floor_rejects_bad_args(n: int, k: int) -> None:
    with pytest.raises(ValueError):
        mutual_knn_floor(n, k)


def test_mknn_floor_matches_iid_draws() -> None:
    """Mean over 20 i.i.d. Gaussian pairs is within 5 % of k/(n-1)."""
    n, k, d, draws = 256, 16, 16, 20
    gen = torch.Generator().manual_seed(110)
    scores = [
        mutual_knn(
            torch.randn(n, d, generator=gen),
            torch.randn(n, d, generator=gen),
            k=k,
        )
        for _ in range(draws)
    ]
    mean = sum(scores) / draws
    floor = mutual_knn_floor(n, k)
    assert abs(mean - floor) / floor < 0.05


def test_mknn_self_alignment_is_one() -> None:
    x = torch.randn(64, 16, generator=torch.Generator().manual_seed(0))
    assert mutual_knn(x, x, k=5) == pytest.approx(1.0)
