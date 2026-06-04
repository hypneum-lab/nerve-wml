"""Debiased HSIC and CKNNA alignment estimators.

Closes gap 3: the "91-96% shared info" claim is at risk of small-sample
upward bias. `hsic_debiased` is the unbiased HSIC of Song et al. 2012
(J. Mach. Learn. Res.), the same correction Kornblith et al. 2019 use for
debiased CKA. `cknna` is the mutual-k-NN alignment metric (Huh et al.
2024, "The Platonic Representation Hypothesis"), robust to global scaling.

Pure-numpy, no torch — mirrors `mi_estimators.py` so callers can mix the
estimators freely. All functions accept `[N, D]` float arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _linear_gram(x: np.ndarray) -> np.ndarray:
    """Linear-kernel Gram matrix `X X^T`, shape `[N, N]`."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] array, got shape {x.shape}")
    return x @ x.T


def hsic_debiased(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012, eq. 5).

    Removes the O(1/N) upward bias of the plain (biased) HSIC. With linear
    kernels this is the numerator of debiased CKA. Returns a scalar that is
    ~0 under independence (may be slightly negative) and positive under
    dependence.

    Parameters
    ----------
    x, y
        `[N, D]` arrays with the SAME number of rows `N`.
    """
    kx = _linear_gram(x)
    ky = _linear_gram(y)
    n = kx.shape[0]
    if ky.shape[0] != n:
        raise ValueError(f"row mismatch: x has {n}, y has {ky.shape[0]}")
    if n < 4:
        raise ValueError(f"debiased HSIC needs N >= 4, got {n}")
    # Zero the diagonals — the unbiased estimator excludes self-pairs.
    np.fill_diagonal(kx, 0.0)
    np.fill_diagonal(ky, 0.0)
    kx_row = kx.sum(axis=1)
    ky_row = ky.sum(axis=1)
    kx_sum = kx_row.sum()
    ky_sum = ky_row.sum()
    term_trace = float((kx * ky).sum())
    term_outer = float(kx_sum * ky_sum) / ((n - 1) * (n - 2))
    term_cross = float(kx_row @ ky_row) * 2.0 / (n - 2)
    unbiased = (
        term_trace + term_outer - term_cross
    ) / (n * (n - 3))
    return float(unbiased)


def _knn_mask(x: np.ndarray, k: int) -> np.ndarray:
    """Boolean `[N, N]` mask: True where column j is one of row i's k
    nearest neighbours by Euclidean distance (self excluded)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N, D] array, got shape {x.shape}")
    n = x.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, N-1]={n - 1}, got {k}")
    sq = (x * x).sum(axis=1)
    dist = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.fill_diagonal(dist, np.inf)  # never pick self
    nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros((n, n), dtype=bool)
    rows = np.repeat(np.arange(n), k)
    mask[rows, nn_idx.reshape(-1)] = True
    return mask


def cknna(x: np.ndarray, y: np.ndarray, *, k: int = 10) -> float:
    """Mutual k-NN alignment (CKNNA), Huh et al. 2024.

    Fraction of x's k-NN graph edges that also appear in y's k-NN graph,
    averaged over rows. Scale- and rotation-tolerant; returns 1.0 when the
    two neighbourhood structures coincide and ~k/N under independence.

    Parameters
    ----------
    x, y
        `[N, D]` arrays with the same `N`.
    k
        Neighbourhood size.
    """
    mx = _knn_mask(x, k)
    my = _knn_mask(y, k)
    if mx.shape != my.shape:
        raise ValueError(f"row mismatch: x {mx.shape}, y {my.shape}")
    intersection = np.logical_and(mx, my).sum(axis=1)
    return float(np.mean(intersection / k))


@dataclass(frozen=True)
class ScaleRobustnessRow:
    """One sample-size point of a scale-robustness sweep."""

    n: int
    hsic: float
    cknna: float


def scale_robustness_sweep(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sizes: tuple[int, ...],
    k: int = 10,
    seed: int = 0,
) -> list[ScaleRobustnessRow]:
    """Recompute HSIC and CKNNA at increasing random subsample sizes.

    PRH critiques (Huh et al. 2024 §6) report that representation alignment
    can degrade as sample size grows; this helper exposes that trend. Each
    size draws a fresh random subset (without replacement) from the same
    `[N, D]` pair.

    Parameters
    ----------
    x, y
        `[N, D]` arrays with the same `N`.
    sizes
        Subsample sizes, each <= N.
    k
        Neighbourhood size forwarded to :func:`cknna`.
    seed
        RNG seed for the subsampling.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_total = x.shape[0]
    if y.shape[0] != n_total:
        raise ValueError(f"row mismatch: x {n_total}, y {y.shape[0]}")
    rng = np.random.default_rng(seed)
    rows: list[ScaleRobustnessRow] = []
    for size in sizes:
        if not 4 <= size <= n_total:
            raise ValueError(f"size {size} out of [4, {n_total}]")
        idx = rng.choice(n_total, size=size, replace=False)
        xs, ys = x[idx], y[idx]
        rows.append(
            ScaleRobustnessRow(
                n=size,
                hsic=hsic_debiased(xs, ys),
                cknna=cknna(xs, ys, k=min(k, size - 1)),
            )
        )
    return rows
