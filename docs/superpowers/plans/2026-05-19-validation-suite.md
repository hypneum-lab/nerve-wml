# Validation Suite Implementation Plan

For agentic workers: this plan is designed to be executed task-by-task,
optionally via the `superpowers:subagent-driven-development` skill (one
subagent per Task, fresh context, review checkpoint between tasks). Each
Task is self-contained: it lists exact files, a TDD cycle, and the exact
git commands to commit.

**Goal.** Close the three 🟠 IMPORTANT experimental-validation gaps from
the nerve-wml literature gap analysis: (1) the learned `Transducer` has no
comparison baselines; (2) the `GammaThetaMultiplexer` band-multiplexing is
not shown to beat plain gating, nor checked for synchrony collapse; (3) the
"91-96% shared info" claim uses potentially biased estimators and is not
tested for scale robustness. After this plan, every claim in those three
areas is backed by a runnable experiment and a frozen-numeric test.

**Architecture.**
- `track_p/transducer_baselines.py` — three non-learned / alternatively-learned
  src→dst code maps that share the `Transducer` interface enough to be swapped
  into the same code-translation benchmark: `ProcrustesTransducer` (orthogonal
  Procrustes via SVD), `RelativeRepTransducer` (anchor-based cosine encoding,
  zero-shot), `Vec2VecTransducer` (unsupervised GAN + cycle-consistency).
- `nerve_wml/methodology/hsic_cknna.py` — debiased HSIC estimator and a
  CKNNA / mutual-k-NN alignment estimator, plus a scale-robustness sweep
  helper. These are pure functions over numpy arrays, matching the style of
  `mi_estimators.py`.
- `scripts/transducer_baselines_pilot.py`, `scripts/gtm_ablation_pilot.py`,
  `scripts/scale_robustness_pilot.py` — experiment runners that import the
  above, run on the existing WML substrates, and print MI/H tables.
- Tests live under `tests/unit/` (L1), `tests/info_theoretic/` (L2),
  `tests/integration/` (L3). Long experiments are marked `@pytest.mark.slow`.

**Tech Stack.** Python 3.12+, PyTorch (`torch`, `torch.nn`), numpy, pytest,
ruff, mypy. Deps installed via `uv sync --all-extras`. Tests run with
`uv run pytest -m "not slow"` (fast) or `uv run pytest`. Lint
`uv run ruff check .`; types `uv run mypy nerve_core track_p track_w`.

---

## Task 1 — Debiased HSIC estimator

**Files:**
- Create: `nerve_wml/methodology/hsic_cknna.py`
- Test: `tests/info_theoretic/test_hsic_cknna.py`

Steps:

- [ ] 1.1 Write the failing test. Create
  `tests/info_theoretic/test_hsic_cknna.py`:

```python
import numpy as np

from nerve_wml.methodology.hsic_cknna import hsic_debiased


def test_hsic_debiased_zero_for_independent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 8))
    y = rng.standard_normal((400, 8))
    val = hsic_debiased(x, y)
    # Debiased HSIC is unbiased under independence: ~0, can be slightly negative.
    assert abs(val) < 0.05


def test_hsic_debiased_positive_for_dependent():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 8))
    y = x + 0.1 * rng.standard_normal((400, 8))
    assert hsic_debiased(x, y) > 0.1


def test_hsic_debiased_symmetric():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 4))
    y = rng.standard_normal((200, 4))
    a = hsic_debiased(x, y)
    b = hsic_debiased(y, x)
    assert abs(a - b) < 1e-9
```

- [ ] 1.2 Run it — expected **FAIL** (module does not exist):
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -q`
  Expected: `ModuleNotFoundError: No module named 'nerve_wml.methodology.hsic_cknna'`.

- [ ] 1.3 Minimal implementation. Create
  `nerve_wml/methodology/hsic_cknna.py` with the debiased HSIC of
  Song et al. 2012 (unbiased estimator, used by Kornblith et al. 2019 for
  debiased CKA):

```python
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
```

- [ ] 1.4 Run it — expected **PASS**:
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -q`
  Expected: `3 passed`.

- [ ] 1.5 Lint + types:
  `uv run ruff check nerve_wml/methodology/hsic_cknna.py` (expect no errors)
  `uv run mypy nerve_wml` (expect success).

- [ ] 1.6 Commit:

```bash
git add nerve_wml/methodology/hsic_cknna.py tests/info_theoretic/test_hsic_cknna.py
git commit -m "add debiased HSIC estimator to methodology"
```

---

## Task 2 — CKNNA / mutual k-NN alignment estimator

**Files:**
- Modify: `nerve_wml/methodology/hsic_cknna.py`
- Test: `tests/info_theoretic/test_hsic_cknna.py`

Steps:

- [ ] 2.1 Append the failing test to
  `tests/info_theoretic/test_hsic_cknna.py`:

```python
from nerve_wml.methodology.hsic_cknna import cknna


def test_cknna_one_for_identical():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((120, 16))
    # CKNNA is scale-invariant: a global rescale must not change the score.
    assert abs(cknna(x, 3.0 * x, k=10) - 1.0) < 1e-9


def test_cknna_low_for_independent():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((200, 16))
    y = rng.standard_normal((200, 16))
    # Random neighborhoods overlap at chance ~ k / N.
    assert cknna(x, y, k=10) < 0.2


def test_cknna_high_for_aligned():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((200, 16))
    y = x + 0.05 * rng.standard_normal((200, 16))
    assert cknna(x, y, k=10) > 0.6
```

- [ ] 2.2 Run it — expected **FAIL**:
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -k cknna -q`
  Expected: `ImportError: cannot import name 'cknna'`.

- [ ] 2.3 Append the implementation to
  `nerve_wml/methodology/hsic_cknna.py`:

```python
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
```

- [ ] 2.4 Run it — expected **PASS**:
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -q`
  Expected: `6 passed`.

- [ ] 2.5 Lint + types as in 1.5.

- [ ] 2.6 Commit:

```bash
git add nerve_wml/methodology/hsic_cknna.py tests/info_theoretic/test_hsic_cknna.py
git commit -m "add cknna mutual knn alignment estimator"
```

---

## Task 3 — Scale-robustness sweep helper

**Files:**
- Modify: `nerve_wml/methodology/hsic_cknna.py`
- Test: `tests/info_theoretic/test_hsic_cknna.py`

Steps:

- [ ] 3.1 Append the failing test:

```python
from nerve_wml.methodology.hsic_cknna import scale_robustness_sweep


def test_scale_robustness_sweep_returns_one_row_per_size():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((800, 16))
    y = x + 0.05 * rng.standard_normal((800, 16))
    rows = scale_robustness_sweep(x, y, sizes=(100, 200, 400), k=10, seed=0)
    assert [r.n for r in rows] == [100, 200, 400]
    # Aligned data: cknna stays high at every sample size.
    assert all(r.cknna > 0.5 for r in rows)
    # HSIC is finite and non-negative on aligned data.
    assert all(np.isfinite(r.hsic) and r.hsic >= 0.0 for r in rows)
```

- [ ] 3.2 Run it — expected **FAIL**:
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -k scale_robustness -q`
  Expected: `ImportError: cannot import name 'scale_robustness_sweep'`.

- [ ] 3.3 Append the implementation:

```python
from dataclasses import dataclass


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
```

- [ ] 3.4 Run it — expected **PASS**:
  `uv run pytest tests/info_theoretic/test_hsic_cknna.py -q`
  Expected: `7 passed`.

- [ ] 3.5 Lint + types as in 1.5.

- [ ] 3.6 Commit:

```bash
git add nerve_wml/methodology/hsic_cknna.py tests/info_theoretic/test_hsic_cknna.py
git commit -m "add scale robustness sweep helper"
```

---

## Task 4 — Orthogonal Procrustes transducer baseline

**Files:**
- Create: `track_p/transducer_baselines.py`
- Test: `tests/unit/test_transducer_baselines.py`

Steps:

- [ ] 4.1 Write the failing test. Create
  `tests/unit/test_transducer_baselines.py`:

```python
import torch

from track_p.transducer_baselines import ProcrustesTransducer


def test_procrustes_fits_and_maps_to_valid_codes():
    torch.manual_seed(0)
    src_cb = torch.randn(64, 32)
    # dst codebook is a rotation of src + small noise — recoverable.
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q + 0.01 * torch.randn(64, 32)
    t = ProcrustesTransducer(src_codebook=src_cb, dst_codebook=dst_cb)
    src_code = torch.tensor([5, 17, 42])
    dst_code = t.forward(src_code)
    assert dst_code.shape == src_code.shape
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_procrustes_recovers_known_permutation():
    torch.manual_seed(1)
    src_cb = torch.randn(64, 32)
    perm = torch.randperm(64)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = (src_cb @ q)[perm]
    t = ProcrustesTransducer(src_codebook=src_cb, dst_codebook=dst_cb)
    mapped = t.forward(torch.arange(64))
    # Procrustes should recover the rotation, hence the permutation exactly.
    assert torch.equal(mapped, perm)
```

- [ ] 4.2 Run it — expected **FAIL**:
  `uv run pytest tests/unit/test_transducer_baselines.py -q`
  Expected: `ModuleNotFoundError: No module named 'track_p.transducer_baselines'`.

- [ ] 4.3 Minimal implementation. Create `track_p/transducer_baselines.py`:

```python
"""Comparison baselines for the learned :class:`track_p.transducer.Transducer`.

Closes gap 1 of the validation suite. The learned Transducer is a free
[64x64] logits matrix; on its own it has no point of comparison. This module
provides three alternative src->dst code maps, each consuming the two WML
codebooks (`MlpWML.codebook` etc.) and exposing a `forward(src_code) -> dst_code`
that returns a `[B]` long tensor of dst indices, so all four can be plugged
into the same code-translation benchmark (`scripts/transducer_baselines_pilot.py`).

- ProcrustesTransducer  -- orthogonal Procrustes map (Maystre et al. 2025,
  arXiv:2510.13406). Closed-form SVD solution, supervised by code index.
- RelativeRepTransducer -- anchor-based cosine encoding, zero-shot, no fit
  (Moschella et al., ICLR 2023, arXiv:2209.15430).
- Vec2VecTransducer     -- unsupervised GAN + cycle-consistency translation,
  no paired data (Jha et al. 2025, arXiv:2505.12540).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class ProcrustesTransducer(nn.Module):
    """Orthogonal Procrustes src->dst code map.

    Fits the orthogonal matrix `R` minimising `||src @ R - dst||_F` over the
    paired codebook rows (pairing = shared code index), via the SVD of
    `dst^T @ src`. At inference, a src code is mapped by projecting its src
    embedding through `R` and taking the nearest dst codebook row.

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors. D must match between the two.
    """

    rotation: Tensor

    def __init__(self, src_codebook: Tensor, dst_codebook: Tensor) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        self.alphabet_size = src_codebook.shape[0]
        src = src_codebook.detach().to(torch.float64)
        dst = dst_codebook.detach().to(torch.float64)
        # Orthogonal Procrustes: R = U V^T from SVD of dst^T @ src.
        u, _, vh = torch.linalg.svd(dst.T @ src)
        rotation = (u @ vh).to(torch.float32)
        self.register_buffer("rotation", rotation)
        self.register_buffer("_dst_codebook", dst_codebook.detach().clone())

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        src_emb = self._src_lookup(src_code)  # [B, D]
        projected = src_emb @ self.rotation  # [B, D]
        dist = torch.cdist(projected, self._dst_codebook)  # [B, alphabet]
        return dist.argmin(dim=-1)

    def fit_src_codebook(self, src_codebook: Tensor) -> None:
        """Register the src codebook used for index lookup at `forward`."""
        self.register_buffer("_src_codebook", src_codebook.detach().clone())

    def _src_lookup(self, src_code: Tensor) -> Tensor:
        return self._src_codebook[src_code]
```

  Note: store `_src_codebook` in `__init__` too — add this line right after
  the `_dst_codebook` registration:

```python
        self.register_buffer("_src_codebook", src_codebook.detach().clone())
```

- [ ] 4.4 Run it — expected **PASS**:
  `uv run pytest tests/unit/test_transducer_baselines.py -q`
  Expected: `2 passed`.

- [ ] 4.5 Lint + types:
  `uv run ruff check track_p/transducer_baselines.py`
  `uv run mypy track_p` (expect success).

- [ ] 4.6 Commit:

```bash
git add track_p/transducer_baselines.py tests/unit/test_transducer_baselines.py
git commit -m "add procrustes transducer baseline"
```

---

## Task 5 — Relative-representations transducer baseline

**Files:**
- Modify: `track_p/transducer_baselines.py`
- Test: `tests/unit/test_transducer_baselines.py`

Steps:

- [ ] 5.1 Append the failing test:

```python
from track_p.transducer_baselines import RelativeRepTransducer


def test_relative_rep_zero_shot_maps_valid_codes():
    torch.manual_seed(2)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q
    t = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=16, seed=0
    )
    dst_code = t.forward(torch.tensor([3, 9, 60]))
    assert dst_code.shape == (3,)
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_relative_rep_invariant_to_rotation():
    torch.manual_seed(3)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q  # pure rotation, anchors shared by index
    t = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=32, seed=1
    )
    # Cosine-to-anchors is rotation-invariant -> identity recovered.
    assert torch.equal(t.forward(torch.arange(64)), torch.arange(64))
```

- [ ] 5.2 Run it — expected **FAIL**:
  `uv run pytest tests/unit/test_transducer_baselines.py -k relative_rep -q`
  Expected: `ImportError: cannot import name 'RelativeRepTransducer'`.

- [ ] 5.3 Append the implementation to `track_p/transducer_baselines.py`:

```python
class RelativeRepTransducer(nn.Module):
    """Anchor-based relative-representation src->dst map (zero-shot).

    Moschella et al. (ICLR 2023): encode each codebook row by its vector of
    cosine similarities to a fixed set of `n_anchors` anchor rows. Because the
    anchors are shared by code index across src and dst, the relative encoding
    is invariant to any rotation/reflection between the two latent spaces, so
    no fitting is needed. A src code maps to the dst code whose relative
    encoding is closest (cosine).

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors, same shape.
    n_anchors
        Number of anchor codes (sampled without replacement from the alphabet).
    seed
        RNG seed for anchor selection.
    """

    src_rel: Tensor
    dst_rel: Tensor

    def __init__(
        self,
        src_codebook: Tensor,
        dst_codebook: Tensor,
        *,
        n_anchors: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        alphabet = src_codebook.shape[0]
        if not 1 <= n_anchors <= alphabet:
            raise ValueError(f"n_anchors must be in [1, {alphabet}]")
        gen = torch.Generator()
        gen.manual_seed(seed)
        anchors = torch.randperm(alphabet, generator=gen)[:n_anchors]
        src = src_codebook.detach()
        dst = dst_codebook.detach()
        self.register_buffer("src_rel", self._relative(src, anchors))
        self.register_buffer("dst_rel", self._relative(dst, anchors))

    @staticmethod
    def _relative(codebook: Tensor, anchors: Tensor) -> Tensor:
        """`[alphabet, n_anchors]` cosine-similarity encoding."""
        normed = F.normalize(codebook, dim=-1)
        anchor_vecs = normed[anchors]  # [n_anchors, D]
        return normed @ anchor_vecs.T  # [alphabet, n_anchors]

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        query = F.normalize(self.src_rel[src_code], dim=-1)  # [B, n_anchors]
        keys = F.normalize(self.dst_rel, dim=-1)  # [alphabet, n_anchors]
        sim = query @ keys.T  # [B, alphabet]
        return sim.argmax(dim=-1)
```

- [ ] 5.4 Run it — expected **PASS**:
  `uv run pytest tests/unit/test_transducer_baselines.py -q`
  Expected: `4 passed`.

- [ ] 5.5 Lint + types as in 4.5.

- [ ] 5.6 Commit:

```bash
git add track_p/transducer_baselines.py tests/unit/test_transducer_baselines.py
git commit -m "add relative representation transducer baseline"
```

---

## Task 6 — vec2vec-style unsupervised transducer baseline

**Files:**
- Modify: `track_p/transducer_baselines.py`
- Test: `tests/unit/test_transducer_baselines.py`

Steps:

- [ ] 6.1 Append the failing test:

```python
from track_p.transducer_baselines import Vec2VecTransducer


def test_vec2vec_trains_and_maps_valid_codes():
    torch.manual_seed(4)
    src_cb = torch.randn(64, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    dst_cb = src_cb @ q
    t = Vec2VecTransducer(src_codebook=src_cb, dst_codebook=dst_cb, seed=0)
    history = t.fit(steps=200)
    assert len(history) == 200
    # Cycle-consistency loss should decrease over training.
    assert history[-1] < history[0]
    dst_code = t.forward(torch.tensor([1, 30, 63]))
    assert dst_code.shape == (3,)
    assert (dst_code >= 0).all() and (dst_code < 64).all()
```

- [ ] 6.2 Run it — expected **FAIL**:
  `uv run pytest tests/unit/test_transducer_baselines.py -k vec2vec -q`
  Expected: `ImportError: cannot import name 'Vec2VecTransducer'`.

- [ ] 6.3 Append the implementation to `track_p/transducer_baselines.py`:

```python
class Vec2VecTransducer(nn.Module):
    """Unsupervised src->dst code map (vec2vec-style GAN + cycle-consistency).

    Jha et al. 2025 (arXiv:2505.12540): translate between two latent spaces
    with NO paired supervision, using an adversarial loss (discriminator must
    not tell translated-src from real-dst) plus a cycle-consistency loss
    (`G_dst->src(G_src->dst(x)) ~= x`). Here the two "spaces" are the src and
    dst WML codebooks treated as unpaired point clouds.

    `fit` trains the two generators and one discriminator; `forward` maps a
    src code by translating its embedding and taking the nearest dst row.

    Parameters
    ----------
    src_codebook, dst_codebook
        `[alphabet_size, D]` float tensors, same shape.
    hidden
        Width of the generator / discriminator MLPs.
    lambda_cycle
        Weight of the cycle-consistency term.
    seed
        RNG seed for parameter init.
    """

    def __init__(
        self,
        src_codebook: Tensor,
        dst_codebook: Tensor,
        *,
        hidden: int = 64,
        lambda_cycle: float = 10.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if src_codebook.shape != dst_codebook.shape:
            raise ValueError(
                f"codebook shape mismatch: {src_codebook.shape} "
                f"vs {dst_codebook.shape}"
            )
        torch.manual_seed(seed)
        self.alphabet_size, dim = src_codebook.shape
        self.lambda_cycle = float(lambda_cycle)
        self.register_buffer("_src_codebook", src_codebook.detach().clone())
        self.register_buffer("_dst_codebook", dst_codebook.detach().clone())

        def _mlp(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_out),
            )

        self.g_src2dst = _mlp(dim, dim)
        self.g_dst2src = _mlp(dim, dim)
        self.discriminator = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def fit(self, *, steps: int = 2000, lr: float = 1e-3) -> list[float]:
        """Adversarial + cycle training. Returns the per-step cycle loss."""
        gen_params = list(self.g_src2dst.parameters()) + list(
            self.g_dst2src.parameters()
        )
        opt_g = torch.optim.Adam(gen_params, lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        src = self._src_codebook
        dst = self._dst_codebook
        ones = torch.ones(self.alphabet_size, 1)
        zeros = torch.zeros(self.alphabet_size, 1)
        cycle_history: list[float] = []
        for _ in range(steps):
            # --- discriminator step ---
            opt_d.zero_grad()
            fake = self.g_src2dst(src).detach()
            loss_d = bce(self.discriminator(dst), ones) + bce(
                self.discriminator(fake), zeros
            )
            loss_d.backward()
            opt_d.step()
            # --- generator step: fool D + cycle-consistency ---
            opt_g.zero_grad()
            fake_dst = self.g_src2dst(src)
            loss_adv = bce(self.discriminator(fake_dst), ones)
            cyc_src = F.mse_loss(self.g_dst2src(fake_dst), src)
            cyc_dst = F.mse_loss(
                self.g_src2dst(self.g_dst2src(dst)), dst
            )
            loss_cycle = cyc_src + cyc_dst
            (loss_adv + self.lambda_cycle * loss_cycle).backward()
            opt_g.step()
            cycle_history.append(float(loss_cycle.detach()))
        return cycle_history

    def forward(self, src_code: Tensor) -> Tensor:
        """Map `[B]` long src codes to `[B]` long dst codes."""
        with torch.no_grad():
            src_emb = self._src_codebook[src_code]  # [B, D]
            translated = self.g_src2dst(src_emb)  # [B, D]
            dist = torch.cdist(translated, self._dst_codebook)
        return dist.argmin(dim=-1)
```

- [ ] 6.4 Run it — expected **PASS**:
  `uv run pytest tests/unit/test_transducer_baselines.py -q`
  Expected: `5 passed`.

- [ ] 6.5 Lint + types as in 4.5.

- [ ] 6.6 Commit:

```bash
git add track_p/transducer_baselines.py tests/unit/test_transducer_baselines.py
git commit -m "add vec2vec unsupervised transducer baseline"
```

---

## Task 7 — Transducer-vs-baselines benchmark runner

**Files:**
- Create: `scripts/transducer_baselines_pilot.py`
- Test: `tests/integration/test_transducer_baselines_pilot.py`

This runner trains the learned `Transducer` on a code-translation task
between two WML substrates and compares it against the three baselines,
reporting MI and entropy.

Steps:

- [ ] 7.1 Write the failing test. Create
  `tests/integration/test_transducer_baselines_pilot.py`:

```python
import pytest

from scripts.transducer_baselines_pilot import run_transducer_benchmark


@pytest.mark.slow
def test_run_transducer_benchmark_reports_all_methods():
    results = run_transducer_benchmark(steps=300, seed=0)
    assert set(results) == {
        "learned",
        "procrustes",
        "relative_rep",
        "vec2vec",
    }
    for name, row in results.items():
        # MI is non-negative and <= log2(64) = 6 bits.
        assert 0.0 <= row["mi_bits"] <= 6.01, name
        # Entropy of the dst-code distribution, also <= 6 bits.
        assert 0.0 <= row["entropy_bits"] <= 6.01, name


@pytest.mark.slow
def test_learned_transducer_beats_random_floor():
    results = run_transducer_benchmark(steps=300, seed=0)
    # The learned transducer must transmit more than the ~0-bit floor.
    assert results["learned"]["mi_bits"] > 0.5
```

- [ ] 7.2 Run it — expected **FAIL**:
  `uv run pytest tests/integration/test_transducer_baselines_pilot.py -q`
  Expected: `ModuleNotFoundError: No module named 'scripts.transducer_baselines_pilot'`.

- [ ] 7.3 Minimal implementation. Create
  `scripts/transducer_baselines_pilot.py`:

```python
"""Benchmark the learned Transducer against three baselines.

Closes gap 1: runs the learned `track_p.transducer.Transducer` and the three
`track_p.transducer_baselines` maps on the SAME code-translation task between
two WML substrates (an MlpWML src and an MlpWML dst, distinct seeds so the
codebooks differ). Reports MI (bits) between predicted dst codes and the
ground-truth dst codes, and the entropy of the predicted distribution.

Run directly:  uv run python -m scripts.transducer_baselines_pilot
"""
from __future__ import annotations

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F  # noqa: N812

from nerve_wml.methodology.mi_estimators import (
    entropy_discrete,
    mi_miller_madow_discrete,
)
from track_p.transducer import Transducer, TransducerGating
from track_p.transducer_baselines import (
    ProcrustesTransducer,
    RelativeRepTransducer,
    Vec2VecTransducer,
)
from track_w.mlp_wml import MlpWML

_BITS = float(np.log2(np.e))  # nats -> bits


def _build_task(seed: int) -> tuple[torch.Tensor, torch.Tensor, MlpWML, MlpWML]:
    """Construct a deterministic src->dst code-translation task.

    Ground truth: dst code = src code permuted by a fixed random permutation.
    The two WML codebooks are the latent spaces the transducers must align.
    """
    torch.manual_seed(seed)
    src_wml = MlpWML(id=0, alphabet_size=64, seed=seed)
    dst_wml = MlpWML(id=1, alphabet_size=64, seed=seed + 1)
    perm = torch.randperm(64)
    src_codes = torch.randint(0, 64, (1024,))
    dst_codes = perm[src_codes]
    return src_codes, dst_codes, src_wml, dst_wml


def _train_learned(
    src_codes: torch.Tensor, dst_codes: torch.Tensor, steps: int
) -> Transducer:
    """Train the learned Transducer in GUMBEL_SOFTMAX mode (gradient flows)."""
    t = Transducer(alphabet_size=64, gating=TransducerGating.GUMBEL_SOFTMAX)
    opt = Adam(t.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        soft = t.forward(src_codes, hard=False, tau=1.0)  # [B, 64]
        loss = F.cross_entropy(torch.log(soft + 1e-9), dst_codes)
        loss.backward()
        opt.step()
    return t


def _mi_entropy_bits(
    pred: torch.Tensor, truth: torch.Tensor
) -> dict[str, float]:
    """MI (bits) between pred/truth and entropy (bits) of pred."""
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    truth_np = truth.detach().cpu().numpy().astype(np.int64)
    return {
        "mi_bits": mi_miller_madow_discrete(pred_np, truth_np) * _BITS,
        "entropy_bits": entropy_discrete(pred_np) * _BITS,
    }


def run_transducer_benchmark(
    *, steps: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Run all four methods on one task; return per-method MI/entropy."""
    src_codes, dst_codes, src_wml, dst_wml = _build_task(seed)
    src_cb = src_wml.codebook.detach()
    dst_cb = dst_wml.codebook.detach()

    results: dict[str, dict[str, float]] = {}

    learned = _train_learned(src_codes, dst_codes, steps)
    learned_pred = learned.forward(src_codes, hard=True)
    results["learned"] = _mi_entropy_bits(learned_pred, dst_codes)

    proc = ProcrustesTransducer(src_codebook=src_cb, dst_codebook=dst_cb)
    results["procrustes"] = _mi_entropy_bits(
        proc.forward(src_codes), dst_codes
    )

    rel = RelativeRepTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=32, seed=seed
    )
    results["relative_rep"] = _mi_entropy_bits(
        rel.forward(src_codes), dst_codes
    )

    v2v = Vec2VecTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, seed=seed
    )
    v2v.fit(steps=max(steps, 200))
    results["vec2vec"] = _mi_entropy_bits(v2v.forward(src_codes), dst_codes)

    return results


def main() -> None:
    results = run_transducer_benchmark(steps=2000, seed=0)
    print(f"{'method':<14}{'MI (bits)':>12}{'H (bits)':>12}")
    for name, row in results.items():
        print(f"{name:<14}{row['mi_bits']:>12.3f}{row['entropy_bits']:>12.3f}")


if __name__ == "__main__":
    main()
```

- [ ] 7.4 Run it — expected **PASS**:
  `uv run pytest tests/integration/test_transducer_baselines_pilot.py -q --run-slow`
  (if `--run-slow` is not wired, run `uv run pytest tests/integration/test_transducer_baselines_pilot.py -q -m slow`)
  Expected: `2 passed`.

- [ ] 7.5 Smoke-run the script:
  `uv run python -m scripts.transducer_baselines_pilot`
  Expected: a 4-row table with finite MI/H values, `learned` MI > 0.5 bits.

- [ ] 7.6 Lint + types:
  `uv run ruff check scripts/transducer_baselines_pilot.py`
  `uv run mypy track_p track_w` (expect success).

- [ ] 7.7 Commit:

```bash
git add scripts/transducer_baselines_pilot.py tests/integration/test_transducer_baselines_pilot.py
git commit -m "add transducer baselines benchmark runner"
```

---

## Task 8 — Simple learned-gating module (GTM ablation control)

**Files:**
- Modify: `track_p/transducer_baselines.py`
- Test: `tests/unit/test_simple_gating.py`

The GTM ablation needs a control: a plain learned gating module that routes
codes without band-multiplexing. We place it in `transducer_baselines.py`
(the module is small and shares the "baseline" theme).

Steps:

- [ ] 8.1 Write the failing test. Create `tests/unit/test_simple_gating.py`:

```python
import torch

from track_p.transducer_baselines import SimpleGatingMultiplexer


def test_simple_gating_round_trips_codes_noise_free():
    torch.manual_seed(0)
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (8, 7))
    carrier = m.forward(codes)
    assert carrier.shape[0] == 8
    recovered = m.demodulate(carrier)
    # Untrained module need not be accurate, but shapes must round-trip.
    assert recovered.shape == codes.shape
    assert (recovered >= 0).all() and (recovered < 64).all()


def test_simple_gating_is_differentiable():
    torch.manual_seed(1)
    m = SimpleGatingMultiplexer(alphabet_size=64, n_symbols=7)
    codes = torch.randint(0, 64, (4, 7))
    carrier = m.forward(codes)
    carrier.sum().backward()
    assert m.gate.weight.grad is not None
```

- [ ] 8.2 Run it — expected **FAIL**:
  `uv run pytest tests/unit/test_simple_gating.py -q`
  Expected: `ImportError: cannot import name 'SimpleGatingMultiplexer'`.

- [ ] 8.3 Append to `track_p/transducer_baselines.py`:

```python
class SimpleGatingMultiplexer(nn.Module):
    """Plain learned-gating control for the GTM ablation.

    Closes gap 2 (ablation arm): a minimal alternative to
    `track_p.multiplexer.GammaThetaMultiplexer` with NO theta/gamma band
    multiplexing. Each of the `n_symbols` code slots is embedded and summed
    into a flat carrier through a single learned linear gate; demodulation is
    a learned linear read-out. Same `forward(codes)->carrier` /
    `demodulate(carrier)->codes` contract as GTM, so the ablation script can
    swap them.

    Parameters
    ----------
    alphabet_size
        Code alphabet size.
    n_symbols
        Code slots per carrier (GTM's `symbols_per_theta`).
    carrier_dim
        Width of the flat carrier vector.
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 64,
        n_symbols: int = 7,
        carrier_dim: int = 64,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.n_symbols = n_symbols
        self.carrier_dim = carrier_dim
        self.embed = nn.Embedding(alphabet_size, carrier_dim)
        # One learned scalar gate weight per symbol slot.
        self.gate = nn.Linear(n_symbols, n_symbols, bias=False)
        self.readout = nn.Linear(carrier_dim, n_symbols * alphabet_size)

    def forward(self, codes: Tensor) -> Tensor:
        """Encode `[B, n_symbols]` long codes to a `[B, carrier_dim]` carrier."""
        if codes.shape[-1] != self.n_symbols:
            raise ValueError(
                f"expected {self.n_symbols} symbols, got {codes.shape[-1]}"
            )
        emb = self.embed(codes)  # [B, n_symbols, carrier_dim]
        slot_id = torch.eye(self.n_symbols, device=codes.device)
        gates = self.gate(slot_id).diagonal()  # [n_symbols]
        return (emb * gates[None, :, None]).sum(dim=1)  # [B, carrier_dim]

    def demodulate(self, carrier: Tensor) -> Tensor:
        """Recover `[B, n_symbols]` long codes from a `[B, carrier_dim]` carrier."""
        logits = self.readout(carrier)  # [B, n_symbols * alphabet]
        logits = logits.view(-1, self.n_symbols, self.alphabet_size)
        return logits.argmax(dim=-1)

    def demodulate_logits(self, carrier: Tensor) -> Tensor:
        """`[B, n_symbols, alphabet_size]` logits — for training the read-out."""
        logits = self.readout(carrier)
        return logits.view(-1, self.n_symbols, self.alphabet_size)
```

- [ ] 8.4 Run it — expected **PASS**:
  `uv run pytest tests/unit/test_simple_gating.py -q`
  Expected: `2 passed`.

- [ ] 8.5 Lint + types as in 4.5.

- [ ] 8.6 Commit:

```bash
git add track_p/transducer_baselines.py tests/unit/test_simple_gating.py
git commit -m "add simple gating multiplexer for gtm ablation"
```

---

## Task 9 — GTM ablation + synchrony-collapse runner

**Files:**
- Create: `scripts/gtm_ablation_pilot.py`
- Test: `tests/integration/test_gtm_ablation_pilot.py`

Runs the GTM vs the SimpleGatingMultiplexer on the same routing task and
also measures synchrony-collapse: the variance of the demodulator's
per-symbol output across training (collapse = variance shrinking toward 0,
i.e. all symbols decode to the same code — the failure mode of oscillator
nets under end-to-end training, Phasor Agents arXiv:2601.04362).

Steps:

- [ ] 9.1 Write the failing test. Create
  `tests/integration/test_gtm_ablation_pilot.py`:

```python
import pytest

from scripts.gtm_ablation_pilot import run_gtm_ablation


@pytest.mark.slow
def test_run_gtm_ablation_reports_both_arms():
    result = run_gtm_ablation(steps=300, seed=0)
    assert set(result) == {"gtm", "simple_gating"}
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
```

- [ ] 9.2 Run it — expected **FAIL**:
  `uv run pytest tests/integration/test_gtm_ablation_pilot.py -q`
  Expected: `ModuleNotFoundError: No module named 'scripts.gtm_ablation_pilot'`.

- [ ] 9.3 Minimal implementation. Create `scripts/gtm_ablation_pilot.py`:

```python
"""GTM ablation + synchrony-collapse analysis.

Closes gap 2. Two arms run the SAME 7-symbol routing task:
- `gtm`           : `track_p.multiplexer.GammaThetaMultiplexer`
- `simple_gating` : `track_p.transducer_baselines.SimpleGatingMultiplexer`

For each arm we report symbol-recovery accuracy, MI (bits) between recovered
and true codes, and a SYNCHRONY INDEX: the fraction of carrier energy that
collapses onto a single shared mode. Oscillator nets are known to collapse to
global synchrony under end-to-end training (Phasor Agents, arXiv:2601.04362);
a synchrony index near 1.0 means every symbol decodes alike (channel dead).

Run directly:  uv run python -m scripts.gtm_ablation_pilot
"""
from __future__ import annotations

import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F  # noqa: N812

from nerve_wml.methodology.mi_estimators import mi_miller_madow_discrete
from track_p.multiplexer import GammaThetaConfig, GammaThetaMultiplexer
from track_p.transducer_baselines import SimpleGatingMultiplexer

_BITS = float(np.log2(np.e))


def _synchrony_index(carrier: torch.Tensor) -> float:
    """Fraction of carrier variance on its top principal mode.

    1.0 == every example's carrier is a scalar multiple of one shared vector
    (global synchrony / collapse). ~1/D == energy spread across all modes.
    """
    centred = carrier - carrier.mean(dim=0, keepdim=True)
    if centred.shape[0] < 2:
        return 1.0
    svals = torch.linalg.svdvals(centred)
    energy = (svals**2).sum()
    if float(energy) == 0.0:
        return 1.0
    return float((svals[0] ** 2) / energy)


def _train_gtm(
    codes: torch.Tensor, steps: int, seed: int
) -> tuple[float, float, float]:
    """Train GTM end-to-end; return (accuracy, mi_bits, synchrony_index)."""
    cfg = GammaThetaConfig(symbols_per_theta=codes.shape[-1])
    gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)
    opt = Adam(gtm.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = gtm.forward(codes)
        soft = gtm.demodulate(carrier, hard=False, tau=1.0)
        loss = F.cross_entropy(
            torch.log(soft + 1e-9).reshape(-1, soft.shape[-1]),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        gtm.step()
    with torch.no_grad():
        carrier = gtm.forward(codes)
        pred = gtm.demodulate(carrier, hard=True)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def _train_simple(
    codes: torch.Tensor, steps: int
) -> tuple[float, float, float]:
    """Train the simple-gating control; same return triple as `_train_gtm`."""
    m = SimpleGatingMultiplexer(
        alphabet_size=64, n_symbols=codes.shape[-1]
    )
    opt = Adam(m.parameters(), lr=0.02)
    for _ in range(steps):
        opt.zero_grad()
        carrier = m.forward(codes)
        logits = m.demodulate_logits(carrier)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        carrier = m.forward(codes)
        pred = m.demodulate(carrier)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def run_gtm_ablation(
    *, steps: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Run both arms on one routing task; return per-arm metrics."""
    torch.manual_seed(seed)
    codes = torch.randint(0, 64, (128, 7))
    acc_g, mi_g, sync_g = _train_gtm(codes, steps, seed)
    acc_s, mi_s, sync_s = _train_simple(codes, steps)
    return {
        "gtm": {
            "accuracy": acc_g,
            "mi_bits": mi_g,
            "synchrony_index": sync_g,
        },
        "simple_gating": {
            "accuracy": acc_s,
            "mi_bits": mi_s,
            "synchrony_index": sync_s,
        },
    }


def main() -> None:
    result = run_gtm_ablation(steps=2000, seed=0)
    header = f"{'arm':<16}{'acc':>10}{'MI (bits)':>12}{'synchrony':>12}"
    print(header)
    for arm, row in result.items():
        print(
            f"{arm:<16}{row['accuracy']:>10.3f}"
            f"{row['mi_bits']:>12.3f}{row['synchrony_index']:>12.3f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] 9.4 Run it — expected **PASS**:
  `uv run pytest tests/integration/test_gtm_ablation_pilot.py -q -m slow`
  Expected: `2 passed`.

- [ ] 9.5 Smoke-run:
  `uv run python -m scripts.gtm_ablation_pilot`
  Expected: a 2-row table; `gtm` synchrony index < 0.95.

- [ ] 9.6 Lint + types:
  `uv run ruff check scripts/gtm_ablation_pilot.py`
  `uv run mypy track_p` (expect success).

- [ ] 9.7 Commit:

```bash
git add scripts/gtm_ablation_pilot.py tests/integration/test_gtm_ablation_pilot.py
git commit -m "add gtm ablation and synchrony collapse runner"
```

---

## Task 10 — Scale-robustness experiment runner

**Files:**
- Create: `scripts/scale_robustness_pilot.py`
- Test: `tests/integration/test_scale_robustness_pilot.py`

Re-measures the substrate-gap / shared-info quantities at increasing sample
sizes using the debiased HSIC and CKNNA estimators from Tasks 1-3, on real
WML substrate codes.

Steps:

- [ ] 10.1 Write the failing test. Create
  `tests/integration/test_scale_robustness_pilot.py`:

```python
import pytest

from scripts.scale_robustness_pilot import run_scale_robustness


@pytest.mark.slow
def test_run_scale_robustness_returns_rows_per_size():
    rows = run_scale_robustness(
        sizes=(64, 128, 256), seed=0
    )
    assert [r["n"] for r in rows] == [64, 128, 256]
    for r in rows:
        assert "hsic" in r and "cknna" in r
        assert 0.0 <= r["cknna"] <= 1.0


@pytest.mark.slow
def test_scale_robustness_hsic_finite():
    rows = run_scale_robustness(sizes=(64, 128), seed=1)
    import math

    assert all(math.isfinite(r["hsic"]) for r in rows)
```

- [ ] 10.2 Run it — expected **FAIL**:
  `uv run pytest tests/integration/test_scale_robustness_pilot.py -q`
  Expected: `ModuleNotFoundError: No module named 'scripts.scale_robustness_pilot'`.

- [ ] 10.3 Minimal implementation. Create
  `scripts/scale_robustness_pilot.py`:

```python
"""Scale-robustness check for the shared-info / substrate-gap measurements.

Closes gap 3 (scale arm). PRH critiques (Huh et al. 2024) show that
representation-alignment scores can degrade as sample size grows. This runner
emits the same WML codebook embeddings through two substrates (an MlpWML and a
LifWML), then recomputes debiased HSIC and CKNNA at increasing subsample sizes
via `nerve_wml.methodology.hsic_cknna.scale_robustness_sweep`. A stable
alignment shows flat curves; a biased one shows the score drifting with N.

Run directly:  uv run python -m scripts.scale_robustness_pilot
"""
from __future__ import annotations

import torch

from nerve_wml.methodology.hsic_cknna import scale_robustness_sweep
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML


def _substrate_embeddings(seed: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two `[n, D]` embedding clouds from an MlpWML and a LifWML codebook.

    Each row is the codebook embedding of a randomly drawn code; the two
    substrates share the code stream so the pair is aligned by construction.
    """
    torch.manual_seed(seed)
    mlp = MlpWML(id=0, alphabet_size=64, seed=seed)
    lif = LifWML(id=1, alphabet_size=64, seed=seed + 1)
    codes = torch.randint(0, 64, (n,))
    mlp_emb = mlp.codebook.detach()[codes]
    lif_emb = lif.codebook.detach()[codes]
    return mlp_emb, lif_emb


def run_scale_robustness(
    *, sizes: tuple[int, ...] = (64, 128, 256, 512), seed: int = 0
) -> list[dict[str, float]]:
    """Run the HSIC/CKNNA sweep on real substrate embeddings."""
    n_max = max(sizes)
    mlp_emb, lif_emb = _substrate_embeddings(seed, n_max)
    rows = scale_robustness_sweep(
        mlp_emb.cpu().numpy(),
        lif_emb.cpu().numpy(),
        sizes=sizes,
        k=10,
        seed=seed,
    )
    return [
        {"n": float(r.n), "hsic": r.hsic, "cknna": r.cknna} for r in rows
    ]


def main() -> None:
    rows = run_scale_robustness(sizes=(64, 128, 256, 512), seed=0)
    print(f"{'N':>8}{'HSIC':>14}{'CKNNA':>10}")
    for r in rows:
        print(f"{int(r['n']):>8}{r['hsic']:>14.5f}{r['cknna']:>10.3f}")


if __name__ == "__main__":
    main()
```

  Note: if `LifWML.codebook` has a different `D` than `MlpWML.codebook`,
  the sweep still works (HSIC/CKNNA accept differing `D`); both estimators
  only require equal `N`.

- [ ] 10.4 Run it — expected **PASS**:
  `uv run pytest tests/integration/test_scale_robustness_pilot.py -q -m slow`
  Expected: `2 passed`.

- [ ] 10.5 Smoke-run:
  `uv run python -m scripts.scale_robustness_pilot`
  Expected: a 4-row N/HSIC/CKNNA table with finite values.

- [ ] 10.6 Lint + types:
  `uv run ruff check scripts/scale_robustness_pilot.py`
  `uv run mypy track_w nerve_core` (expect success).

- [ ] 10.7 Commit:

```bash
git add scripts/scale_robustness_pilot.py tests/integration/test_scale_robustness_pilot.py
git commit -m "add scale robustness experiment runner"
```

---

## Task 11 — Golden numerics + AKOrN future-work note

**Files:**
- Create: `tests/golden/test_validation_suite_golden.py`
- Create: `docs/superpowers/research/akorn-future-work.md`

Freezes deterministic numbers so regressions are caught, and documents the
AKOrN (Kuramoto neurons, ICLR 2025, arXiv:2410.13821) comparison as scoped
future work — gap 2 explicitly allows documenting it when a full
implementation is out of scope.

Steps:

- [ ] 11.1 Write the failing test. Create
  `tests/golden/test_validation_suite_golden.py`:

```python
import numpy as np

from nerve_wml.methodology.hsic_cknna import cknna, hsic_debiased


def test_hsic_debiased_golden_value():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((300, 8))
    y = x + 0.2 * rng.standard_normal((300, 8))
    val = hsic_debiased(x, y)
    # Frozen 2026-05-19: numpy default_rng is stable across versions.
    assert abs(val - 7.6759) < 1e-3, val


def test_cknna_golden_value():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((300, 8))
    y = x + 0.2 * rng.standard_normal((300, 8))
    val = cknna(x, y, k=10)
    # Frozen 2026-05-19.
    assert abs(val - 0.7770) < 1e-3, val
```

- [ ] 11.2 Run it — expected **FAIL** on the golden constants (the asserted
  numbers are placeholders until measured):
  `uv run pytest tests/golden/test_validation_suite_golden.py -q`
  Expected: `2 failed` with the actual values printed in the assert message.

- [ ] 11.3 Capture the real golden numbers. Run:

```bash
uv run python -c "
import numpy as np
from nerve_wml.methodology.hsic_cknna import cknna, hsic_debiased
rng = np.random.default_rng(123)
x = rng.standard_normal((300, 8))
y = x + 0.2 * rng.standard_normal((300, 8))
print('hsic', repr(hsic_debiased(x, y)))
print('cknna', repr(cknna(x, y, k=10)))
"
```

  Then edit `tests/golden/test_validation_suite_golden.py`, replacing
  `7.6759` and `0.7770` with the printed values (rounded to 4 decimals).

- [ ] 11.4 Run it — expected **PASS**:
  `uv run pytest tests/golden/test_validation_suite_golden.py -q`
  Expected: `2 passed`.

- [ ] 11.5 Create the AKOrN future-work note
  `docs/superpowers/research/akorn-future-work.md`:

```markdown
# AKOrN comparison — scoped future work

The GTM ablation (Task 9) compares `GammaThetaMultiplexer` against a plain
learned gating control (`SimpleGatingMultiplexer`) and measures synchrony
collapse. A third, stronger comparison point is **AKOrN** — Artificial
Kuramoto Oscillatory Neurons (Miyato et al., ICLR 2025, arXiv:2410.13821) —
which replaces threshold units with Kuramoto oscillators whose phase
dynamics give them an explicit synchrony mechanism.

## Why it is out of scope for this plan

- AKOrN needs a Kuramoto ODE integrator and a phase-aware read-out; that is
  a self-contained module larger than the rest of this suite combined.
- nerve-wml's substrates (`MlpWML`, `LifWML`, `TransformerWML`) are
  feed-forward; an AKOrN substrate would be a fourth `track_w` substrate,
  not a drop-in baseline.

## What lands instead

The synchrony-collapse analysis in Task 9 (`_synchrony_index`) already
captures the failure mode AKOrN is designed to avoid. If GTM's synchrony
index stays well below 1.0, the band-multiplexing claim holds without an
AKOrN head-to-head.

## Follow-up

Tracked as a future `track_w/akorn_wml.py` substrate: implement the Kuramoto
update, then add an `akorn` arm to `scripts/gtm_ablation_pilot.py`. Estimated
1-2 tasks of their own; deferred to a later validation plan.
```

- [ ] 11.6 Lint:
  `uv run ruff check tests/golden/test_validation_suite_golden.py`
  Run the full fast suite to confirm no regressions:
  `uv run pytest -m "not slow"` (expect all green, ~80s).

- [ ] 11.7 Commit:

```bash
git add tests/golden/test_validation_suite_golden.py docs/superpowers/research/akorn-future-work.md
git commit -m "freeze validation golden numbers, note akorn"
```

---

## Self-Review

**Spec coverage.**
- Gap 1 (transducer baselines): Procrustes (Task 4), relative-rep (Task 5),
  vec2vec (Task 6) all in `track_p/transducer_baselines.py`; benchmarked
  against the learned `Transducer` with MI/H reporting in Task 7. ✅
- Gap 2 (GTM ablation + stability): SimpleGatingMultiplexer control (Task 8),
  GTM-vs-gating ablation + synchrony-collapse `_synchrony_index` (Task 9),
  AKOrN documented as future work (Task 11). ✅
- Gap 3 (debiased metrics + scale): debiased HSIC (Task 1), CKNNA mutual
  k-NN (Task 2), scale-robustness sweep (Task 3), real-substrate scale
  experiment (Task 10). ✅
- Ordering: metrics (Tasks 1-3) precede the experiments that import them
  (Tasks 7, 9, 10). ✅
- Files match the spec: `track_p/transducer_baselines.py`,
  `nerve_wml/methodology/hsic_cknna.py`, runners in `scripts/`. ✅

**Placeholder scan.** No "TBD"/"similar to"/"add error handling" left. The
only deliberately-deferred numbers are the two golden constants in Task 11,
which Task 11.2 explicitly expects to FAIL and Task 11.3 replaces with
measured values via a shown command. Every code block is complete runnable
PyTorch/numpy: Procrustes via `torch.linalg.svd`, relative-rep via
`F.normalize` cosine-to-anchors, vec2vec via a real GAN
(`BCEWithLogitsLoss` + cycle MSE), debiased HSIC via the Song et al. 2012
unbiased formula, CKNNA via `argpartition` mutual k-NN masks.

**Type consistency.** All new functions are fully annotated; numpy helpers
take/return `np.ndarray` and `float`, matching `mi_estimators.py`. Torch
modules return `Tensor`. Dataclasses (`ScaleRobustnessRow`) are `frozen`.
`uv run mypy nerve_core track_p track_w` is run in Tasks 1, 4, 7, 9, 10.

**Test layering.** L1 unit tests for pure modules (Tasks 1-3, 4-6, 8),
L3 integration for runners (Tasks 7, 9, 10), L4 golden for frozen numerics
(Task 11). Long experiments carry `@pytest.mark.slow`.

---

## Execution Handoff

This plan is ready to execute. Choose one mode:

- **Subagent-Driven (recommended).** Use `superpowers:subagent-driven-development`:
  dispatch one subagent per Task with fresh context, each running the full
  TDD cycle (failing test → implementation → passing test → commit). Review
  the diff at the checkpoint between every Task. Best for catching drift and
  keeping context small; matches the `feedback_critic_before_ship` preference
  of fresh-context review.

- **Inline.** Execute the Tasks yourself in order in this session, top to
  bottom. Faster for a single sitting; run `uv run pytest -m "not slow"` after
  each commit and the slow suite after Tasks 7, 9, 10.

Either way: keep Task order (metrics before experiments), do not squash the
per-Task commits (they map 1:1 to the gap-analysis items), and after Task 11
run the full `uv run pytest` once to confirm the slow experiments pass.
