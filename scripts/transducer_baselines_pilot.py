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
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

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
    src_codes: torch.Tensor, dst_codes: torch.Tensor, steps: int, *, lr: float = 0.05
) -> Transducer:
    """Train the learned Transducer in GUMBEL_SOFTMAX mode (gradient flows)."""
    t = Transducer(alphabet_size=64, gating=TransducerGating.GUMBEL_SOFTMAX)
    opt = Adam(t.parameters(), lr=lr)
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
    *, steps: int = 2000, seed: int = 0,
    lambda_cycle: float = 10.0, n_anchors: int = 32,
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
        src_codebook=src_cb, dst_codebook=dst_cb, n_anchors=n_anchors, seed=seed
    )
    results["relative_rep"] = _mi_entropy_bits(
        rel.forward(src_codes), dst_codes
    )

    v2v = Vec2VecTransducer(
        src_codebook=src_cb, dst_codebook=dst_cb, lambda_cycle=lambda_cycle, seed=seed
    )
    v2v.fit(steps=max(steps, 200))
    results["vec2vec"] = _mi_entropy_bits(v2v.forward(src_codes), dst_codes)

    # Null arm: feed the learned transducer a shuffled src stream; the
    # prediction should be near-independent of the (un-shuffled) dst codes,
    # so MI collapses near zero. Deterministic via derived seed.
    perm = np.random.default_rng(seed + 9973).permutation(src_codes.shape[0])
    shuffled_pred = learned.forward(
        src_codes[torch.from_numpy(perm).long()], hard=True
    )
    results["null"] = _mi_entropy_bits(shuffled_pred, dst_codes)

    return results


def main() -> None:
    results = run_transducer_benchmark(steps=2000, seed=0)
    print(f"{'method':<14}{'MI (bits)':>12}{'H (bits)':>12}")
    for name, row in results.items():
        print(f"{name:<14}{row['mi_bits']:>12.3f}{row['entropy_bits']:>12.3f}")


if __name__ == "__main__":
    main()
