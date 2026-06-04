"""DVNC baseline on HardFlowProxyTask.

Re-implements the core VQ-bottleneck mechanism of DVNC
(Liu et al. 2021, "Discrete-Valued Neural Communication",
NeurIPS 2021, arXiv:2107.02367) in a supervised regime matching
run_w2_hard (scripts/track_w_pilot.py:429).

Why a re-implementation rather than vendoring:
  * The reference repo
    (github.com/kaiyuanmifen/Discrete-Valued-Neural-Communication)
    ships without a LICENSE file, so we cannot legally copy their
    Quantization.py into this MIT repo.
  * DVNC's core is VQ-VAE (van den Oord et al. 2017) with a
    codebook shared between two agents; this is a well-known
    ~50-line recipe that we re-derive from the public algorithm
    without touching their sources.
  * Keeps the comparison fair: same codebook size (64), same
    d_hidden (16), same task (HardFlowProxyTask, 12-class XOR),
    same seeds and hyperparameters as scripts/save_codes_for_checks.py,
    the only structural difference being (a) a shared codebook
    (DVNC) vs per-WML codebook + transducers (nerve-wml) and
    (b) two homogeneous MLP agents (DVNC) vs heterogeneous
    MLP+LIF (nerve-wml).

Usage:
    uv run python scripts/baseline_dvnc.py \\
        --seeds 0 1 2 --n-eval 5000 --steps 800 \\
        --out tests/golden/codes_dvnc.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from track_w.tasks.hard_flow_proxy import HardFlowProxyTask


class _SharedVectorQuantizer(nn.Module):
    """Minimal VQ-VAE bottleneck with shared codebook.

    Faithful re-implementation of the VQ-VAE quantisation step
    (van den Oord et al. 2017, arXiv:1711.00937) -- the basis of
    DVNC's Quantization.Quantize class. A single shared codebook
    is quantised independently for each agent's encoding (the core
    DVNC principle: shared discrete communication channel).

    Does NOT include DVNC's multi-headed grouping extension or
    k-means initialisation; the baseline mirrors van den Oord's
    original single-head formulation, which is a strict subset of
    DVNC and the common denominator for comparison.
    """

    def __init__(
        self,
        codebook_size:     int,
        embedding_dim:     int,
        commitment_cost:   float = 0.25,
        kld_scale:         float = 10.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / codebook_size, 1.0 / codebook_size,
        )
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.kld_scale = kld_scale

    def forward(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise z: [B, D] -> (z_q [B, D], commit_loss, codes [B])."""
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        codes = dist.argmin(dim=-1)
        z_q = self.embedding(codes)

        e_loss = F.mse_loss(z_q.detach(), z)
        q_loss = F.mse_loss(z_q, z.detach())
        commit_loss = self.kld_scale * (self.commitment_cost * e_loss + q_loss)

        z_q = z + (z_q - z).detach()
        return z_q, commit_loss, codes


class _Agent(nn.Module):
    """Homogeneous encoder-classifier agent (DVNC style).

    Two instances of this class share the same structure; the
    comparison with nerve-wml is meaningful because any observed
    gap reflects the shared-codebook vs per-WML-codebook-plus-
    transducers difference, not substrate asymmetry.
    """

    def __init__(
        self,
        input_dim:  int,
        d_hidden:   int,
        n_classes:  int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.classifier = nn.Linear(d_hidden, n_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


def _train_dvnc_pair(
    seed:           int,
    steps:          int,
    d_hidden:       int   = 16,
    codebook_size:  int   = 64,
    lr:             float = 1e-2,
    batch:          int   = 64,
) -> tuple[_Agent, _Agent, _SharedVectorQuantizer, HardFlowProxyTask]:
    """Train two homogeneous agents sharing a VQ codebook.

    Both agents see the same x, encode independently, quantise via
    the shared codebook, then classify from the quantised vector.
    Loss = CE(a) + CE(b) + commit(a) + commit(b).
    """
    torch.manual_seed(seed)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)

    agent_a = _Agent(input_dim=16, d_hidden=d_hidden, n_classes=12)
    agent_b = _Agent(input_dim=16, d_hidden=d_hidden, n_classes=12)
    shared_vq = _SharedVectorQuantizer(
        codebook_size=codebook_size, embedding_dim=d_hidden,
    )

    params = (
        list(agent_a.parameters())
        + list(agent_b.parameters())
        + list(shared_vq.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr)

    for _ in range(steps):
        x, y = task.sample(batch=batch)
        z_a = agent_a.encode(x)
        z_b = agent_b.encode(x)
        z_a_q, commit_a, _ = shared_vq(z_a)
        z_b_q, commit_b, _ = shared_vq(z_b)
        logits_a = agent_a.classify(z_a_q)
        logits_b = agent_b.classify(z_b_q)
        loss = (
            F.cross_entropy(logits_a, y)
            + F.cross_entropy(logits_b, y)
            + commit_a
            + commit_b
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    return agent_a, agent_b, shared_vq, task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--d-hidden", type=int, default=16)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden/codes_dvnc.npz"),
    )
    args = parser.parse_args()

    all_codes_a: list[np.ndarray] = []
    all_codes_b: list[np.ndarray] = []
    all_emb_a: list[np.ndarray] = []
    all_emb_b: list[np.ndarray] = []
    accs_a: list[float] = []
    accs_b: list[float] = []

    for seed in args.seeds:
        print(f"seed {seed}: training DVNC pair ({args.steps} steps)...")
        agent_a, agent_b, shared_vq, _task = _train_dvnc_pair(
            seed=seed,
            steps=args.steps,
            d_hidden=args.d_hidden,
            codebook_size=args.codebook_size,
        )

        eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
        x_eval, y_eval = eval_task.sample(batch=args.n_eval)

        agent_a.eval()
        agent_b.eval()
        shared_vq.eval()
        with torch.no_grad():
            z_a = agent_a.encode(x_eval)
            z_b = agent_b.encode(x_eval)
            _, _, codes_a = shared_vq(z_a)
            _, _, codes_b = shared_vq(z_b)
            z_a_q = shared_vq.embedding(codes_a)
            z_b_q = shared_vq.embedding(codes_b)
            pred_a = agent_a.classify(z_a_q).argmax(-1)
            pred_b = agent_b.classify(z_b_q).argmax(-1)
            acc_a = (pred_a == y_eval).float().mean().item()
            acc_b = (pred_b == y_eval).float().mean().item()

        all_codes_a.append(codes_a.cpu().numpy().astype(np.int64))
        all_codes_b.append(codes_b.cpu().numpy().astype(np.int64))
        all_emb_a.append(z_a.cpu().numpy().astype(np.float32))
        all_emb_b.append(z_b.cpu().numpy().astype(np.float32))
        accs_a.append(acc_a)
        accs_b.append(acc_b)
        print(
            f"  acc_a={acc_a:.4f}, acc_b={acc_b:.4f}, "
            f"alphabet_a={len(np.unique(all_codes_a[-1]))}/64, "
            f"alphabet_b={len(np.unique(all_codes_b[-1]))}/64"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        agent_a_codes=np.stack(all_codes_a),
        agent_b_codes=np.stack(all_codes_b),
        agent_a_embeddings=np.stack(all_emb_a),
        agent_b_embeddings=np.stack(all_emb_b),
        acc_a=np.asarray(accs_a, dtype=np.float32),
        acc_b=np.asarray(accs_b, dtype=np.float32),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        n_eval=args.n_eval,
        steps=args.steps,
    )
    print()
    print(f"Saved: {args.out}")
    print(f"Mean acc: a={np.mean(accs_a):.4f}, b={np.mean(accs_b):.4f}")
    print(
        f"Mean pairwise gap: {abs(np.mean(accs_a) - np.mean(accs_b)):.4f}"
    )


if __name__ == "__main__":
    main()
