"""DVNC baseline on HardFlowProxyTask (scaffold, wiring TODO).

Adapts Tang et al. 2021 "Discrete-Valued Neural Communication"
(NeurIPS, arXiv:2107.02367) from its original cooperative-RL
regime to the supervised classification setup used by
scripts/save_codes_for_checks.py, so that the resulting emitted
codes can be fed into the same nerve_wml.methodology.* measurement
pipeline (null-model, bootstrap CI, multi-estimator).

*** SCAFFOLD -- wiring is not yet in place. ***

Before this script runs, complete Day 1 of the protocol in
docs/research-notes/dvnc-baseline-protocol.md:

  1. Clone DVNC reference implementation to a sibling directory:
       ssh kxkm@kxkm-ai
       cd ~ && git clone <DVNC_REPO_URL> dvnc-reference
  2. Identify the VectorQuantizer / VQBottleneck module in their
     source tree. Either:
       (a) copy that module into third_party/dvnc/ in nerve-wml, or
       (b) pip install -e ../dvnc-reference and import directly.
  3. Uncomment the TODO-marked imports and the
     _build_vq_bottleneck() body below, matching their API.

Then run:

    uv run python scripts/baseline_dvnc.py \\
        --seeds 0 1 2 --n-eval 5000 --steps 800 \\
        --out tests/golden/codes_dvnc.npz

Hyperparameters (lr, batch, steps, d_hidden, codebook_size) MUST
match scripts/save_codes_for_checks.py for the comparison to be
meaningful. Argparse defaults already mirror run_w2_hard; do not
change them without updating the comparison methodology in
docs/research-notes/dvnc-baseline-protocol.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

# TODO Day 1 PM: uncomment one of the following after cloning DVNC:
#
# Option A (vendored copy in nerve-wml):
#   from third_party.dvnc.vq_bottleneck import VectorQuantizer
#
# Option B (install DVNC as editable dep):
#   from dvnc.vq_bottleneck import VectorQuantizer
#
# Option C (minimal re-implementation if DVNC repo is obsolete):
#   see _MinimalSharedVectorQuantizer class stub below.


class _MinimalSharedVectorQuantizer(nn.Module):
    """Placeholder VQ module for Option C (reimplementation).

    Shared codebook between two agents, commitment-loss trained
    (VQ-VAE style). Quantises each agent's continuous encoding to
    the nearest codebook vector via Euclidean distance, returns
    the index + the straight-through quantized vector.

    *** STUB: Day 2 AM wiring task is to either replace this with
    DVNC's canonical VectorQuantizer (preferred) or to flesh this
    out minimally from the paper's algorithm section. ***
    """

    def __init__(self, codebook_size: int, d_hidden: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, d_hidden)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size,
        )
        self.codebook_size = codebook_size
        self.d_hidden = d_hidden

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_q, codes, commitment_loss). TODO verify against DVNC paper."""
        raise NotImplementedError(
            "Stub: replace with DVNC VectorQuantizer or complete the "
            "minimal re-implementation per the paper algorithm."
        )


class _Agent(nn.Module):
    """Homogeneous encoder-classifier pair, DVNC style (no substrate asymmetry)."""

    def __init__(self, input_dim: int, d_hidden: int, n_classes: int) -> None:
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


def _build_shared_vq(codebook_size: int, d_hidden: int):
    """Instantiate the shared VQ bottleneck.

    TODO Day 1 PM: replace the stub with the actual DVNC
    VectorQuantizer instance.
    """
    raise NotImplementedError(
        "Wiring TODO: swap in DVNC VectorQuantizer. See top-of-file "
        "docstring for instructions."
    )
    # Example wiring once DVNC is imported:
    #   return VectorQuantizer(
    #       num_embeddings=codebook_size,
    #       embedding_dim=d_hidden,
    #       commitment_cost=0.25,  # DVNC default, verify
    #   )


def _train_dvnc_pair(
    seed: int,
    steps: int,
    d_hidden: int = 16,
    codebook_size: int = 64,
    lr: float = 1e-2,
) -> tuple[_Agent, _Agent, nn.Module, HardFlowProxyTask]:
    """Train two homogeneous agents with a shared VQ codebook.

    Both agents receive the same input x, encode it via their
    respective encoder, then both encodings are quantised through
    the SHARED VectorQuantizer. Loss = CE(agent_a) + CE(agent_b)
    + commitment_loss (VQ-VAE standard).

    Returns: (agent_a, agent_b, shared_vq, task) for downstream
    eval-time code extraction.

    TODO Day 2 AM: complete the forward + loss + optimizer loop.
    """
    torch.manual_seed(seed)

    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    agent_a = _Agent(input_dim=16, d_hidden=d_hidden, n_classes=12)
    agent_b = _Agent(input_dim=16, d_hidden=d_hidden, n_classes=12)
    shared_vq = _build_shared_vq(codebook_size, d_hidden)

    # TODO: optimizer = torch.optim.Adam(all_params, lr=lr)
    # TODO: for step in range(steps):
    #           x, y = task.sample(batch=64)
    #           z_a = agent_a.encode(x)
    #           z_b = agent_b.encode(x)
    #           z_a_q, codes_a, commit_a = shared_vq(z_a)
    #           z_b_q, codes_b, commit_b = shared_vq(z_b)
    #           logits_a = agent_a.classify(z_a_q)
    #           logits_b = agent_b.classify(z_b_q)
    #           loss = F.cross_entropy(logits_a, y) \\
    #                + F.cross_entropy(logits_b, y) \\
    #                + commit_a + commit_b
    #           opt.zero_grad(); loss.backward(); opt.step()

    raise NotImplementedError("TODO Day 2 AM: complete training loop")


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
        agent_a, agent_b, shared_vq, task = _train_dvnc_pair(
            seed=seed,
            steps=args.steps,
            d_hidden=args.d_hidden,
            codebook_size=args.codebook_size,
        )

        eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
        x_eval, y_eval = eval_task.sample(batch=args.n_eval)

        # TODO Day 2 PM: extract codes + embeddings via forward pass.
        with torch.no_grad():
            # z_a = agent_a.encode(x_eval)
            # z_b = agent_b.encode(x_eval)
            # _, codes_a, _ = shared_vq(z_a)
            # _, codes_b, _ = shared_vq(z_b)
            # pred_a = agent_a.classify(z_a).argmax(-1)
            # pred_b = agent_b.classify(z_b).argmax(-1)
            raise NotImplementedError(
                "TODO Day 2 PM: extract codes + accs after forward pass."
            )

        # Placeholder for when wiring is done:
        # all_codes_a.append(codes_a.cpu().numpy().astype(np.int64))
        # all_codes_b.append(codes_b.cpu().numpy().astype(np.int64))
        # all_emb_a.append(z_a.cpu().numpy().astype(np.float32))
        # all_emb_b.append(z_b.cpu().numpy().astype(np.float32))
        # accs_a.append((pred_a == y_eval).float().mean().item())
        # accs_b.append((pred_b == y_eval).float().mean().item())

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


if __name__ == "__main__":
    main()
