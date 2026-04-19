"""Neuroletter semantics extractor.

For each of the 64 codes, characterise what the code 'means' via:
  - top_inputs:              summaries of the K inputs that map here.
  - activation_centroid:     mean hidden state over those inputs.
  - next_codes_distribution: softmax of the π-head row for this code.
  - n_samples_mapped:        count of inputs whose argmax is this code.

Consumed by the HTML renderer (interpret.visualise) and by the gate
(gate-interp-passed).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def _summarise(x: Tensor) -> dict:
    """Compact summary of a single input row: mean, norm, argmax_dim."""
    return {
        "mean":       float(x.mean().item()),
        "norm":       float(x.norm().item()),
        "argmax_dim": int(x.argmax().item()),
    }


def build_semantics_table(
    wml,
    inputs: Tensor,
    *,
    top_k_inputs: int = 3,
    alphabet_size: int = 64,
) -> dict[int, dict]:
    """Extract a semantics table for every code in [0, alphabet_size)."""
    wml.eval()
    with torch.no_grad():
        # Forward pass.
        h      = wml.core(inputs)                          # [N, d_hidden]
        logits = wml.emit_head_pi(h)                       # [N, alphabet_size]
        codes  = logits.argmax(dim=-1)                     # [N]

        # Pre-compute softmax of each row of the emit_head_pi weight matrix
        # so `next_codes_distribution` for code c is the softmax of the vector
        # that would drive the π head if c were a template hidden state.
        # Shortcut: use the output weight directly.
        W = wml.emit_head_pi.weight  # noqa: N806 (matrix convention)
        next_dist = F.softmax(W @ W.T, dim=-1)             # [alphabet_size, alphabet_size]

    table: dict[int, dict] = {}
    for c in range(alphabet_size):
        mask = codes == c
        n = int(mask.sum().item())
        if n == 0:
            table[c] = {
                "top_inputs":              [],
                "activation_centroid":     torch.zeros(h.shape[1]),
                "next_codes_distribution": next_dist[c].detach().clone(),
                "n_samples_mapped":        0,
            }
            continue

        mapped_inputs = inputs[mask]
        mapped_h      = h[mask]

        # Top-K by π-head score on that code (most "confident" inputs).
        mapped_scores = logits[mask, c]
        k = min(top_k_inputs, n)
        top_k_idx = mapped_scores.topk(k).indices
        top_inputs = [_summarise(mapped_inputs[i]) for i in top_k_idx.tolist()]

        table[c] = {
            "top_inputs":              top_inputs,
            "activation_centroid":     mapped_h.mean(dim=0),
            "next_codes_distribution": next_dist[c].detach().clone(),
            "n_samples_mapped":        n,
        }
    return table
