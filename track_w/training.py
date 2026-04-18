"""Minimal training loop for Track-W pilots.

train_wml_on_task: drive a WML against a task, using its π head as a
classifier and cross-entropy as the task loss. This is the inner loop
reused by W1-W4 pilots. VQ commitment loss is derived from the WML's
codebook assignments.
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from .losses import composite_loss


def train_wml_on_task(wml, nerve, task, *, steps: int = 500, lr: float = 1e-3) -> list[float]:
    """Train wml's classification head on task; return per-step loss.

    wml       — any module with .core, .codebook, .emit_head_pi
    nerve     — MockNerve (unused for this simple task loss but kept for interface uniformity)
    task      — any object with .n_classes and .sample(batch) → (x, y)
    """
    opt = Adam(wml.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(steps):
        x, y = task.sample(batch=64)

        h      = wml.core(x)
        logits = wml.emit_head_pi(h)
        # Map 64-code logits to task classes by taking the first n_classes columns.
        task_logits = logits[:, : task.n_classes]
        task_loss   = torch.nn.functional.cross_entropy(task_logits, y)

        # VQ commitment loss on the hidden state.
        dist  = torch.cdist(h, wml.codebook)
        idx   = dist.argmin(-1)
        q     = wml.codebook[idx]
        vq_loss = 0.25 * ((h - q.detach()) ** 2).mean() + ((q - h.detach()) ** 2).mean()

        total = composite_loss(task_loss=task_loss, vq_loss=vq_loss)
        opt.zero_grad(); total.backward(); opt.step()

        losses.append(total.item())

    return losses
