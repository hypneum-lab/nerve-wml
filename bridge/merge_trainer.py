"""MergeTrainer — fine-tune only the nerve transducers.

Spec §7.4: swap MockNerve for SimNerve, freeze WML internals and
codebooks, fine-tune only the nerve transducers for ~20 % of Track-W
step count. Gate M asserts merged perf ≥ 95 % of Track-W-with-mock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.optim import Adam


@dataclass
class MergeTrainer:
    wmls:  list[Any]         # pool of MlpWML / LifWML
    nerve: torch.nn.Module   # SimNerveAdapter
    task:  Any               # FlowProxyTask
    steps: int = 100
    lr:    float = 1e-2

    def train(self) -> list[float]:
        """Fine-tune nerve.transducers while WML params are frozen. Return losses."""
        # Freeze WML params.
        for wml in self.wmls:
            for p in wml.parameters():
                p.requires_grad_(False)

        trainable = [p for p in self.nerve.parameters() if p.requires_grad]
        opt = Adam(trainable, lr=self.lr)

        losses: list[float] = []

        for _ in range(self.steps):
            x, y = self.task.sample(batch=64)

            # Route x through WML 0's core (frozen) and take π logits.
            h = self.wmls[0].core(x)
            pi_logits = self.wmls[0].emit_head_pi(h)[:, : self.task.n_classes]
            task_loss = torch.nn.functional.cross_entropy(pi_logits, y)

            # Transducer entropy regulariser to avoid collapse-to-identity.
            ent_reg: Tensor = torch.tensor(0.0)
            for t in self.nerve._transducers.values():  # type: ignore[operator,union-attr]
                ent_reg = ent_reg + t.entropy()  # type: ignore[union-attr,operator]
            reg = -0.01 * ent_reg  # maximise entropy (negative sign)

            total = task_loss + reg
            opt.zero_grad()
            total.backward()
            opt.step()
            losses.append(total.item())

        # Unfreeze WML params after training so downstream code is not surprised.
        for wml in self.wmls:
            for p in wml.parameters():
                p.requires_grad_(True)

        return losses
