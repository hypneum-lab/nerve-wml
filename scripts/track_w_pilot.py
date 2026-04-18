"""Track-W pilot scripts: W1-W4 curriculum drivers + Gate W aggregator."""
from __future__ import annotations

import torch

from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.training import train_wml_on_task


def run_w1(steps: int = 400) -> float:
    """W1 — train two MlpWMLs on FlowProxyTask; return accuracy of WML 0."""
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wmls  = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    for wml in wmls:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # Evaluate WML 0 by classifying via π head.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h = wmls[0].core(x)
        pred = wmls[0].emit_head_pi(h)[:, : task.n_classes].argmax(-1)
    return (pred == y).float().mean().item()
