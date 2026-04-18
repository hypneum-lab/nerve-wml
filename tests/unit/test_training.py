import torch

from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.training import train_wml_on_task


def test_train_wml_improves_accuracy():
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # Baseline accuracy before training.
    pre = _classify_via_pi_head(wml, task, n_samples=128)

    train_wml_on_task(wml, nerve, task, steps=400, lr=1e-2)

    post = _classify_via_pi_head(wml, task, n_samples=128)
    assert post > pre
    assert post > 0.4  # random baseline ~0.25


def _classify_via_pi_head(wml, task, n_samples):
    """Feed task samples through wml.core and read emit_head_pi as classifier."""
    x, y = task.sample(batch=n_samples)
    with torch.no_grad():
        h = wml.core(x)
        logits = wml.emit_head_pi(h)
        pred   = logits.argmax(-1) % task.n_classes
    return (pred == y).float().mean().item()
