"""MNIST pilots — real-data validation of Claims A and B.

Uses WmlConfig.mnist() (input_dim=784, d_hidden=128, alphabet=256)
and the same RNG isolation discipline as run_w2_hard: fresh task
instance per cohort, shared eval instance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.configs.wml_config import WmlConfig  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.tasks.mnist import MNISTTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402


def _train_mnist_pair(seed: int, steps: int) -> tuple:
    """Train one MLP and one LIF on MNIST with RNG isolation per cohort."""
    cfg = WmlConfig.mnist()
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = MNISTTask(seed=seed)
    mlp = MlpWML(
        id=0, input_dim=cfg.input_dim, d_hidden=cfg.d_hidden,
        alphabet_size=cfg.alphabet_size, seed=seed,
    )
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=5e-3)

    task_lif = MNISTTask(seed=seed)
    lif = LifWML(
        id=0, input_dim=cfg.input_dim, n_neurons=cfg.n_neurons,
        alphabet_size=cfg.alphabet_size, seed=seed + 10,
    )
    opt = torch.optim.Adam(lif.parameters(), lr=5e-3)
    for _ in range(steps):
        x, y = task_lif.sample(batch=64)
        i_in = lif.input_proj(x)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mlp, lif


def run_mnist_polymorphism(
    n_wmls: int = 16, steps: int = 300, seeds: list[int] | None = None,
) -> dict:
    """Per-seed MLP/LIF accuracy comparison on MNIST.

    Note: `n_wmls` is a legacy parameter from the synthetic pilots; this
    MNIST variant trains one MLP and one LIF per seed. A full pool
    version would follow `_run_w2_hard_scale` and is deferred.
    """
    del n_wmls  # unused but accepted for test-contract compatibility
    if seeds is None:
        seeds = [0]
    gaps, accs_mlp, accs_lif = [], [], []
    for s in seeds:
        mlp, lif = _train_mnist_pair(seed=s, steps=steps)
        task = MNISTTask(seed=s)
        x, y = task.sample(batch=512)
        with torch.no_grad():
            pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : task.n_classes].argmax(-1)
            acc_mlp = (pred_mlp == y).float().mean().item()
            i_in = lif.input_proj(x)
            spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
            pred_lif = lif.emit_head_pi(spikes)[:, : task.n_classes].argmax(-1)
            acc_lif = (pred_lif == y).float().mean().item()
        gaps.append(abs(acc_mlp - acc_lif) / max(acc_mlp, 1e-6))
        accs_mlp.append(acc_mlp)
        accs_lif.append(acc_lif)
    return {
        "seeds":        list(seeds),
        "gaps":         gaps,
        "accs_mlp":     accs_mlp,
        "accs_lif":     accs_lif,
        "mean_gap":     float(np.mean(gaps)),
        "median_gap":   float(np.median(gaps)),
        "max_gap":      float(np.max(gaps)),
        "mean_acc_mlp": float(np.mean(accs_mlp)),
        "mean_acc_lif": float(np.mean(accs_lif)),
    }


def run_mnist_mi(seeds=None, steps=300, batch=1024) -> list:
    """MI(codes_MLP, codes_LIF) / H(codes_MLP) on MNIST."""
    from scripts.measure_info_transmission import mutual_info_score

    if seeds is None:
        seeds = [0, 1, 2]
    results = []
    for s in seeds:
        mlp, lif = _train_mnist_pair(seed=s, steps=steps)
        task = MNISTTask(seed=s)
        x, _ = task.sample(batch=batch)
        with torch.no_grad():
            codes_mlp = mlp.emit_head_pi(mlp.core(x)).argmax(-1).numpy()
            i_in = lif.input_proj(x)
            spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
            codes_lif = lif.emit_head_pi(spikes).argmax(-1).numpy()
        mi = mutual_info_score(codes_mlp, codes_lif)
        _, c = np.unique(codes_mlp, return_counts=True)
        h_mlp = float(-(c / c.sum() * np.log(c / c.sum())).sum())
        results.append({
            "seed":      s,
            "mi":        float(mi),
            "h_mlp":     h_mlp,
            "mi_over_h": float(mi / max(h_mlp, 1e-9)),
        })
    return results
