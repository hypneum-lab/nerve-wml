"""Bigger-architecture pilot — d_hidden=128, alphabet=256, N=16.

Verifies the N=16/N=32 plateau observed on d_hidden=16 is not an
artefact of tiny networks. Uses HardFlowProxyTask to stay comparable
with the v1.1 scaling law.
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
from track_w.pool_factory import build_pool_cfg, k_for_n  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402


def _bigger_cfg() -> WmlConfig:
    return WmlConfig(
        input_dim=16, d_hidden=128, n_neurons=128,
        d_model=128, n_heads=4, n_tokens=8, alphabet_size=256,
    )


def run_hard_bigger_arch(
    n_wmls: int = 16, steps: int = 300, seed: int = 0,
) -> dict:
    cfg = _bigger_cfg()
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool_cfg(n_wmls=n_wmls, cfg=cfg, seed=seed)

    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in pool:
        if isinstance(w, MlpWML):
            train_wml_on_task(w, nerve, task_mlp, steps=steps, lr=5e-3)

    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in pool:
        if isinstance(w, LifWML):
            opt = torch.optim.Adam(w.parameters(), lr=5e-3)
            for _ in range(steps):
                x, y = task_lif.sample(batch=64)
                i_in = w.input_proj(x)
                spikes = spike_with_surrogate(i_in, v_thr=w.v_thr)
                logits = w.emit_head_pi(spikes)[:, : task_lif.n_classes]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = task_eval.sample(batch=512)
    accs_mlp, accs_lif = [], []
    with torch.no_grad():
        for w in pool:
            if isinstance(w, MlpWML):
                pred = w.emit_head_pi(w.core(x))[
                    :, : task_eval.n_classes
                ].argmax(-1)
                accs_mlp.append((pred == y).float().mean().item())
            elif isinstance(w, LifWML):
                i_in = w.input_proj(x)
                spikes = spike_with_surrogate(i_in, v_thr=w.v_thr)
                pred = w.emit_head_pi(spikes)[
                    :, : task_eval.n_classes
                ].argmax(-1)
                accs_lif.append((pred == y).float().mean().item())
    mean_mlp = float(np.mean(accs_mlp))
    mean_lif = float(np.mean(accs_lif))
    return {
        "mean_acc_mlp": mean_mlp,
        "mean_acc_lif": mean_lif,
        "gap":          abs(mean_mlp - mean_lif) / max(mean_mlp, 1e-6),
    }


def run_hard_bigger_arch_multiseed(
    seeds: list[int] | None = None, steps: int = 300,
) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]
    per = [run_hard_bigger_arch(steps=steps, seed=s) for s in seeds]
    gaps = [r["gap"] for r in per]
    return {
        "seeds":      list(seeds),
        "gaps":       gaps,
        "median_gap": float(np.median(gaps)),
        "max_gap":    float(np.max(gaps)),
        "mean_gap":   float(np.mean(gaps)),
    }


def run_mi_bigger_arch(
    seeds=None, steps: int = 300, batch: int = 1024,
) -> list:
    """MI between MLP and LIF emissions at d_hidden=128."""
    from scripts.measure_info_transmission import mutual_info_score

    cfg = _bigger_cfg()
    if seeds is None:
        seeds = [0, 1, 2]
    results = []
    for s in seeds:
        torch.manual_seed(s)
        nerve = MockNerve(n_wmls=2, k=1, seed=s)
        nerve.set_phase_active(gamma=True, theta=False)
        mlp = MlpWML(
            id=0, input_dim=cfg.input_dim, d_hidden=cfg.d_hidden,
            alphabet_size=cfg.alphabet_size, seed=s,
        )
        task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=s)
        train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=5e-3)

        lif = LifWML(
            id=0, input_dim=cfg.input_dim, n_neurons=cfg.n_neurons,
            alphabet_size=cfg.alphabet_size, seed=s + 10,
        )
        opt = torch.optim.Adam(lif.parameters(), lr=5e-3)
        task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=s)
        for _ in range(steps):
            x, y = task_lif.sample(batch=64)
            i_in = lif.input_proj(x)
            spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
            logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=s)
        x, _ = task_eval.sample(batch=batch)
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
