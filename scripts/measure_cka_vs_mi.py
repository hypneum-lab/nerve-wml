"""Compare CKA (Kornblith 2019) to our MI/H ratio on the same substrate pairs.

Motivation: our MI/H measurement operates on argmax-discretized emitted
codes (one of 64 / 256 slots). CKA operates on continuous hidden states.
If CKA >> MI/H, we're measuring something stricter than CKA (discrete
code agreement, not just geometric alignment). If CKA ≈ MI/H, we're
reinventing CKA under a different name.

Linear CKA for mean-centered features X [n, p] and Y [n, q]:

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

Values in [0, 1], 1 = identical up to orthogonal transform / scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from scripts.measure_info_transmission import mutual_info_score  # noqa: E402
from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Kornblith 2019 linear CKA (mean-centered)."""
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    num = (xy * xy).sum()
    denom = float(np.sqrt((xx * xx).sum()) * np.sqrt((yy * yy).sum()))
    return float(num / max(denom, 1e-9))


def _train_pair(seed: int, steps: int) -> tuple:
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    enc = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(enc.parameters()), lr=1e-2,
    )
    for _ in range(steps):
        x, y = task_lif.sample(batch=64)
        i_in = lif.input_proj(enc(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mlp, lif, enc


def run_cka_vs_mi(
    seeds: list[int] | None = None,
    steps: int = 400,
    batch: int = 1024,
) -> list:
    """For each seed, measure CKA at three layers + our MI/H ratio.

    Layers probed:
      - pre_emit: (MLP.core output) vs (LIF post-spike features)
      - emit_logits: (MLP.emit_head_pi output) vs (LIF.emit_head_pi output)
      - argmax_codes: one-hot of argmax — our discrete MI surface
    """
    if seeds is None:
        seeds = [0, 1, 2]
    results = []
    for s in seeds:
        mlp, lif, enc = _train_pair(seed=s, steps=steps)
        task = HardFlowProxyTask(dim=16, n_classes=12, seed=s)
        x, _ = task.sample(batch=batch)
        with torch.no_grad():
            h_mlp = mlp.core(x).numpy()                           # [B, d_hidden]
            i_in = lif.input_proj(enc(x))
            spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
            h_lif = spikes.numpy()                                # [B, n_neurons]

            logits_mlp = mlp.emit_head_pi(mlp.core(x)).numpy()    # [B, 64]
            logits_lif = lif.emit_head_pi(spikes).numpy()         # [B, 64]

            codes_mlp = logits_mlp.argmax(axis=1)
            codes_lif = logits_lif.argmax(axis=1)
            alphabet = logits_mlp.shape[1]
            onehot_mlp = np.eye(alphabet)[codes_mlp]
            onehot_lif = np.eye(alphabet)[codes_lif]

        mi = mutual_info_score(codes_mlp, codes_lif)
        _, c = np.unique(codes_mlp, return_counts=True)
        h_mlp_entropy = float(-(c / c.sum() * np.log(c / c.sum())).sum())
        mi_over_h = float(mi / max(h_mlp_entropy, 1e-9))

        cka_pre_emit = linear_cka(h_mlp, h_lif)
        cka_emit_logits = linear_cka(logits_mlp, logits_lif)
        cka_argmax = linear_cka(onehot_mlp, onehot_lif)

        results.append({
            "seed":             s,
            "mi_over_h":        mi_over_h,
            "cka_pre_emit":     cka_pre_emit,
            "cka_emit_logits":  cka_emit_logits,
            "cka_argmax_onehot": cka_argmax,
        })
    return results


def main() -> None:
    print("Running CKA vs MI/H comparison (3 seeds, 400 steps)...")
    r = run_cka_vs_mi(seeds=[0, 1, 2], steps=400, batch=1024)
    print()
    print(f"{'seed':>4} {'MI/H':>6} {'CKA pre-emit':>14} "
          f"{'CKA logits':>12} {'CKA argmax':>12}")
    for rec in r:
        print(f"{rec['seed']:>4} "
              f"{rec['mi_over_h']:>6.3f} "
              f"{rec['cka_pre_emit']:>14.3f} "
              f"{rec['cka_emit_logits']:>12.3f} "
              f"{rec['cka_argmax_onehot']:>12.3f}")
    print()
    avg_mi = np.mean([rec["mi_over_h"] for rec in r])
    avg_pre = np.mean([rec["cka_pre_emit"] for rec in r])
    avg_log = np.mean([rec["cka_emit_logits"] for rec in r])
    avg_arg = np.mean([rec["cka_argmax_onehot"] for rec in r])
    print(f"mean: MI/H={avg_mi:.3f}  CKA_pre={avg_pre:.3f}  "
          f"CKA_logits={avg_log:.3f}  CKA_argmax={avg_arg:.3f}")


if __name__ == "__main__":
    main()
