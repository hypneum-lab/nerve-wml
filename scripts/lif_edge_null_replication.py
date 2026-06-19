"""Null-controlled replication of the LIF spike-expressivity edge.

Run on kxkm-ai per compute policy (x86 Linux, RTX 4090, actual compute
device: CPU — the committed drivers use ``torch.Generator()`` without
CUDA device, so all tensor ops remain on CPU even on CUDA hardware).

What this script tests
----------------------
LIF vs MLP accuracy edge on ``HardFlowProxyTask`` (12-class XOR-on-noise)
across pool sizes N ∈ {2, 16, 32, 64}, n=50 seeds:

  real edge   = acc_lif - acc_mlp  (LIF advantage if > 0)
  null edge   = acc_mlp_B - acc_mlp_A  (two same-arch MLP cohorts,
                init-seed difference only, same pipeline)

The MLP-vs-MLP null mirrors the exact train/eval pipeline of the
committed ``run_w2_hard`` / ``_run_w2_hard_scale`` functions —
same task seeds, same step budgets, same pool construction — but
replaces LIF slots with a second MLP cohort seeded at ``seed + 10``
(the same offset used for LIF in the real run). This isolates the
init-variance band attributable to different random initializations
from the same architecture, providing a baseline against which the
real heterogeneous-substrate (LIF-vs-MLP) edge is compared.

Result artifact: ``papers/paper1/figures/lif_edge_null_n50.json``

Findings (kxkm run, 2026-06-19):
- Pooled sign-rate: 173/200 favour LIF across all N.
- Bootstrap CI95 on the per-seed signed edge excludes 0 at every N.
- MLP-vs-MLP null is symmetric around 0 (sign-rate ≈ coin-flip).
- Real edge is 2.7×–5.4× the null mean |edge| at pool scale.
- Effect is cleanest at N≥16; at N=2 the null has a fat right tail
  (real_edge_exceeds_null_p95=False) — single seeds can match by
  init luck, but the pooled direction holds (46/50 positive seeds).
- Compute: x86 Linux CPU (cross-arch replication of macM1 origin).

Usage (on kxkm-ai)::

    uv run python scripts/lif_edge_null_replication.py

The script writes no output file locally — the result JSON is the
pre-committed ``papers/paper1/figures/lif_edge_null_n50.json``.
Re-running reproduces the statistics up to bootstrap-resample variance
(n_boot=10000, seed=12345).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

# Import the committed LIF-vs-MLP drivers directly (no edits).
from track_w_pilot import _run_w2_hard_scale, run_w2_hard

from nerve_wml.methodology import (
    accuracy_edge_bootstrap_ci,
    same_substrate_null_result,
)
from track_w._surrogate import spike_with_surrogate  # noqa: F401  (parity import)
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.pool_factory import build_pool, k_for_n
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
from track_w.training import train_wml_on_task

N_SEEDS = 50
SEEDS = list(range(N_SEEDS))

# Step budgets matching the committed multiseed wrappers.
STEPS: dict[int, int] = {2: 800, 16: 400, 32: 200, 64: 150}


# ---------------------------------------------------------------------------
# NULL CONTROL — MLP vs MLP, same substrate, different init seed only.
# ---------------------------------------------------------------------------


def _null_w2_hard_n2(seed: int) -> dict[str, float]:
    """Mirror run_w2_hard but build two MlpWMLs instead of MLP + LIF.

    MLP-A uses init seed = ``seed`` (matches run_w2_hard's MLP seed).
    MLP-B uses init seed = ``seed + 10`` (matches run_w2_hard's LIF
    ``+10`` offset), so the null reproduces exactly the init-seed
    separation that distinguishes the LIF instance from the MLP
    instance in the real experiment.

    Each MLP is trained with ``train_wml_on_task`` on a fresh task
    instance with the same task_seed (cross-cohort RNG isolation, as
    in ``run_w2_hard``), then both are evaluated on a fresh eval task.
    Edge = acc_B - acc_A.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    # Cohort A — fresh task, MLP seeded with ``seed`` (mirrors real MLP path).
    task_a = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp_a = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp_a, nerve, task_a, steps=STEPS[2], lr=1e-2)

    # Cohort B — fresh task (same seed), MLP seeded with ``seed + 10``
    # (mirrors the real LIF path's seed offset and fresh-task isolation).
    task_b = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp_b = MlpWML(id=0, d_hidden=16, seed=seed + 10)
    train_wml_on_task(mlp_b, nerve, task_b, steps=STEPS[2], lr=1e-2)

    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = task_eval.sample(batch=512)
    with torch.no_grad():
        pred_a = mlp_a.emit_head_pi(mlp_a.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_a = (pred_a == y).float().mean().item()
        pred_b = mlp_b.emit_head_pi(mlp_b.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_b = (pred_b == y).float().mean().item()

    return {"acc_a": acc_a, "acc_b": acc_b, "edge": acc_b - acc_a}


def _null_w2_hard_scale(n_wmls: int, seed: int) -> dict[str, float]:
    """Mirror _run_w2_hard_scale but with an all-MLP pool split into two cohorts.

    The real function builds a half-MLP / half-LIF pool (even ids → MLP
    via ``train_wml_on_task``, odd ids → LIF via spike pipeline) and
    compares cohort means.  The null keeps the SAME pool construction
    (``build_pool``, same per-WML seeds ``seed * 1000 + id``), but turns
    every odd (LIF) slot into an MLP with the SAME per-WML seed and trains
    it with ``train_wml_on_task`` too.  Cohort A = even ids, cohort B =
    odd ids.  Both trained on fresh same-seed task instances, evaluated on
    a fresh eval task.  Edge = mean(B) - mean(A).

    This isolates init-variance between two same-architecture MLP cohorts
    at the identical pool scale and step budget as the real LIF-vs-MLP
    comparison.
    """
    steps = STEPS[n_wmls]
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    # Recover per-WML seeds from the real heterogeneous pool construction.
    ref_pool = build_pool(n_wmls=n_wmls, mlp_frac=0.5, seed=seed)
    cohort_a: list[MlpWML] = []
    cohort_b: list[MlpWML] = []
    for w in ref_pool:
        wml_seed = seed * 1000 + w.id
        twin = MlpWML(id=w.id, d_hidden=16, seed=wml_seed)
        if w.id % 2 == 0:
            cohort_a.append(twin)
        else:
            cohort_b.append(twin)

    # Cohort A — fresh task (mirrors MLP cohort path).
    task_a = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in cohort_a:
        train_wml_on_task(w, nerve, task_a, steps=steps, lr=1e-2)

    # Cohort B — fresh task, same seed (mirrors LIF cohort fresh-task isolation).
    task_b = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in cohort_b:
        train_wml_on_task(w, nerve, task_b, steps=steps, lr=1e-2)

    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = task_eval.sample(batch=512)
    accs_a: list[float] = []
    accs_b: list[float] = []
    with torch.no_grad():
        for w in cohort_a:
            pred = w.emit_head_pi(w.core(x))[:, : task_eval.n_classes].argmax(-1)
            accs_a.append((pred == y).float().mean().item())
        for w in cohort_b:
            pred = w.emit_head_pi(w.core(x))[:, : task_eval.n_classes].argmax(-1)
            accs_b.append((pred == y).float().mean().item())

    mean_a = float(np.mean(accs_a))
    mean_b = float(np.mean(accs_b))
    return {"acc_a": mean_a, "acc_b": mean_b, "edge": mean_b - mean_a}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    results: dict = {
        "meta": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "actual_compute_device": str(torch.empty(1).device),
            "n_seeds": N_SEEDS,
            "steps_per_n": STEPS,
        },
        "lif_vs_mlp": {},
        "mlp_vs_mlp_null": {},
        "verdict": {},
    }

    for n in (2, 16, 32, 64):
        print(f"=== N={n} : LIF vs MLP (real) ===", flush=True)
        real_edges: list[float] = []
        real_mlp: list[float] = []
        real_lif: list[float] = []
        for s in SEEDS:
            if n == 2:
                r = run_w2_hard(steps=STEPS[2], seed=s)
            else:
                r = _run_w2_hard_scale(n_wmls=n, steps=STEPS[n], seed=s)
            acc_mlp = r["mean_acc_mlp"]
            acc_lif = r["mean_acc_lif"]
            real_mlp.append(acc_mlp)
            real_lif.append(acc_lif)
            real_edges.append(acc_lif - acc_mlp)
            if s % 10 == 0:
                print(
                    f"  seed {s}: mlp={acc_mlp:.4f} lif={acc_lif:.4f} "
                    f"edge={acc_lif - acc_mlp:+.4f}",
                    flush=True,
                )

        real_ci = accuracy_edge_bootstrap_ci(real_edges)
        results["lif_vs_mlp"][str(n)] = {
            "n": real_ci.n,
            "mean_edge": real_ci.mean_edge,
            "median_edge": real_ci.median_edge,
            "sign_rate_pos": real_ci.sign_rate_pos,
            "sign_rate_neg": real_ci.sign_rate_neg,
            "ci95_lo": real_ci.ci95_low,
            "ci95_hi": real_ci.ci95_high,
            "ci_excludes_zero": real_ci.ci_excludes_zero,
            "mean_acc_mlp": float(np.mean(real_mlp)),
            "mean_acc_lif": float(np.mean(real_lif)),
        }
        print(
            f"  -> mean edge {real_ci.mean_edge:+.4f}  "
            f"sign+ {real_ci.sign_rate_pos}/{N_SEEDS}  "
            f"CI95 [{real_ci.ci95_low:+.4f}, {real_ci.ci95_high:+.4f}]  "
            f"excl0={real_ci.ci_excludes_zero}",
            flush=True,
        )

        print(f"=== N={n} : MLP vs MLP (null) ===", flush=True)
        null_edges: list[float] = []
        for s in SEEDS:
            if n == 2:
                rn = _null_w2_hard_n2(seed=s)
            else:
                rn = _null_w2_hard_scale(n_wmls=n, seed=s)
            null_edges.append(rn["edge"])

        null_summ = same_substrate_null_result(null_edges)
        results["mlp_vs_mlp_null"][str(n)] = {
            "n": null_summ.n,
            "mean_edge": null_summ.mean_edge,
            "mean_abs_edge": null_summ.mean_abs_edge,
            "abs_p95": null_summ.abs_p95,
            "sign_rate_pos": null_summ.sign_rate_pos,
            "sign_rate_neg": null_summ.sign_rate_neg,
            "ci95_lo": null_summ.ci95_low,
            "ci95_hi": null_summ.ci95_high,
            "ci_excludes_zero": null_summ.ci_excludes_zero,
        }
        print(
            f"  -> null mean|edge| {null_summ.mean_abs_edge:.4f}  "
            f"abs_p95 {null_summ.abs_p95:.4f}  "
            f"signedCI95 [{null_summ.ci95_low:+.4f}, {null_summ.ci95_high:+.4f}]  "
            f"symm={not null_summ.ci_excludes_zero}",
            flush=True,
        )

        # Verdict: does real edge exceed the null's init-variance band?
        real_ci_lo = results["lif_vs_mlp"][str(n)]["ci95_lo"]
        real_mean = results["lif_vs_mlp"][str(n)]["mean_edge"]
        results["verdict"][str(n)] = {
            "real_mean_edge": real_mean,
            "real_sign_pos": real_ci.sign_rate_pos,
            "real_ci95": [real_ci.ci95_low, real_ci.ci95_high],
            "real_ci_excludes_zero": real_ci.ci_excludes_zero,
            "null_mean_abs_edge": null_summ.mean_abs_edge,
            "null_abs_p95": null_summ.abs_p95,
            "real_edge_exceeds_null_p95": bool(real_mean > null_summ.abs_p95),
            "real_ci_lo_above_null_p95": bool(real_ci_lo > null_summ.abs_p95),
        }

    results["meta"]["elapsed_sec"] = time.time() - t0

    # Compact verdict table.
    print("\n===== VERDICT TABLE =====", flush=True)
    print(
        f"{'N':>4} {'LIF edge':>10} {'sign+/50':>9} {'CI95':>22} "
        f"{'null|e|':>9} {'null p95':>9} {'>null?':>7} {'CIexcl0':>8}",
        flush=True,
    )
    for n in (2, 16, 32, 64):
        v = results["verdict"][str(n)]
        ci = f"[{v['real_ci95'][0]:+.3f},{v['real_ci95'][1]:+.3f}]"
        print(
            f"{n:>4} {v['real_mean_edge']:>+10.4f} {v['real_sign_pos']:>7}/50 "
            f"{ci:>22} {v['null_mean_abs_edge']:>9.4f} {v['null_abs_p95']:>9.4f} "
            f"{str(v['real_ci_lo_above_null_p95']):>7} "
            f"{str(v['real_ci_excludes_zero']):>8}",
            flush=True,
        )

    print(f"\nTotal elapsed: {results['meta']['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
