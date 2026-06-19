"""Null-controlled replication: orthogonality (arch-scale) + scaling plateau.

Run on kxkm-ai per compute policy (x86 Linux, CPU compute — the committed
drivers use ``torch.Generator()`` without CUDA device, so all tensor ops
remain on CPU even on CUDA hardware).

What this script replicates
---------------------------
Two surviving findings from the null-control campaign (2026-06-20):

(a) SCALING PLATEAU — already covered by the committed
    ``scripts/lif_edge_null_replication.py`` (artifact
    ``papers/paper1/figures/lif_edge_null_n50.json``).  This script
    re-reads that artifact and prints the N=64 plateau summary so the
    result is cross-referenced in one place.  Pass ``--rerun-scaling``
    to force a fresh 50-seed run instead of reading the committed JSON.

(b) ORTHOGONALITY (arch-scale vs pool-scale) — the committed
    ``scripts/run_bigger_arch.py`` reports a ~26% relative gap at
    d_hidden=128 on 3 seeds with NO null.  Here, at n=50 seeds, we run:
      real signed edge   = mean_acc_lif - mean_acc_mlp  (per seed)
      MLP-vs-MLP null    = mean_acc_B - mean_acc_A      (two same-arch
                           MLP cohorts at the same d_hidden, mirroring
                           _null_w2_hard_scale in the LIF replication)
    CI via ``accuracy_edge_bootstrap_ci``; null via
    ``same_substrate_null_result``.

(d) N × d CORNER CHECK — pool scale (N ∈ {2, 64}) × arch width
    (d ∈ {16, 128}) relative-gap grid, showing the two axes move the gap
    independently.

Committed artifacts
-------------------
- ``papers/paper1/figures/orthogonality_null_n50.json``
  — heavy orthogonality phase (50 seeds, d ∈ {16, 128}, N=16)
- ``papers/paper1/figures/ortho_corner_check_n10.json``
  — N×d corner check (10 seeds)
- ``papers/paper1/figures/lif_edge_null_n50.json``
  — scaling plateau (50 seeds, N ∈ {2, 16, 32, 64})

Scope: substrate-INTRINSIC arch-scale effect on HardFlowProxyTask.
NOT agnostic/SOTA. Never fabricate.

Usage (on kxkm-ai, from repo root)::

    uv run python scripts/orthogonality_null_replication.py
    uv run python scripts/orthogonality_null_replication.py --rerun-scaling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402, N812

# Import committed driver directly — no duplication of null logic.
from track_w_pilot import _run_w2_hard_scale  # noqa: E402  # type: ignore[import-untyped]

from nerve_wml.methodology import (  # noqa: E402
    accuracy_edge_bootstrap_ci,
    same_substrate_null_result,
)
from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.configs.wml_config import WmlConfig  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.pool_factory import build_pool_cfg, k_for_n  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402

ROOT = _ROOT

# ---------------------------------------------------------------------------
# Helpers mirroring the ortho_driver (no inline duplication of null logic).
# ---------------------------------------------------------------------------

STEPS_ORTHO = 300
N_WMLS_ORTHO = 16

# Step budgets for scaling rerun (mirrors lif_edge_null_replication.py).
STEPS_SCALE: dict[int, int] = {2: 800, 16: 400, 32: 200, 64: 150}


def _cfg(d_hidden: int) -> WmlConfig:
    """Mirror _bigger_cfg from run_bigger_arch (d_hidden=128 default)."""
    return WmlConfig(
        input_dim=16,
        d_hidden=d_hidden,
        n_neurons=d_hidden,
        d_model=128,
        n_heads=4,
        n_tokens=8,
        alphabet_size=256,
    )


def _k_for(n_wmls: int) -> int:
    """k_for_n clamped to SparseRouter constraint 1 <= k < n_wmls."""
    return max(1, min(k_for_n(n_wmls), n_wmls - 1))


# ---------------------------------------------------------------------------
# Real arch run — LIF vs MLP at given width.
# ---------------------------------------------------------------------------


def _real_arch_run(n_wmls: int, d_hidden: int, steps: int, seed: int) -> dict:
    """LIF-vs-MLP at given width, mirroring run_hard_bigger_arch exactly."""
    cfg = _cfg(d_hidden)
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=_k_for(n_wmls), seed=seed)
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
                pred = w.emit_head_pi(w.core(x))[:, : task_eval.n_classes].argmax(-1)
                accs_mlp.append((pred == y).float().mean().item())
            elif isinstance(w, LifWML):
                i_in = w.input_proj(x)
                spikes = spike_with_surrogate(i_in, v_thr=w.v_thr)
                pred = w.emit_head_pi(spikes)[:, : task_eval.n_classes].argmax(-1)
                accs_lif.append((pred == y).float().mean().item())

    mean_mlp = float(np.mean(accs_mlp))
    mean_lif = float(np.mean(accs_lif))
    edge = mean_lif - mean_mlp
    return {
        "mean_acc_mlp": mean_mlp,
        "mean_acc_lif": mean_lif,
        "edge": edge,
        "gap_rel": abs(mean_mlp - mean_lif) / max(mean_mlp, 1e-6),
    }


# ---------------------------------------------------------------------------
# Null arch run — MLP-vs-MLP same-substrate null at given width.
# ---------------------------------------------------------------------------


def _null_arch_run(n_wmls: int, d_hidden: int, steps: int, seed: int) -> dict:
    """MLP-vs-MLP same-substrate null, mirroring the LIF replication null."""
    cfg = _cfg(d_hidden)
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=_k_for(n_wmls), seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    ref_pool = build_pool_cfg(n_wmls=n_wmls, cfg=cfg, seed=seed)
    cohort_a: list[MlpWML] = []
    cohort_b: list[MlpWML] = []
    for w in ref_pool:
        wml_seed = seed * 1000 + w.id
        twin = MlpWML(
            id=w.id,
            input_dim=cfg.input_dim,
            d_hidden=cfg.d_hidden,
            alphabet_size=cfg.alphabet_size,
            seed=wml_seed,
        )
        if w.id % 2 == 0:
            cohort_a.append(twin)
        else:
            cohort_b.append(twin)

    task_a = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in cohort_a:
        train_wml_on_task(w, nerve, task_a, steps=steps, lr=5e-3)

    task_b = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in cohort_b:
        train_wml_on_task(w, nerve, task_b, steps=steps, lr=5e-3)

    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = task_eval.sample(batch=512)
    accs_a, accs_b = [], []
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
# Orthogonality sweep.
# ---------------------------------------------------------------------------


def _run_orthogonality(seeds: list[int], steps: int, widths: list[int]) -> dict:
    """(b) LIF-MLP edge + MLP-MLP null for each d in widths at N=N_WMLS_ORTHO."""
    out: dict = {}
    for d in widths:
        print(f"\n=== ORTHOGONALITY  d_hidden={d}  N={N_WMLS_ORTHO} ===", flush=True)
        real_edges, real_gaps = [], []
        for s in seeds:
            r = _real_arch_run(
                n_wmls=N_WMLS_ORTHO, d_hidden=d, steps=steps, seed=s
            )
            real_edges.append(r["edge"])
            real_gaps.append(r["gap_rel"])
            if s % 10 == 0:
                print(
                    f"  [real] seed {s}: mlp={r['mean_acc_mlp']:.4f} "
                    f"lif={r['mean_acc_lif']:.4f} edge={r['edge']:+.4f}",
                    flush=True,
                )

        real_ci = accuracy_edge_bootstrap_ci(real_edges)

        null_edges = []
        for s in seeds:
            rn = _null_arch_run(
                n_wmls=N_WMLS_ORTHO, d_hidden=d, steps=steps, seed=s
            )
            null_edges.append(rn["edge"])

        null_summ = same_substrate_null_result(null_edges)
        v = {
            "real_mean_abs_edge": abs(real_ci.mean_edge),
            "real_ci_excludes_zero": real_ci.ci_excludes_zero,
            "null_mean_abs_edge": null_summ.mean_abs_edge,
            "null_abs_p95": null_summ.abs_p95,
            "real_abs_edge_exceeds_null_p95": bool(
                abs(real_ci.mean_edge) > null_summ.abs_p95
            ),
            "real_over_null_ratio": (
                abs(real_ci.mean_edge) / null_summ.mean_abs_edge
                if null_summ.mean_abs_edge > 0
                else None
            ),
            "null_symmetric": not null_summ.ci_excludes_zero,
        }
        out[str(d)] = {
            "real_signed_edge": {
                "mean_abs_edge": abs(real_ci.mean_edge),
                "ci95_lo": real_ci.ci95_low,
                "ci95_hi": real_ci.ci95_high,
                "ci_excludes_zero": real_ci.ci_excludes_zero,
                "sign_rate_pos": real_ci.sign_rate_pos,
            },
            "real_relative_gap_mean": float(np.mean(real_gaps)),
            "mlp_vs_mlp_null": {
                "mean_abs_edge": null_summ.mean_abs_edge,
                "abs_p95": null_summ.abs_p95,
                "symmetric": not null_summ.ci_excludes_zero,
            },
            "verdict": v,
        }
        print(
            f"  -> d={d}: |edge|={abs(real_ci.mean_edge):.4f} "
            f"CI95 [{real_ci.ci95_low:+.4f},{real_ci.ci95_high:+.4f}] "
            f"excl0={real_ci.ci_excludes_zero} sign+={real_ci.sign_rate_pos}/{len(seeds)} "
            f"null|e|={null_summ.mean_abs_edge:.4f} p95={null_summ.abs_p95:.4f} "
            f"ratio={v['real_over_null_ratio']}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# N × d corner check.
# ---------------------------------------------------------------------------


def _run_corner(seeds: list[int], steps: int) -> dict:
    """(d) N × d corner check — relative-gap interaction probe."""
    out: dict = {}
    for n_wmls in (2, 64):
        for d in (16, 128):
            key = f"N{n_wmls}_d{d}"
            gaps = []
            for s in seeds:
                r = _real_arch_run(
                    n_wmls=n_wmls, d_hidden=d, steps=steps, seed=s
                )
                gaps.append(r["gap_rel"])
            arr = np.asarray(gaps)
            out[key] = {
                "n_wmls": n_wmls,
                "d_hidden": d,
                "n_seeds": len(seeds),
                "gap_rel_mean": float(arr.mean()),
                "gap_rel_median": float(np.median(arr)),
                "gap_rel_std": float(arr.std()),
            }
            print(
                f"  corner {key}: gap_rel mean={arr.mean():.4f} "
                f"median={np.median(arr):.4f}",
                flush=True,
            )
    return out


# ---------------------------------------------------------------------------
# Scaling plateau — read from committed JSON or rerun.
# ---------------------------------------------------------------------------


def _scaling_plateau_summary(rerun: bool, seeds: list[int]) -> dict:
    """Return N=64 plateau summary from the committed artifact or a fresh run."""
    artifact = (
        ROOT / "papers" / "paper1" / "figures" / "lif_edge_null_n50.json"
    )
    if not rerun and artifact.exists():
        data = json.loads(artifact.read_text())
        v64 = data.get("verdict", {}).get("64", {})
        null64 = data.get("mlp_vs_mlp_null", {}).get("64", {})
        return {
            "source": "committed_lif_edge_null_n50.json",
            "n64_real_mean_edge": v64.get("real_mean_edge"),
            "n64_real_ci95": v64.get("real_ci95"),
            "n64_real_ci_excludes_zero": v64.get("real_ci_excludes_zero"),
            "n64_real_sign_pos": v64.get("real_sign_pos"),
            "n64_null_mean_abs_edge": null64.get("mean_abs_edge"),
            "n64_null_abs_p95": null64.get("abs_p95"),
            "n64_real_edge_exceeds_null_p95": v64.get("real_edge_exceeds_null_p95"),
        }

    # Fresh rerun — mirrors lif_edge_null_replication.py for N=64.
    print("=== SCALING PLATEAU N=64 (rerun) ===", flush=True)
    real_edges, null_edges = [], []
    for s in seeds:
        r = _run_w2_hard_scale(n_wmls=64, steps=STEPS_SCALE[64], seed=s)
        real_edges.append(r["mean_acc_lif"] - r["mean_acc_mlp"])

    # Null mirrors _null_w2_hard_scale in lif_edge_null_replication.
    from lif_edge_null_replication import _null_w2_hard_scale  # type: ignore[import-untyped]

    for s in seeds:
        rn = _null_w2_hard_scale(n_wmls=64, seed=s)
        null_edges.append(rn["edge"])

    real_ci = accuracy_edge_bootstrap_ci(real_edges)
    null_summ = same_substrate_null_result(null_edges)
    return {
        "source": "fresh_rerun",
        "n64_real_mean_edge": real_ci.mean_edge,
        "n64_real_ci95": [real_ci.ci95_low, real_ci.ci95_high],
        "n64_real_ci_excludes_zero": real_ci.ci_excludes_zero,
        "n64_real_sign_pos": real_ci.sign_rate_pos,
        "n64_null_mean_abs_edge": null_summ.mean_abs_edge,
        "n64_null_abs_p95": null_summ.abs_p95,
        "n64_real_edge_exceeds_null_p95": bool(
            real_ci.mean_edge > null_summ.abs_p95
        ),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orthogonality + scaling-plateau null-control replication."
    )
    ap.add_argument(
        "--seeds", type=int, default=50, help="Number of seeds for orthogonality."
    )
    ap.add_argument(
        "--steps", type=int, default=300, help="Training steps per seed."
    )
    ap.add_argument(
        "--corner-seeds",
        type=int,
        default=10,
        help="Seeds for N×d corner check.",
    )
    ap.add_argument(
        "--rerun-scaling",
        action="store_true",
        help="Force fresh scaling-plateau run instead of reading the committed JSON.",
    )
    ap.add_argument(
        "--corner-only",
        action="store_true",
        help="Run only the N×d corner check (skip expensive orthogonality phase).",
    )
    args = ap.parse_args()

    seeds_ortho = list(range(args.seeds))
    seeds_corner = list(range(args.corner_seeds))

    t0 = time.time()
    report: dict = {}

    # (a) Scaling plateau summary.
    print("\n=== SCALING PLATEAU ===", flush=True)
    report["scaling_plateau_n64"] = _scaling_plateau_summary(
        rerun=args.rerun_scaling, seeds=seeds_ortho
    )
    s64 = report["scaling_plateau_n64"]
    print(
        f"  N=64 edge={s64['n64_real_mean_edge']:.4f} "
        f"CI95={s64['n64_real_ci95']} excl0={s64['n64_real_ci_excludes_zero']} "
        f"sign+={s64['n64_real_sign_pos']}/{args.seeds} "
        f"null|e|={s64['n64_null_mean_abs_edge']:.4f} "
        f">null_p95={s64['n64_real_edge_exceeds_null_p95']}",
        flush=True,
    )

    # (b) Orthogonality arch-scale.
    if not args.corner_only:
        report["orthogonality_arch_scale"] = _run_orthogonality(
            seeds=seeds_ortho, steps=args.steps, widths=[16, 128]
        )

    # (d) N × d corner check.
    print("\n=== N × d CORNER CHECK ===", flush=True)
    report["corner_check"] = _run_corner(seeds=seeds_corner, steps=args.steps)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s", flush=True)

    # Compact verdict.
    if not args.corner_only and "orthogonality_arch_scale" in report:
        print("\n=== ORTHOGONALITY VERDICT ===", flush=True)
        for d, blk in report["orthogonality_arch_scale"].items():
            v = blk["verdict"]
            print(
                f"  d={d}: |edge|={blk['real_signed_edge']['mean_abs_edge']:.4f} "
                f"ratio={v['real_over_null_ratio']:.2f}× null "
                f"excl0={blk['real_signed_edge']['ci_excludes_zero']} "
                f"symm={v['null_symmetric']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
