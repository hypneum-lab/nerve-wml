"""BioFieldWML readout ablation — does the +0.30 edge survive without |error|?

Critical ablation on the BioFieldWML belief-state readout result. A prior
n=50 run gave BioField (feature ``[mu, sigma, |error|]``, width 48) a huge
edge over the matched MLP on ``HardFlowProxyTask`` (acc 0.8502 vs 0.5549,
edge +0.2953, 50/50, CI95 [0.2745, 0.3150]). But ``|error| = |observed| =
|input_proj(x)|`` (the prior mu is 0), so the readout is fed the raw
absolute input projection directly — plausibly a task-aligned shortcut for
XOR-on-noise rather than genuine (mu, sigma) belief-state expressivity.

This script re-runs the SAME pipeline as ``scripts/biofield_pilot.py``
(same MLP baseline, same MLP-vs-MLP null, same bootstrap-CI machinery) for
three feature-mask variants of the BioField readout:

  - full      ``[mu, sigma, error]``  (width 3*n) — reproduce the +0.2953
  - no_error  ``[mu, sigma]``         (width 2*n) — KEY ablation: drop error
  - mu_only   ``[mu]``                (width 1*n) — optional

Only the BioField cohort's ``feature_mask`` changes; the VMP belief update,
the input projection, the MLP baseline, the task seeds, the +10 substrate
seed offset and the stats helpers are identical to the pilot. The MLP-vs-MLP
null is mask-independent (no BioField in it) so it is computed once.

Never fabricates — NaN / crash / chance-level results are reported as-is.

Usage (kxkm-ai, n=50)::

    .venv/bin/python scripts/biofield_ablation.py --seeds 50 \
        --out ~/saisail/biofield/biofield_ablation_n50.json
"""
from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import biofield_pilot as bp  # noqa: E402  (scripts/ dir on path via sys.path)
import numpy as np
import torch

from nerve_wml.methodology import (
    accuracy_edge_bootstrap_ci,
    same_substrate_null_result,
)
from track_w.bio_field_readout import BioFieldWMLReadout
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
from track_w.training import train_wml_on_task

# bp mirrors biofield_pilot.py constants exactly.

STEPS_DEFAULT = bp.STEPS_DEFAULT
N_NEURONS = bp.N_NEURONS
D_HIDDEN = bp.D_HIDDEN
DIM = bp.DIM
N_CLASSES = bp.N_CLASSES

VARIANTS: dict[str, tuple[str, ...]] = {
    "full": ("mu", "sigma", "error"),
    "no_error": ("mu", "sigma"),
    "mu_only": ("mu",),
}


def run_w2_hard_biofield_masked(
    seed: int, feature_mask: tuple[str, ...], steps: int = STEPS_DEFAULT
) -> dict:
    """W2-HARD BioField-vs-MLP for a given belief ``feature_mask``.

    Byte-for-byte identical to ``biofield_pilot.run_w2_hard_biofield``
    except the BioField readout is built with ``feature_mask``. Same task
    seeds, same MLP baseline, same +10 substrate-seed offset, same eval.
    """
    import torch.nn.functional as F  # noqa: N812

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    # MLP path — fresh task instance (matched-capacity baseline).
    task_mlp = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    mlp = MlpWML(id=0, d_hidden=D_HIDDEN, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    # BioField path — fresh task instance, independent RNG; +10 seed offset.
    task_bio = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    bio = BioFieldWMLReadout(
        id=0,
        n_neurons=N_NEURONS,
        input_dim=DIM,
        seed=seed + 10,
        feature_mask=feature_mask,
    )
    opt = torch.optim.Adam(bio.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = task_bio.sample(batch=64)
        logits = bio.emit_head_pi(bio.core(x))[:, : task_bio.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    task_eval = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    x, y = task_eval.sample(batch=512)
    with torch.no_grad():
        pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()
        pred_bio = bio.emit_head_pi(bio.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_bio = (pred_bio == y).float().mean().item()

    return {
        "acc_mlp": acc_mlp,
        "acc_biofield": acc_bio,
        "edge": acc_bio - acc_mlp,
        "feature_width": bio.feature_width,
    }


def run_variant(
    seeds: list[int], feature_mask: tuple[str, ...], steps: int, *, verbose: bool
) -> dict:
    real_edges: list[float] = []
    real_mlp: list[float] = []
    real_bio: list[float] = []
    fw = None
    for s in seeds:
        r = run_w2_hard_biofield_masked(s, feature_mask, steps=steps)
        real_mlp.append(r["acc_mlp"])
        real_bio.append(r["acc_biofield"])
        real_edges.append(r["edge"])
        fw = r["feature_width"]
        if verbose and s % 10 == 0:
            print(
                f"    seed {s}: mlp={r['acc_mlp']:.4f} "
                f"bio={r['acc_biofield']:.4f} edge={r['edge']:+.4f}",
                flush=True,
            )
    ci = accuracy_edge_bootstrap_ci(real_edges)
    return {
        "feature_mask": list(feature_mask),
        "feature_width": fw,
        "n": ci.n,
        "mean_edge": ci.mean_edge,
        "median_edge": ci.median_edge,
        "sign_rate_pos": ci.sign_rate_pos,
        "sign_rate_neg": ci.sign_rate_neg,
        "ci95_lo": ci.ci95_low,
        "ci95_hi": ci.ci95_high,
        "ci_excludes_zero": ci.ci_excludes_zero,
        "mean_acc_mlp": float(np.mean(real_mlp)),
        "mean_acc_biofield": float(np.mean(real_bio)),
        "per_seed_edges": real_edges,
        "per_seed_acc_mlp": real_mlp,
        "per_seed_acc_biofield": real_bio,
    }


def run_ablation(
    seeds: list[int],
    variants: list[str],
    steps: int = STEPS_DEFAULT,
    *,
    verbose: bool = True,
) -> dict:
    t0 = time.time()

    # MLP-vs-MLP null — mask-independent (no BioField), computed once,
    # using the pilot's exact null function.
    null_edges = [bp._null_w2_hard_mlp(s, steps=steps)["edge"] for s in seeds]
    null_summ = same_substrate_null_result(null_edges)

    results: dict[str, dict] = {}
    for name in variants:
        if verbose:
            print(f"  --- variant {name} {VARIANTS[name]} ---", flush=True)
        results[name] = run_variant(
            seeds, VARIANTS[name], steps, verbose=verbose
        )

    out = {
        "meta": {
            "task": "HardFlowProxyTask",
            "n_classes": N_CLASSES,
            "n_neurons": N_NEURONS,
            "d_hidden": D_HIDDEN,
            "steps": steps,
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "actual_compute_device": str(torch.empty(1).device),
            "elapsed_sec": time.time() - t0,
            "chance_accuracy": 1.0 / N_CLASSES,
        },
        "mlp_vs_mlp_null": {
            "n": null_summ.n,
            "mean_edge": null_summ.mean_edge,
            "mean_abs_edge": null_summ.mean_abs_edge,
            "abs_p95": null_summ.abs_p95,
            "sign_rate_pos": null_summ.sign_rate_pos,
            "sign_rate_neg": null_summ.sign_rate_neg,
            "ci95_lo": null_summ.ci95_low,
            "ci95_hi": null_summ.ci95_high,
            "ci_excludes_zero": null_summ.ci_excludes_zero,
            "per_seed_edges": null_edges,
        },
        "variants": results,
        "verdict_per_variant": {
            name: {
                "mean_edge": r["mean_edge"],
                "median_edge": r["median_edge"],
                "sign_rate_pos": r["sign_rate_pos"],
                "ci95": [r["ci95_lo"], r["ci95_hi"]],
                "ci_excludes_zero": r["ci_excludes_zero"],
                "edge_exceeds_null_p95": bool(
                    r["mean_edge"] > null_summ.abs_p95
                ),
                "ci_lo_above_null_p95": bool(
                    r["ci95_lo"] > null_summ.abs_p95
                ),
            }
            for name, r in results.items()
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    p.add_argument(
        "--variants",
        type=str,
        default="full,no_error,mu_only",
        help="comma-separated subset of full,no_error,mu_only",
    )
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; choose from {list(VARIANTS)}")

    seeds = list(range(args.seeds))
    print(
        f"=== BioField ablation — n={len(seeds)} seeds, steps={args.steps}, "
        f"variants={variants} ===",
        flush=True,
    )
    out = run_ablation(seeds, variants, steps=args.steps)

    nul = out["mlp_vs_mlp_null"]
    print("\n===== ABLATION VERDICT =====", flush=True)
    print(
        f"null mean|edge| {nul['mean_abs_edge']:.4f}  abs_p95 {nul['abs_p95']:.4f}",
        flush=True,
    )
    for name in variants:
        r = out["variants"][name]
        v = out["verdict_per_variant"][name]
        print(
            f"[{name:8s}] w={r['feature_width']:>2}  "
            f"bio={r['mean_acc_biofield']:.4f} mlp={r['mean_acc_mlp']:.4f}  "
            f"edge={r['mean_edge']:+.4f} (med {r['median_edge']:+.4f})  "
            f"sign+ {r['sign_rate_pos']}/{len(seeds)}  "
            f"CI95 [{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] excl0={r['ci_excludes_zero']}  "
            f">null_p95={v['edge_exceeds_null_p95']} CIlo>p95={v['ci_lo_above_null_p95']}",
            flush=True,
        )
    print(f"elapsed {out['meta']['elapsed_sec']:.1f}s", flush=True)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
