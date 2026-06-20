"""BioFieldWML Phase 2 pilot — belief-state readout vs MLP on HardFlowProxy.

Turns the BioFieldWML SNN-PC substrate from emission-scheduling unit-test
scaffolding into a measured task result (or honest null).

What this tests
---------------
BioFieldWML-readout vs MLP accuracy edge on ``HardFlowProxyTask`` (12-class
XOR-on-noise), the same task the LIF spike-expressivity edge was measured
on (``scripts/lif_edge_null_replication.py``).

  real edge = acc_biofield - acc_mlp   (BioField advantage if > 0)
  null edge = acc_mlp_B   - acc_mlp_A  (two same-arch MLP cohorts, init-
              seed difference only, same pipeline) — the init-variance band.

The MLP is the matched-capacity baseline: the BioField readout's only
trained parameters are a learned input projection (input_dim -> n_neurons)
plus a learned linear π head (3*n_neurons -> n_classes); the belief VMP
update is fixed. The MLP path is the canonical ``run_w2_hard`` MLP cohort.

Methodology mirrors ``run_w2_hard`` exactly:
  - per-cohort fresh task instances with the SAME task seed (cross-cohort
    RNG isolation so one cohort's training RNG draws do not poison the
    other's data stream);
  - global random / numpy / torch RNG locked to ``seed`` at function top;
  - BioField substrate seed uses the legacy ``seed + 10`` offset (matches
    the LIF run's offset for an apples-to-apples comparison).

Stats use the committed ``nerve_wml.methodology`` helpers:
  - ``accuracy_edge_bootstrap_ci``  on the per-seed signed real edges;
  - ``same_substrate_null_result``  on the per-seed MLP-vs-MLP null edges.

Usage
-----
Smoke (macM1)::

    uv run python scripts/biofield_pilot.py --smoke

Full run (kxkm-ai, n=50)::

    uv run python scripts/biofield_pilot.py --seeds 50 \
        --out ~/saisail/biofield/biofield_edge_null_n50.json

Never fabricates — a NaN / crash / chance-level result is reported as-is.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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

# Match the LIF N=2 step budget (scripts/lif_edge_null_replication.py STEPS[2]).
STEPS_DEFAULT = 800
N_NEURONS = 16
D_HIDDEN = 16
DIM = 16
N_CLASSES = 12


def run_w2_hard_biofield(seed: int = 0, steps: int = STEPS_DEFAULT) -> dict:
    """W2-HARD — BioFieldWML belief-state readout vs MLP baseline.

    Returns acc_mlp, acc_biofield, signed edge (biofield - mlp), and the
    relative gap, plus mean_acc_* aliases so a multiseed wrapper can mirror
    the run_w2_hard aggregator shape.
    """
    import random as _random

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

    # BioField path — fresh task instance, independent RNG; +10 seed offset
    # (mirrors the LIF run for cross-substrate comparability).
    task_bio = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    bio = BioFieldWMLReadout(
        id=0, n_neurons=N_NEURONS, input_dim=DIM, seed=seed + 10,
    )
    opt = torch.optim.Adam(bio.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = task_bio.sample(batch=64)
        logits = bio.emit_head_pi(bio.core(x))[:, : task_bio.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Eval — fresh task instance, same seed, same distribution for both.
    task_eval = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    x, y = task_eval.sample(batch=512)
    with torch.no_grad():
        pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()
        pred_bio = bio.emit_head_pi(bio.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_bio = (pred_bio == y).float().mean().item()

    edge = acc_bio - acc_mlp
    gap = abs(acc_mlp - acc_bio) / max(acc_mlp, 1e-6)
    return {
        "acc_mlp": acc_mlp,
        "acc_biofield": acc_bio,
        "edge": edge,
        "gap": gap,
        "mean_acc_mlp": acc_mlp,
        "mean_acc_biofield": acc_bio,
    }


def _null_w2_hard_mlp(seed: int, steps: int = STEPS_DEFAULT) -> dict:
    """MLP-vs-MLP null at N=2 — two MLP cohorts, init-seed difference only.

    Mirrors ``run_w2_hard_biofield`` but replaces the BioField cohort with
    a second MLP seeded at ``seed + 10`` (the same offset the BioField path
    uses), isolating the init-variance band. Edge = acc_B - acc_A.
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_a = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    mlp_a = MlpWML(id=0, d_hidden=D_HIDDEN, seed=seed)
    train_wml_on_task(mlp_a, nerve, task_a, steps=steps, lr=1e-2)

    task_b = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    mlp_b = MlpWML(id=0, d_hidden=D_HIDDEN, seed=seed + 10)
    train_wml_on_task(mlp_b, nerve, task_b, steps=steps, lr=1e-2)

    task_eval = HardFlowProxyTask(dim=DIM, n_classes=N_CLASSES, seed=seed)
    x, y = task_eval.sample(batch=512)
    with torch.no_grad():
        pred_a = mlp_a.emit_head_pi(mlp_a.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_a = (pred_a == y).float().mean().item()
        pred_b = mlp_b.emit_head_pi(mlp_b.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_b = (pred_b == y).float().mean().item()

    return {"acc_a": acc_a, "acc_b": acc_b, "edge": acc_b - acc_a}


def run_biofield_multiseed(
    seeds: list[int], steps: int = STEPS_DEFAULT, *, verbose: bool = True
) -> dict:
    """Multi-seed BioField-vs-MLP edge + MLP-vs-MLP null, with bootstrap CI.

    Returns a JSON-serialisable dict matching the shape of the LIF edge-null
    artifact so the two findings can be compared side by side.
    """
    t0 = time.time()
    real_edges: list[float] = []
    real_mlp: list[float] = []
    real_bio: list[float] = []
    for s in seeds:
        r = run_w2_hard_biofield(seed=s, steps=steps)
        real_mlp.append(r["acc_mlp"])
        real_bio.append(r["acc_biofield"])
        real_edges.append(r["edge"])
        if verbose and s % 10 == 0:
            print(
                f"  [real] seed {s}: mlp={r['acc_mlp']:.4f} "
                f"bio={r['acc_biofield']:.4f} edge={r['edge']:+.4f}",
                flush=True,
            )

    null_edges: list[float] = []
    for s in seeds:
        rn = _null_w2_hard_mlp(seed=s, steps=steps)
        null_edges.append(rn["edge"])
        if verbose and s % 10 == 0:
            print(f"  [null] seed {s}: edge={rn['edge']:+.4f}", flush=True)

    real_ci = accuracy_edge_bootstrap_ci(real_edges)
    null_summ = same_substrate_null_result(null_edges)

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
        },
        "biofield_vs_mlp": {
            "n": real_ci.n,
            "mean_edge": real_ci.mean_edge,
            "median_edge": real_ci.median_edge,
            "sign_rate_pos": real_ci.sign_rate_pos,
            "sign_rate_neg": real_ci.sign_rate_neg,
            "ci95_lo": real_ci.ci95_low,
            "ci95_hi": real_ci.ci95_high,
            "ci_excludes_zero": real_ci.ci_excludes_zero,
            "mean_acc_mlp": float(np.mean(real_mlp)),
            "mean_acc_biofield": float(np.mean(real_bio)),
            "per_seed_edges": real_edges,
            "per_seed_acc_mlp": real_mlp,
            "per_seed_acc_biofield": real_bio,
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
        "verdict": {
            "real_mean_edge": real_ci.mean_edge,
            "real_sign_pos": real_ci.sign_rate_pos,
            "real_ci95": [real_ci.ci95_low, real_ci.ci95_high],
            "real_ci_excludes_zero": real_ci.ci_excludes_zero,
            "null_mean_abs_edge": null_summ.mean_abs_edge,
            "null_abs_p95": null_summ.abs_p95,
            "real_edge_exceeds_null_p95": bool(
                real_ci.mean_edge > null_summ.abs_p95
            ),
            "real_ci_lo_above_null_p95": bool(
                real_ci.ci95_low > null_summ.abs_p95
            ),
            "chance_accuracy": 1.0 / N_CLASSES,
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=50, help="number of seeds")
    p.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    p.add_argument(
        "--smoke", action="store_true",
        help="quick 2-seed smoke (seed 0,1), short steps, no file write",
    )
    p.add_argument("--out", type=str, default=None, help="output JSON path")
    args = p.parse_args()

    if args.smoke:
        print("=== BioField pilot SMOKE (seeds 0,1) ===", flush=True)
        for s in (0, 1):
            r = run_w2_hard_biofield(seed=s, steps=min(args.steps, 200))
            print(
                f"seed {s}: acc_mlp={r['acc_mlp']:.4f} "
                f"acc_biofield={r['acc_biofield']:.4f} "
                f"edge={r['edge']:+.4f} gap={r['gap']:.4f}",
                flush=True,
            )
        return

    seeds = list(range(args.seeds))
    print(
        f"=== BioField pilot — n={len(seeds)} seeds, steps={args.steps} ===",
        flush=True,
    )
    out = run_biofield_multiseed(seeds, steps=args.steps)

    v = out["verdict"]
    bvm = out["biofield_vs_mlp"]
    nul = out["mlp_vs_mlp_null"]
    print("\n===== VERDICT =====", flush=True)
    print(
        f"BioField acc {bvm['mean_acc_biofield']:.4f}  "
        f"MLP acc {bvm['mean_acc_mlp']:.4f}  "
        f"chance {v['chance_accuracy']:.4f}",
        flush=True,
    )
    print(
        f"edge (bio-mlp) {bvm['mean_edge']:+.4f}  "
        f"sign+ {bvm['sign_rate_pos']}/{len(seeds)}  "
        f"CI95 [{bvm['ci95_lo']:+.4f}, {bvm['ci95_hi']:+.4f}]  "
        f"excl0={bvm['ci_excludes_zero']}",
        flush=True,
    )
    print(
        f"null mean|edge| {nul['mean_abs_edge']:.4f}  "
        f"abs_p95 {nul['abs_p95']:.4f}  symm={not nul['ci_excludes_zero']}",
        flush=True,
    )
    print(
        f"real edge > null p95? {v['real_edge_exceeds_null_p95']}  "
        f"CI_lo > null p95? {v['real_ci_lo_above_null_p95']}",
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
