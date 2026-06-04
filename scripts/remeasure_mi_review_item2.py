"""Re-measure MI(c_MLP;c_LIF)/H against REVIEW.md item (2).

Runs, on the REAL saved substrate codes (tests/golden/codes_mlp_lif.npz),
the three things the external review demanded before the 0.91-0.96
number can be cited as corroboration:

  (a) plug-in MI/H        -> reproduce the headline number
  (b) bias-corrected MI   -> Panzeri-Treves (analytic) + Miller-Madow,
                             and KSG on the pre-VQ continuous embeddings
                             (already ~debiased)
  (c) permutation null    -> for EACH discrete estimator, shuffle the
                             c_MLP<->c_LIF pairing N_perm times to get
                             the null distribution, its mean (the bias
                             floor) and a one-sided p-value.

Pure numpy/scipy, CPU. No training: consumes pre-computed codes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerve_wml.methodology.mi_estimators import (
    entropy_discrete,
    mi_kraskov_ksg_continuous,
    mi_miller_madow_discrete,
    mi_panzeri_treves_discrete,
    mi_plugin_discrete,
)


def perm_null(estimator, a, b, n_perm, rng):
    obs = estimator(a, b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = estimator(a, rng.permutation(b))
    mean = float(null.mean())
    std = float(null.std())
    p = float(np.sum(null >= obs) + 1) / (n_perm + 1)  # add-one, never 0
    excess = obs - mean
    return {
        "observed": float(obs),
        "null_mean": mean,
        "null_std": std,
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "excess_over_null": float(excess),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codes", type=Path,
                    default=Path("tests/golden/codes_mlp_lif.npz"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-kraskov", type=int, default=2000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("papers/paper1/figures/mi_review_item2.json"))
    args = ap.parse_args()

    d = np.load(args.codes, allow_pickle=True)
    mlp = d["mlp_codes"]
    lif = d["lif_codes"]
    has_emb = "mlp_embeddings" in d and "lif_embeddings" in d
    if has_emb:
        mlp_emb = d["mlp_embeddings"]
        lif_emb = d["lif_embeddings"]

    per_seed = []
    for si, s in enumerate(args.seeds):
        a = mlp[si].astype(np.int64)
        b = lif[si].astype(np.int64)
        rng = np.random.default_rng(s)

        plugin = perm_null(mi_plugin_discrete, a, b, args.n_perm, rng)
        rng = np.random.default_rng(s + 1000)
        pt = perm_null(mi_panzeri_treves_discrete, a, b, args.n_perm, rng)
        rng = np.random.default_rng(s + 2000)
        mm = perm_null(mi_miller_madow_discrete, a, b, args.n_perm, rng)

        ksg = None
        if has_emb:
            xe = mlp_emb[si][: args.n_kraskov]
            ye = lif_emb[si][: args.n_kraskov]
            mi_ksg = mi_kraskov_ksg_continuous(xe, ye, k=args.k)
            h_a = entropy_discrete(a)
            # KSG null: shuffle rows of y embeddings
            rng = np.random.default_rng(s + 3000)
            n_ksg_null = min(args.n_perm, 50)  # KSG is O(N^2), keep small
            null = np.empty(n_ksg_null)
            for i in range(n_ksg_null):
                perm = rng.permutation(ye.shape[0])
                null[i] = mi_kraskov_ksg_continuous(xe, ye[perm], k=args.k)
            ksg = {
                "observed_nats": float(mi_ksg),
                "observed_over_h_a": float(mi_ksg / max(h_a, 1e-9)),
                "null_mean_nats": float(null.mean()),
                "null_p95_nats": float(np.percentile(null, 95)),
                "p_value": float(np.sum(null >= mi_ksg) + 1) / (n_ksg_null + 1),
                "n_null": n_ksg_null,
                "h_a_nats": float(h_a),
            }

        per_seed.append({
            "seed": s,
            "plugin": plugin,
            "panzeri_treves": pt,
            "miller_madow": mm,
            "kraskov_ksg": ksg,
        })
        print(f"seed {s}: "
              f"plugin={plugin['observed']:.3f} (null {plugin['null_mean']:.3f}) | "
              f"PT={pt['observed']:.3f} (null {pt['null_mean']:.3f}) | "
              f"MM={mm['observed']:.3f} | "
              f"KSG/H={ksg['observed_over_h_a']:.3f} (null/H "
              f"{ksg['null_mean_nats']/ksg['h_a_nats']:.3f})"
              if ksg else f"seed {s}: plugin={plugin['observed']:.3f}")

    def mean(key_path):
        vals = []
        for r in per_seed:
            cur = r
            for k in key_path:
                cur = cur[k]
            vals.append(cur)
        return float(np.mean(vals))

    summary = {
        "plugin_mean": mean(["plugin", "observed"]),
        "plugin_null_mean": mean(["plugin", "null_mean"]),
        "plugin_excess_mean": mean(["plugin", "excess_over_null"]),
        "panzeri_treves_mean": mean(["panzeri_treves", "observed"]),
        "panzeri_treves_null_mean": mean(["panzeri_treves", "null_mean"]),
        "miller_madow_mean": mean(["miller_madow", "observed"]),
        "n_seeds": len(args.seeds),
        "n_perm": args.n_perm,
    }
    if has_emb:
        summary["ksg_over_h_a_mean"] = mean(["kraskov_ksg", "observed_over_h_a"])
        ksg_null_over_h = [
            r["kraskov_ksg"]["null_mean_nats"] / r["kraskov_ksg"]["h_a_nats"]
            for r in per_seed
        ]
        summary["ksg_null_over_h_a_mean"] = float(np.mean(ksg_null_over_h))
        summary["vq_amplification_ratio"] = (
            summary["plugin_mean"] / max(summary["ksg_over_h_a_mean"], 1e-9)
        )

    print()
    print("SUMMARY:", json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"per_seed": per_seed, "summary": summary}, indent=2))
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
