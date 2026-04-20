"""Render the paper's §Information Transmission figure (v0.8).

Output: papers/paper1/figures/info_transmission.{pdf,png}

A 3-panel figure — one panel per test:
  (1) MI ratio at N=1 and N=16 pool scale (bar + error bar)
  (2) Round-trip fidelity ratio per seed (bar)
  (3) Cross-substrate merge ratio per seed (bar)

Data recomputed on each invocation (deterministic seeds).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.measure_info_transmission import (  # noqa: E402
    run_test_1_mutual_information,
    run_test_1_pool_scale,
    run_test_2_round_trip_fidelity,
    run_test_3_cross_substrate_merge,
)

OUT_DIR = Path("papers/paper1/figures")


def main() -> None:
    print("Gathering data (may take a few minutes)...")
    r1 = run_test_1_mutual_information(seeds=list(range(5)), steps=400, batch=2048)
    r1_pool = run_test_1_pool_scale(n_wmls=16, seeds=list(range(3)), steps=400, batch=1024)
    r2 = run_test_2_round_trip_fidelity(seeds=list(range(3)), steps=400, transducer_steps=200)
    r3 = run_test_3_cross_substrate_merge(seeds=list(range(3)), steps=400, merge_steps=300)

    mi_n1 = [r["mi_over_h_mlp"] for r in r1]
    mi_n16 = [r["mean_mi_over_h"] for r in r1_pool]
    rt = [r["fidelity_ratio"] for r in r2]
    merge = [r["merge_ratio"] for r in r3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel 1: MI ratio at two scales.
    ax = axes[0]
    means = [np.mean(mi_n1), np.mean(mi_n16)]
    stds  = [np.std(mi_n1),  np.std(mi_n16)]
    ax.bar(["N=1", "N=16 pool"], means, yerr=stds, color=["#1f5fa3", "#3e8ac7"],
           capsize=6, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50 % threshold")
    ax.set_ylabel(r"MI($c_\mathrm{MLP}$, $c_\mathrm{LIF}$) / H($c_\mathrm{MLP}$)")
    ax.set_title("(1) Code-level mutual information")
    ax.set_ylim(0, 1.05)
    for i, (m, s) in enumerate(zip(means, stds, strict=False)):
        ax.annotate(f"{m:.2f}±{s:.2f}", xy=(i, m + 0.04), ha="center", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Round-trip fidelity per seed.
    ax = axes[1]
    ax.bar([f"seed {r['seed']}" for r in r2], rt, color="#5fa341", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="direct baseline")
    ax.axhline(0.85, color="#d96941", linestyle=":", linewidth=1, label="85 % threshold")
    ax.set_ylabel(r"acc$_\mathrm{round\text{-}trip}$ / acc$_\mathrm{direct}$")
    ax.set_title("(2) Round-trip fidelity MLP→LIF→MLP")
    ax.set_ylim(0, 1.25)
    for i, v in enumerate(rt):
        ax.annotate(f"{v:.2f}", xy=(i, v + 0.03), ha="center", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Cross-substrate merge per seed.
    ax = axes[2]
    ax.bar([f"seed {r['seed']}" for r in r3], merge, color="#a3411f", alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="MLP-alone baseline")
    ax.axhline(0.85, color="#d96941", linestyle=":", linewidth=1, label="85 % threshold")
    ax.set_ylabel(r"acc$_\mathrm{cross\text{-}merge}$ / acc$_\mathrm{MLP}$")
    ax.set_title("(3) Cross-substrate merge (LIF fed by MLP only)")
    ax.set_ylim(0, 1.15)
    for i, v in enumerate(merge):
        ax.annotate(f"{v:.2f}", xy=(i, v + 0.02), ha="center", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Inter-substrate information transmission (Claim B empirical)", fontsize=12, y=1.02)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "info_transmission.pdf"
    out_png = OUT_DIR / "info_transmission.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
