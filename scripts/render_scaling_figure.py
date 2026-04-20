"""Render the 4-point W2-hard scaling-law figure for paper §Threats.

Output: papers/paper1/figures/w2_hard_scaling.pdf

Plots median gap vs pool size N, with IQR error bars, on a semi-log x
axis. Annotates each point with the per-seed max (worst case) and
colors the region above the 5 % contract differently from below it.

Data is recomputed on each invocation — deterministic via fixed seeds
0..4 — so the figure tracks the source of truth (scripts/track_w_pilot.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `uv run python scripts/render_scaling_figure.py` without
# the scripts/ parent being on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.track_w_pilot import (  # noqa: E402
    run_w2_hard,
    run_w2_hard_n16_multiseed,
    run_w2_hard_n32_multiseed,
    run_w2_hard_n64_multiseed,
)

OUT_DIR = Path("papers/paper1/figures")


def main() -> None:
    # N=2 point: single-seed (legacy reference).
    r2  = run_w2_hard(steps=800)
    r16 = run_w2_hard_n16_multiseed(seeds=list(range(5)), steps=400)
    r32 = run_w2_hard_n32_multiseed(seeds=list(range(5)), steps=200)
    r64 = run_w2_hard_n64_multiseed(seeds=list(range(5)), steps=150)

    points = [
        ("N=2",  2,  r2["gap"],  r2["gap"],  r2["gap"],  r2["gap"]),
        ("N=16", 16, r16["median_gap"], r16["p25_gap"], r16["p75_gap"], r16["max_gap"]),
        ("N=32", 32, r32["median_gap"], r32["p25_gap"], r32["p75_gap"], r32["max_gap"]),
        ("N=64", 64, r64["median_gap"], r64["p25_gap"], r64["p75_gap"], r64["max_gap"]),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Contract-violation region.
    ax.axhspan(0.05, 0.15, color="#fde4d8", alpha=0.6, label="gap > 5 % (contract violated)")
    ax.axhline(0.05, color="#d96941", linewidth=1.2, linestyle="--", label="5 % contract")

    ns      = np.array([p[1] for p in points])
    medians = np.array([p[2] for p in points])
    p25s    = np.array([p[3] for p in points])
    p75s    = np.array([p[4] for p in points])
    maxes   = np.array([p[5] for p in points])

    # IQR as error bars (asymmetric).
    yerr = np.array([medians - p25s, p75s - medians])
    ax.errorbar(
        ns, medians, yerr=yerr, fmt="o-", color="#1f5fa3", markersize=7,
        capsize=4, linewidth=2, label="median gap ± IQR (5 seeds)",
    )

    # Max overlay.
    ax.scatter(ns, maxes, marker="x", color="#7a2c17", s=50, zorder=3, label="worst seed")

    # Annotate.
    for label, n, med, *_ in points:
        ax.annotate(
            f"{med*100:.1f}%",
            xy=(n, med),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=9,
            color="#1f5fa3",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 16, 32, 64])
    ax.set_xticklabels(["2", "16", "32", "64"])
    ax.set_xlim(1.5, 90)
    ax.set_ylim(0, 0.13)
    ax.set_xlabel("Pool size N (log scale)")
    ax.set_ylabel("Polymorphism gap |acc_MLP − acc_LIF| / max")
    ax.set_title("W2-hard polymorphism gap vs pool size\n"
                 "HardFlowProxyTask, 5 seeds, LIF ≥ MLP in 15/15")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "w2_hard_scaling.pdf"
    out_png = OUT_DIR / "w2_hard_scaling.png"
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")
    print()
    print("Data summary:")
    for label, n, med, p25, p75, mx in points:
        print(f"  {label:5s} (N={n:3d}): median={med*100:5.2f}%  IQR=[{p25*100:5.2f}, {p75*100:5.2f}]  max={mx*100:5.2f}%")


if __name__ == "__main__":
    main()
