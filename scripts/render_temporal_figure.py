"""Temporal figure: MI / H at trained vs filler timesteps."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.run_temporal_pilots import run_temporal_mi  # noqa: E402

OUT = Path("papers/paper1/figures")


def main() -> None:
    r = run_temporal_mi(seeds=[0, 1, 2], steps=200)
    fig, ax = plt.subplots(figsize=(7, 3.6))

    seeds = r["seeds"]
    x = np.arange(len(seeds))
    trained = r["trained_step_ratios"]
    filler = r["filler_step_ratios"]

    ax.bar(x - 0.18, trained, 0.36, color="#1f5fa3", label="trained step t_b=13")
    ax.bar(x + 0.18, filler, 0.36, color="gray",    label="filler step t_mid=7")
    ax.axhline(0.30, color="#d96941", linestyle=":", label="30 % threshold")
    ax.axhline(0.50, color="#5fa341", linestyle="--", label="50 % threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel(r"MI / H(MLP) per timestep")
    ax.set_title(
        f"Temporal MI — trained {r['mean_mi_at_trained_step']:.2f} / "
        f"filler {r['mean_mi_at_filler_steps']:.2f}"
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "temporal_info_tx.pdf", bbox_inches="tight")
    fig.savefig(OUT / "temporal_info_tx.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT / 'temporal_info_tx.pdf'}")


if __name__ == "__main__":
    main()
