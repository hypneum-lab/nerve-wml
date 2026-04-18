"""Regenerate paper figures from frozen golden artefacts.

Figure 1 (cycle_trace.pdf): first 60 cycles of the frozen trace, drawn as
a 2-row heatmap where row 0 = γ deliveries, row 1 = θ deliveries.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_cycle_trace(
    golden_path:   str = "tests/golden/cycle_trace_4wmls_seed0.npz",
    output_path:   str = "papers/paper1/figures/cycle_trace.pdf",
    n_to_show:     int = 60,
) -> None:
    data = np.load(golden_path)["codes"]
    shown = data[:n_to_show]

    # Column 0 = γ delivery; column 1 = θ delivery (per freeze_golden.py structure).
    row_gamma = (shown[:, 0] >= 0).astype(int)
    row_theta = (shown[:, 1] >= 0).astype(int) if shown.shape[1] > 1 else np.zeros(n_to_show, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(np.stack([row_gamma, row_theta]), aspect="auto", cmap="Greys")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$\gamma$ (predictions)", r"$\theta$ (errors)"])
    ax.set_xlabel("cycle")
    ax.set_title("γ/θ multiplexing — first 60 cycles (golden trace)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    render_cycle_trace()
    print("paper figures rendered.")


if __name__ == "__main__":
    main()
