"""Generate figures for the CKNNA N-dependence supplementary replication.

Reads the 1750-cell scientific evaluation JSON committed at
docs/superpowers/research/2026-05-20-macm1-scientific-eval.json
and writes two PNG figures into ./figures/.

Run from the repository root:

    uv run python papers/paper1/supplementary/cknna-n-dependence-replication/make_figures.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = REPO / "docs" / "superpowers" / "research" / "2026-05-20-macm1-scientific-eval.json"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def cohens_dz(real: list[float], null: list[float]) -> float:
    diffs = [r - n for r, n in zip(real, null, strict=True)]
    m, s = mean_std(diffs)
    return m / s if s > 0 else float("nan")


def load_cells() -> list[dict]:
    blob = json.loads(DATA.read_text())
    return blob["cells"]


def collect(cells: list[dict], sigma: float) -> dict[int, dict[str, list[float]]]:
    """Return {N: {metric_arm: [values per seed]}} at the given sigma."""
    by_n: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in cells:
        if abs(c["sigma"] - sigma) > 1e-9:
            continue
        n = c["n"]
        for m in ("hsic", "cknna_10", "linear_cka", "mutual_knn_corr_10"):
            v = c["real"].get(m)
            if v is not None:
                by_n[n][f"real.{m}"].append(v)
        for m in ("hsic", "cknna_10"):
            v = c["null"].get(m)
            if v is not None:
                by_n[n][f"null.{m}"].append(v)
    return by_n


def figure1_cknna_decay(by_n: dict[int, dict[str, list[float]]]) -> Path:
    ns = sorted(by_n.keys())
    real_mean, real_std = [], []
    null_mean, null_std = [], []
    for n in ns:
        m, s = mean_std(by_n[n]["real.cknna_10"])
        real_mean.append(m)
        real_std.append(s)
        m, s = mean_std(by_n[n]["null.cknna_10"])
        null_mean.append(m)
        null_std.append(s)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(ns, real_mean, yerr=real_std, marker="o", capsize=3,
                label="aligned (real arm)", color="#1f77b4")
    ax.errorbar(ns, null_mean, yerr=null_std, marker="s", capsize=3,
                label="shuffled (null arm)", color="#d62728")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=0)
    ax.set_xlabel("N (sample size)")
    ax.set_ylabel("CKNNA@k=10")
    ax.set_title(r"CKNNA@10 vs N at $\sigma=0.05$ (50 seeds, error bars = 1 SD)")
    ax.set_ylim(-0.05, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    p = OUT / "fig1_cknna_decay.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def figure2_dz_growth(cells: list[dict], sigma: float = 0.05) -> Path:
    by_n_seed: dict[int, dict[int, dict]] = defaultdict(dict)
    for c in cells:
        if abs(c["sigma"] - sigma) > 1e-9:
            continue
        by_n_seed[c["n"]][c["seed"]] = c

    ns = sorted(by_n_seed.keys())
    metrics = [
        ("cknna_10", "CKNNA@k=10", "#1f77b4", "o"),
        ("hsic", "HSIC", "#2ca02c", "s"),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key, label, color, marker in metrics:
        dz_vals = []
        for n in ns:
            cells_n = by_n_seed[n]
            real = []
            null = []
            for seed in sorted(cells_n.keys()):
                c = cells_n[seed]
                r = c["real"].get(key)
                nu = c["null"].get(key)
                if r is None or nu is None:
                    continue
                real.append(r)
                null.append(nu)
            dz_vals.append(cohens_dz(real, null))
        ax.plot(ns, dz_vals, marker=marker, color=color, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (sample size)")
    ax.set_ylabel(r"Cohen's $d_z$ (real vs null, paired)")
    ax.set_title(r"Signal-vs-null separation grows monotonically with N ($\sigma=0.05$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = OUT / "fig2_dz_growth.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main() -> None:
    cells = load_cells()
    print(f"loaded {len(cells)} cells from {DATA}")
    by_n = collect(cells, sigma=0.05)
    p1 = figure1_cknna_decay(by_n)
    print(f"wrote {p1}")
    p2 = figure2_dz_growth(cells, sigma=0.05)
    print(f"wrote {p2}")

    # Echo headline numbers for the note.
    for n in sorted(by_n.keys()):
        m, s = mean_std(by_n[n]["real.cknna_10"])
        nm, ns_ = mean_std(by_n[n]["null.cknna_10"])
        print(f"  N={n:>5d}  real CKNNA@10 = {m:.4f} ± {s:.4f}   null = {nm:.4f} ± {ns_:.4f}")


if __name__ == "__main__":
    main()
