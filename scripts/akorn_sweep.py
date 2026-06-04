"""AKOrN parametrisation sweep.

Plan A.4's AKOrN ablation tested the minimal KuramotoMultiplexer with a
single config (32 oscillators, 8 Euler steps, lr=0.02) and found that
its synchrony index clusters with the non-oscillator simple_gating
control (~0.07), not with GTM (~0.20). PR #18 calls this out as a
limitation: the result may be a sub-parametrisation artifact.

This sweep grids n_oscillators x n_steps x lr (3x3x3 = 27 cells), runs
5 seeds per cell, and asks: does any larger config lift AKOrN's
synchrony index toward GTM's ~0.20, or does it stay clustered with
simple_gating across the board?

Run:  uv run python -m scripts.akorn_sweep
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from nerve_wml.methodology.mi_estimators import mi_miller_madow_discrete
from nerve_wml.methodology.paired_tests import bootstrap_ci
from scripts.gtm_ablation_pilot import _synchrony_index
from scripts.multi_seed import run_multi_seed
from track_p.transducer_baselines import KuramotoMultiplexer

_BITS = float(np.log2(np.e))

# Baselines lifted from docs/superpowers/research/2026-05-20-akorn-comparison.md
# (5-seed means at n_osc=32, n_steps=8, lr=0.02, train_steps=2000).
_GTM_SYNC_REF = 0.2004
_SIMPLE_SYNC_REF = 0.0762
_AKORN_SYNC_REF = 0.0727


def _train_akorn_cell(
    codes: torch.Tensor,
    *,
    n_oscillators: int,
    n_steps: int,
    lr: float,
    train_steps: int,
) -> tuple[float, float, float]:
    """Train one parametrised AKOrN cell; return (accuracy, mi_bits, sync)."""
    m = KuramotoMultiplexer(
        alphabet_size=64,
        n_symbols=codes.shape[-1],
        n_oscillators=n_oscillators,
        n_steps=n_steps,
    )
    opt = Adam(m.parameters(), lr=lr)
    for _ in range(train_steps):
        opt.zero_grad()
        carrier = m.forward(codes)
        logits = m.demodulate_logits(carrier)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), codes.reshape(-1)
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        carrier = m.forward(codes)
        pred = m.demodulate(carrier)
        acc = float((pred == codes).float().mean())
        mi = mi_miller_madow_discrete(
            pred.reshape(-1).cpu().numpy().astype(np.int64),
            codes.reshape(-1).cpu().numpy().astype(np.int64),
        ) * _BITS
        sync = _synchrony_index(carrier)
    return acc, mi, sync


def _make_runner(
    *, n_oscillators: int, n_steps: int, lr: float, train_steps: int
):
    """Build a single-cell runner compatible with run_multi_seed."""

    def runner(*, seed: int) -> dict[str, dict[str, float]]:
        torch.manual_seed(seed)
        codes = torch.randint(0, 64, (128, 7))
        acc, mi, sync = _train_akorn_cell(
            codes,
            n_oscillators=n_oscillators,
            n_steps=n_steps,
            lr=lr,
            train_steps=train_steps,
        )
        return {
            "akorn": {
                "accuracy":        acc,
                "mi_bits":         mi,
                "synchrony_index": sync,
            }
        }

    return runner


def run_akorn_sweep(
    *,
    n_oscillators_grid: tuple[int, ...] = (32, 64, 128),
    n_steps_grid:       tuple[int, ...] = (8, 16, 32),
    lr_grid:            tuple[float, ...] = (0.01, 0.02, 0.05),
    seeds:              tuple[int, ...] = (0, 1, 2, 3, 4),
    train_steps:        int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """For each (n_osc, n_steps, lr) cell, train AKOrN over seeds and
    report accuracy / mi_bits / synchrony_index with bootstrap 95% CI.
    """
    cells: list[dict[str, Any]] = []
    for n_osc in n_oscillators_grid:
        for n_st in n_steps_grid:
            for lr in lr_grid:
                runner = _make_runner(
                    n_oscillators=n_osc,
                    n_steps=n_st,
                    lr=lr,
                    train_steps=train_steps,
                )
                agg = run_multi_seed(runner, seeds=seeds)
                metrics = agg["akorn"]
                cell: dict[str, Any] = {
                    "cell_id":       f"n{n_osc}_s{n_st}_lr{lr}",
                    "n_oscillators": int(n_osc),
                    "n_steps":       int(n_st),
                    "lr":            float(lr),
                    "n_seeds":       len(seeds),
                }
                for metric, stats in metrics.items():
                    vals = list(stats["values"])
                    ci = bootstrap_ci(vals, n_resamples=1000, seed=0)
                    cell[metric] = {
                        "values":    vals,
                        "mean":      float(stats["mean"]),
                        "std":       float(stats["std"]),
                        "ci95_low":  ci["ci95_low"],
                        "ci95_high": ci["ci95_high"],
                    }
                cells.append(cell)
    return {"cells": cells}


def _fmt_pm(stats: dict[str, Any]) -> str:
    return f"{stats['mean']:.4f} ± {stats['std']:.4f}"


def _fmt_ci(stats: dict[str, Any]) -> str:
    return f"[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]"


def _render_report(
    result: dict[str, list[dict[str, Any]]],
    *,
    train_steps: int,
    n_seeds: int,
    wall_clock_s: float,
) -> str:
    cells = sorted(
        result["cells"],
        key=lambda c: c["synchrony_index"]["mean"],
        reverse=True,
    )
    top10 = cells[:10]
    bottom5 = cells[-5:]
    top3 = cells[:3]

    max_sync = top10[0]["synchrony_index"]["mean"] if top10 else 0.0
    if max_sync >= 0.15:
        headline = (
            f"**Verdict: WEAKENED** — best cell reaches synchrony "
            f"{max_sync:.4f}, crossing the 0.15 threshold toward "
            f"GTM's ~{_GTM_SYNC_REF:.2f}. AKOrN's earlier "
            f"clustering with simple_gating was a "
            f"sub-parametrisation artifact."
        )
    elif max_sync >= 0.10:
        headline = (
            f"**Verdict: SOFTENED** — best cell reaches synchrony "
            f"{max_sync:.4f}, above simple_gating "
            f"(~{_SIMPLE_SYNC_REF:.2f}) but below GTM "
            f"(~{_GTM_SYNC_REF:.2f}). AKOrN-simple_gating clustering "
            f"is sensitive to sub-parametrisation; the "
            f"distinct-from-GTM claim still holds at this scale."
        )
    else:
        headline = (
            f"**Verdict: ROBUST** — best cell across 27 configs reaches "
            f"only synchrony {max_sync:.4f}, still clustered with "
            f"simple_gating (~{_SIMPLE_SYNC_REF:.2f}) and well below "
            f"GTM (~{_GTM_SYNC_REF:.2f}). The AKOrN-clusters-with-"
            f"simple_gating result from Plan A.4 survives larger "
            f"parametrisation."
        )

    lines: list[str] = []
    lines.append("# AKOrN parametrisation sweep")
    lines.append("")
    lines.append(f"_Generated: 2026-05-20, wall-clock {wall_clock_s:.1f}s._")
    lines.append("")
    lines.append(
        f"Sweep: n_oscillators ∈ {{32, 64, 128}} × n_steps ∈ "
        f"{{8, 16, 32}} × lr ∈ {{0.01, 0.02, 0.05}} = 27 cells × "
        f"{n_seeds} seeds, train_steps={train_steps}."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(headline)
    lines.append("")
    lines.append("## Top 3 cells (synchrony_index, mean ± std)")
    lines.append("")
    lines.append("| cell_id | synchrony | accuracy | mi_bits |")
    lines.append("|---------|-----------|----------|---------|")
    for c in top3:
        lines.append(
            f"| `{c['cell_id']}` | {_fmt_pm(c['synchrony_index'])} | "
            f"{_fmt_pm(c['accuracy'])} | {_fmt_pm(c['mi_bits'])} |"
        )
    lines.append("")
    lines.append(
        "Baselines for reference (Plan A.4, 2000 train_steps, 5 seeds):"
    )
    lines.append("")
    lines.append("| arm | synchrony_index |")
    lines.append("|-----|-----------------|")
    lines.append(f"| GTM | {_GTM_SYNC_REF:.4f} |")
    lines.append(f"| simple_gating | {_SIMPLE_SYNC_REF:.4f} |")
    lines.append(f"| AKOrN (32 osc, 8 steps, lr=0.02) | {_AKORN_SYNC_REF:.4f} |")
    lines.append("")
    lines.append("## Top 10 cells by synchrony_index")
    lines.append("")
    lines.append(
        "| rank | cell_id | n_osc | n_steps | lr | "
        "synchrony (mean ± std) | 95% CI | accuracy | mi_bits |"
    )
    lines.append(
        "|------|---------|-------|---------|----|"
        "------------------------|--------|----------|---------|"
    )
    for i, c in enumerate(top10, start=1):
        lines.append(
            f"| {i} | `{c['cell_id']}` | {c['n_oscillators']} | "
            f"{c['n_steps']} | {c['lr']} | "
            f"{_fmt_pm(c['synchrony_index'])} | "
            f"{_fmt_ci(c['synchrony_index'])} | "
            f"{_fmt_pm(c['accuracy'])} | "
            f"{_fmt_pm(c['mi_bits'])} |"
        )
    lines.append("")
    lines.append("## Bottom 5 cells by synchrony_index")
    lines.append("")
    lines.append(
        "| cell_id | n_osc | n_steps | lr | "
        "synchrony (mean ± std) | 95% CI |"
    )
    lines.append(
        "|---------|-------|---------|----|"
        "------------------------|--------|"
    )
    for c in bottom5:
        lines.append(
            f"| `{c['cell_id']}` | {c['n_oscillators']} | "
            f"{c['n_steps']} | {c['lr']} | "
            f"{_fmt_pm(c['synchrony_index'])} | "
            f"{_fmt_ci(c['synchrony_index'])} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if max_sync >= 0.15:
        lines.append(
            "At least one AKOrN config drives the synchrony index above "
            "0.15, suggesting that the earlier ~0.07 reading reflected "
            "an under-parametrised oscillator head rather than an "
            "intrinsic property of coupled-oscillator multiplexers. The "
            "Plan A.4 robust-distinctness conclusion should be revisited "
            "with the best-cell config or a hyperparameter-fair "
            "comparison protocol."
        )
    elif max_sync >= 0.10:
        lines.append(
            "The best AKOrN config lifts synchrony above simple_gating "
            "but stays below GTM, so the qualitative ordering "
            "(GTM > AKOrN > simple_gating) is still preserved at the "
            "high end — yet the AKOrN-equals-simple_gating clustering "
            "observed in Plan A.4 is sensitive to the chosen "
            "hyperparameters and should be reported with a sensitivity "
            "caveat."
        )
    else:
        lines.append(
            "Across the entire 27-cell grid (4× more oscillators, 4× "
            "more Euler steps, 2.5× learning-rate range), AKOrN never "
            "approaches GTM's synchrony regime and stays statistically "
            "in the simple_gating neighbourhood. The Plan A.4 claim "
            "that GTM's elevated synchrony is genuinely tied to its "
            "band structure rather than a generic oscillator-head "
            "feature is robust to sub-parametrisation."
        )
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append(
        "This sweep uses the minimalist `KuramotoMultiplexer` in "
        "`track_p.transducer_baselines` — learned natural frequencies "
        "and coupling matrix plus a linear readout. It is **not** "
        "Miyato et al.'s full AKOrN with phase-aware readout, "
        "block-structured coupling, or stimulus-driven driving forces. "
        "A full AKOrN port could still behave differently; this result "
        "only rules out a sub-parametrisation explanation for the "
        "minimalist flavour benchmarked in Plan A.4."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    train_steps = 200
    seeds = (0, 1, 2, 3, 4)
    t0 = time.time()
    result = run_akorn_sweep(seeds=seeds, train_steps=train_steps)
    wall_clock_s = time.time() - t0

    out_dir = Path("docs/superpowers/research")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "2026-05-20-akorn-sweep.json"
    md_path = out_dir / "2026-05-20-akorn-sweep.md"

    json_path.write_text(
        json.dumps(
            {
                "train_steps":  train_steps,
                "seeds":        list(seeds),
                "wall_clock_s": wall_clock_s,
                "result":       result,
            },
            indent=2,
        )
    )

    md = _render_report(
        result,
        train_steps=train_steps,
        n_seeds=len(seeds),
        wall_clock_s=wall_clock_s,
    )
    md_path.write_text(md)

    cells = sorted(
        result["cells"],
        key=lambda c: c["synchrony_index"]["mean"],
        reverse=True,
    )
    print(f"Sweep complete in {wall_clock_s:.1f}s.")
    print("Top 3 cells (synchrony_index):")
    for c in cells[:3]:
        print(
            f"  {c['cell_id']:<20}  "
            f"sync={c['synchrony_index']['mean']:.4f} ± "
            f"{c['synchrony_index']['std']:.4f}"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
