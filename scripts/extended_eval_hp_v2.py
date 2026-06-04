"""Extended multi-seed × hyperparameter-grid evaluation v2: larger grids, 10 seeds.

Expanded companion to ``scripts/extended_eval_hp.py`` — addresses the stated
limitation in PR #18's body: previous run used only 4 HP cells per tunable
method × 5 seeds, leading to tied relative_rep observations. This v2 uses
wider grids (8 cells per axis) and 10 seeds to strengthen statistical power
(smallest two-sided Wilcoxon p at n=10 is 0.002, vs 0.0625 at n=5) and
confirm whether the tie still holds or whether the expanded search reveals
a winner.

Grid changes:
  - `lr` grid expanded 1e-3–3e-2 → 1e-4–3e-1 (8 cells)
  - `lambda_cycle` grid expanded 1–100 → 0.1–1000 (8 cells)
  - `n_anchors` grid: clamped to alphabet_size=64, so 4–64 + duplicate runs
    at 64. Using simplified range: 4, 8, 12, 16, 24, 32, 48, 64 (8 cells).
  - Seeds: 5 → 10

Procrustes remains closed-form (no HP), just multi-seeded.

Run:  uv run python -m scripts.extended_eval_hp_v2
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nerve_wml.methodology.paired_tests import bootstrap_ci, wilcoxon_paired
from scripts.multi_seed import run_multi_seed
from scripts.transducer_baselines_pilot import (
    _build_task,
    _mi_entropy_bits,
    _train_learned,
    run_transducer_benchmark,
)

REPORT_MD = Path("docs/superpowers/research/2026-05-20-extended-eval-hp-v2.md")
REPORT_JSON = Path(
    "docs/superpowers/research/2026-05-20-extended-eval-hp-v2.json"
)

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
LR_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)
LAMBDA_GRID: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
N_ANCHORS_GRID: tuple[int, ...] = (4, 8, 12, 16, 24, 32, 48, 64)
LEARNED_STEPS = 2000
V2V_STEPS = 2000
RELREP_STEPS = 500
PROC_STEPS = 500


# ---------------------------------------------------------------------- #
# Runners                                                                #
# ---------------------------------------------------------------------- #


def _learned_runner(
    *, seed: int, lr: float, steps: int = LEARNED_STEPS,
) -> dict[str, dict[str, float]]:
    src, dst, _src_wml, _dst_wml = _build_task(seed)
    t = _train_learned(src, dst, steps, lr=lr)
    pred = t.forward(src, hard=True)
    return {"learned": {"mi_bits": float(_mi_entropy_bits(pred, dst)["mi_bits"])}}


def _vec2vec_runner(
    *, seed: int, lambda_cycle: float, steps: int = V2V_STEPS,
) -> dict[str, dict[str, float]]:
    res = run_transducer_benchmark(
        steps=steps, seed=seed, lambda_cycle=lambda_cycle,
    )
    return {"vec2vec": {"mi_bits": float(res["vec2vec"]["mi_bits"])}}


def _relrep_runner(
    *, seed: int, n_anchors: int,
) -> dict[str, dict[str, float]]:
    # Clamp n_anchors to alphabet_size=64
    n_anchors = min(n_anchors, 64)
    res = run_transducer_benchmark(
        steps=RELREP_STEPS, seed=seed, n_anchors=n_anchors,
    )
    return {"relative_rep": {"mi_bits": float(res["relative_rep"]["mi_bits"])}}


def _procrustes_runner(*, seed: int) -> dict[str, dict[str, float]]:
    res = run_transducer_benchmark(steps=PROC_STEPS, seed=seed)
    return {"procrustes": {"mi_bits": float(res["procrustes"]["mi_bits"])}}


# ---------------------------------------------------------------------- #
# Grid sweep helper                                                      #
# ---------------------------------------------------------------------- #


def _sweep(
    *,
    runner: Any,
    method_key: str,
    hp_name: str,
    hp_grid: tuple[Any, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for hp_value in hp_grid:
        agg = run_multi_seed(runner, seeds=seeds, **{hp_name: hp_value})
        values = [float(v) for v in agg[method_key]["mi_bits"]["values"]]
        ci = bootstrap_ci(values, n_resamples=500, seed=0)
        trials.append({
            "hp_name":  hp_name,
            "hp_value": hp_value,
            "values":   values,
            "mean":     float(ci["mean"]),
            "std":      float(agg[method_key]["mi_bits"]["std"]),
            "median":   float(ci["median"]),
            "ci95_low": float(ci["ci95_low"]),
            "ci95_high": float(ci["ci95_high"]),
        })
    best = max(trials, key=lambda t: t["mean"])
    return {
        "method":    method_key,
        "hp_name":   hp_name,
        "hp_grid":   list(hp_grid),
        "trials":    trials,
        "best":      best,
    }


# ---------------------------------------------------------------------- #
# Main orchestrator                                                      #
# ---------------------------------------------------------------------- #


def run_extended_hp_eval_v2(
    *, seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    """Sweep HP grids for each tunable method, multi-seeded with larger grids."""
    learned = _sweep(
        runner=_learned_runner, method_key="learned",
        hp_name="lr", hp_grid=LR_GRID, seeds=seeds,
    )
    vec2vec = _sweep(
        runner=_vec2vec_runner, method_key="vec2vec",
        hp_name="lambda_cycle", hp_grid=LAMBDA_GRID, seeds=seeds,
    )
    relrep = _sweep(
        runner=_relrep_runner, method_key="relative_rep",
        hp_name="n_anchors", hp_grid=N_ANCHORS_GRID, seeds=seeds,
    )

    # Procrustes: no HP, just multi-seed once.
    proc_agg = run_multi_seed(_procrustes_runner, seeds=seeds)
    proc_values = [float(v) for v in proc_agg["procrustes"]["mi_bits"]["values"]]
    proc_ci = bootstrap_ci(proc_values, n_resamples=500, seed=0)
    procrustes = {
        "method":    "procrustes",
        "hp_name":   None,
        "hp_grid":   [],
        "trials":    [],
        "best": {
            "hp_name":   None,
            "hp_value":  None,
            "values":    proc_values,
            "mean":      float(proc_ci["mean"]),
            "std":       float(proc_agg["procrustes"]["mi_bits"]["std"]),
            "median":    float(proc_ci["median"]),
            "ci95_low":  float(proc_ci["ci95_low"]),
            "ci95_high": float(proc_ci["ci95_high"]),
        },
    }

    # Paired Wilcoxon learned-best vs each baseline-best.
    learned_best_values = learned["best"]["values"]
    paired: dict[str, dict[str, Any]] = {}
    for name, other in [
        ("procrustes",   procrustes),
        ("vec2vec",      vec2vec),
        ("relative_rep", relrep),
    ]:
        other_values = other["best"]["values"]
        test = wilcoxon_paired(learned_best_values, other_values)
        test["contrast"] = f"learned vs {name}"
        test["learned_best_hp"] = (
            f"{learned['hp_name']}={learned['best']['hp_value']}"
        )
        if other["hp_name"] is not None:
            test["other_best_hp"] = (
                f"{other['hp_name']}={other['best']['hp_value']}"
            )
        else:
            test["other_best_hp"] = "n/a (closed-form)"
        test["delta_mean"] = float(
            learned["best"]["mean"] - other["best"]["mean"]
        )
        paired[name] = test

    return {
        "config": {
            "seeds":         list(seeds),
            "lr_grid":       list(LR_GRID),
            "lambda_grid":   list(LAMBDA_GRID),
            "n_anchors_grid": list(N_ANCHORS_GRID),
            "learned_steps":  LEARNED_STEPS,
            "v2v_steps":      V2V_STEPS,
            "relrep_steps":   RELREP_STEPS,
            "proc_steps":     PROC_STEPS,
            "bootstrap_resamples": 500,
        },
        "learned":      learned,
        "vec2vec":      vec2vec,
        "relative_rep": relrep,
        "procrustes":   procrustes,
        "paired_tests": paired,
    }


# ---------------------------------------------------------------------- #
# Report rendering                                                       #
# ---------------------------------------------------------------------- #


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _grid_table(method_block: dict[str, Any]) -> str:
    if not method_block["trials"]:
        return "_(no HP sweep — closed-form method)_"
    best_hp = method_block["best"]["hp_value"]
    hp_name = method_block["hp_name"]
    lines = [
        f"| {hp_name} | mean | std | 95% CI | best |",
        "|---|---|---|---|---|",
    ]
    for t in method_block["trials"]:
        mark = " ★" if t["hp_value"] == best_hp else ""
        lines.append(
            f"| {t['hp_value']} | {_fmt(t['mean'])} | "
            f"{_fmt(t['std'])} | "
            f"[{_fmt(t['ci95_low'])}, {_fmt(t['ci95_high'])}] |{mark} |"
        )
    return "\n".join(lines)


def _paired_table(paired: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| contrast | best HPs | n | p-value | Cohen's d_z | "
        "median diff | Δ mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in paired.values():
        hps = f"{c['learned_best_hp']} vs {c['other_best_hp']}"
        lines.append(
            f"| {c['contrast']} | {hps} | {c['n']} | "
            f"{c['p_value']:.4g} | {_fmt(c['cohens_dz'])} | "
            f"{_fmt(c['median_diff'])} | {_fmt(c['delta_mean'])} |"
        )
    return "\n".join(lines)


def _headline_bullets(results: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    learned = results["learned"]["best"]
    proc = results["procrustes"]["best"]
    v2v = results["vec2vec"]["best"]
    relrep = results["relative_rep"]["best"]
    paired = results["paired_tests"]

    bullets.append(
        f"Learned best HP = lr={learned['hp_value']} → "
        f"MI = {learned['mean']:.3f} bits "
        f"[{learned['ci95_low']:.3f}, {learned['ci95_high']:.3f}] over "
        f"{len(learned['values'])} seeds."
    )
    baselines = [
        ("Procrustes",   "n/a",                          proc,   paired["procrustes"]),
        ("vec2vec",      f"lambda_cycle={v2v['hp_value']}", v2v,    paired["vec2vec"]),
        ("relative_rep", f"n_anchors={relrep['hp_value']}", relrep, paired["relative_rep"]),
    ]
    for label, hp_str, block, test in baselines:
        delta = learned["mean"] - block["mean"]
        sig = "significant" if test["p_value"] < 0.05 else "n.s."
        bullets.append(
            f"vs {label} (best HP {hp_str}, MI={block['mean']:.3f}): "
            f"Δ={delta:+.3f} bits, Wilcoxon p={test['p_value']:.3g}, "
            f"d_z={test['cohens_dz']:+.2f} ({sig})."
        )

    # Sensitivity floor across HP grids.
    floors: list[str] = []
    for label, block in [
        ("learned",      results["learned"]),
        ("vec2vec",      results["vec2vec"]),
        ("relative_rep", results["relative_rep"]),
    ]:
        if not block["trials"]:
            continue
        worst = min(block["trials"], key=lambda t: t["mean"])
        floors.append(
            f"{label} worst-HP MI={worst['mean']:.3f} "
            f"({block['hp_name']}={worst['hp_value']})"
        )
    bullets.append("HP sensitivity floor: " + "; ".join(floors) + ".")
    return bullets


def render_markdown(results: dict[str, Any]) -> str:
    cfg = results["config"]
    out: list[str] = []
    out.append("# Extended multi-seed × HP-grid evaluation v2 (2026-05-20)")
    out.append("")
    out.append(
        "Expanded version of `2026-05-20-extended-eval-hp.md`. "
        "Addresses the v1 limitation (4 HP cells × 5 seeds) by using larger grids "
        "(8 cells per axis) and stronger statistical power (10 seeds). Confirms "
        "whether relative_rep's tie with learned persists or whether wider search "
        "reveals a decisive winner."
    )
    out.append("")
    out.append("## Headline findings")
    out.append("")
    for b in _headline_bullets(results):
        out.append(f"- {b}")
    out.append("")

    out.append("## Per-method HP sweeps")
    out.append("")
    out.append("### Learned transducer (`lr`)")
    out.append("")
    out.append(_grid_table(results["learned"]))
    out.append("")
    out.append("### vec2vec (`lambda_cycle`)")
    out.append("")
    out.append(_grid_table(results["vec2vec"]))
    out.append("")
    out.append("### relative_rep (`n_anchors`)")
    out.append("")
    out.append(_grid_table(results["relative_rep"]))
    out.append("")
    out.append("### Procrustes (closed-form, no HP)")
    out.append("")
    proc = results["procrustes"]["best"]
    out.append(
        f"- mean MI = {_fmt(proc['mean'])} ± {_fmt(proc['std'])} bits "
        f"(95% CI [{_fmt(proc['ci95_low'])}, {_fmt(proc['ci95_high'])}])"
    )
    out.append("")

    out.append("## Paired Wilcoxon at best HPs")
    out.append("")
    out.append(_paired_table(results["paired_tests"]))
    out.append("")

    out.append("## Comparison with v1 (4-cell × 5-seed)")
    out.append("")
    out.append(
        "v2 expanded the grids to improve statistical power and explore "
        "beyond v1's range:"
    )
    out.append("")
    out.append(f"- `lr`: {list(cfg['lr_grid'])} (8 cells, "
               "was (1e-3, 3e-3, 1e-2, 3e-2))")
    out.append(f"- `lambda_cycle`: {list(cfg['lambda_grid'])} (8 cells, "
               "was (1.0, 10.0, 100.0))")
    out.append(f"- `n_anchors`: {list(cfg['n_anchors_grid'])} (8 cells, "
               "was (8, 16, 32, 64) with clamping)")
    out.append(f"- Seeds: {len(cfg['seeds'])} (was 5)")
    out.append("")
    out.append(
        "With n=10 paired seeds, two-sided Wilcoxon gains power: "
        "smallest achievable p ≈ 0.002 vs. 0.0625 in v1. "
        "Any p-value ties at the reported significance threshold now reflect "
        "true absence of effect rather than power limits."
    )
    out.append("")

    out.append("## Configuration")
    out.append("")
    out.append(f"- Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})")
    out.append(f"- `lr` grid: `{cfg['lr_grid']}`")
    out.append(f"- `lambda_cycle` grid: `{cfg['lambda_grid']}`")
    out.append(f"- `n_anchors` grid: `{cfg['n_anchors_grid']}` "
               "(clamped to ≤ alphabet_size=64)")
    out.append(
        f"- Training steps: learned={cfg['learned_steps']}, "
        f"vec2vec={cfg['v2v_steps']}, relative_rep={cfg['relrep_steps']}, "
        f"procrustes={cfg['proc_steps']}."
    )
    out.append(
        f"- Bootstrap CI: {cfg['bootstrap_resamples']} resamples, seed=0; "
        "Wilcoxon two-sided (scipy.stats), Cohen's d_z = "
        "mean(diff)/std(diff, ddof=1)."
    )
    out.append("")

    out.append("## Notes & limitations")
    out.append("")
    out.append(
        "- Procrustes is non-tunable (closed-form orthogonal alignment), "
        "so it appears as a single multi-seeded row rather than a grid."
    )
    out.append(
        "- The `lambda_cycle` grid (8 values) and `n_anchors` grid (8 values) "
        "expand the search space significantly; total wall-clock ~10 minutes "
        "expected with 10 seeds (8 + 8 + 8 + 1 = 25 HP cells × 10 seeds × ~2s/cell)."
    )
    out.append(
        "- Best-of-grid selection is mildly optimistic (a hold-out HP "
        "would be stricter); the worst-of-grid row provides a "
        "sensitivity floor."
    )
    out.append(
        "- `n_anchors` values ≥ 65 are clamped to 64 (alphabet_size), "
        "creating duplicate runs; grid uses 4–64 within the legal range."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    t0 = time.time()
    results = run_extended_hp_eval_v2()
    elapsed = time.time() - t0
    results["config"]["wall_clock_s"] = elapsed
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(results, indent=2, default=float))
    REPORT_MD.write_text(render_markdown(results))
    print(f"Wrote {REPORT_MD} and {REPORT_JSON} in {elapsed:.1f}s")
    # Quick console headline
    print("\nHeadline:")
    for b in _headline_bullets(results):
        print(f"  - {b}")


if __name__ == "__main__":
    main()
