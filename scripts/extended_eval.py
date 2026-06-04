"""Extended multi-seed evaluation across the three Plan A runners.

Aggregates mean/std/95% bootstrap CI for every method/metric, then runs
paired Wilcoxon tests for the key contrasts. Writes a Markdown report
and a JSON dump under docs/superpowers/research/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nerve_wml.methodology.paired_tests import bootstrap_ci, wilcoxon_paired
from scripts.gtm_ablation_pilot import run_gtm_ablation
from scripts.multi_seed import run_multi_seed
from scripts.scale_robustness_pilot import run_scale_robustness
from scripts.transducer_baselines_pilot import run_transducer_benchmark

REPORT_MD = Path("docs/superpowers/research/2026-05-20-extended-eval.md")
REPORT_JSON = Path("docs/superpowers/research/2026-05-20-extended-eval.json")

SCALE_SIZES: tuple[int, ...] = (64, 128, 256, 512)


def _aggregate_with_ci(
    aggregated: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Augment ``run_multi_seed`` output with bootstrap 95% CI per metric."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for method, metrics in aggregated.items():
        out[method] = {}
        for metric, stats in metrics.items():
            ci = bootstrap_ci(stats["values"], n_resamples=1000, seed=0)
            out[method][metric] = {
                "values":    list(stats["values"]),
                "mean":      stats["mean"],
                "std":       stats["std"],
                "ci95_low":  ci["ci95_low"],
                "ci95_high": ci["ci95_high"],
                "median":    ci["median"],
            }
    return out


def _paired(
    aggregated: dict[str, dict[str, dict[str, Any]]],
    method_a:   str,
    method_b:   str,
    metric:     str,
) -> dict[str, Any]:
    a = aggregated[method_a][metric]["values"]
    b = aggregated[method_b][metric]["values"]
    res = wilcoxon_paired(a, b)
    res["contrast"] = f"{method_a} vs {method_b}"
    res["metric"] = metric
    return res


def _eval_transducer(seeds: tuple[int, ...]) -> dict[str, Any]:
    raw = run_multi_seed(run_transducer_benchmark, seeds=seeds)
    agg = _aggregate_with_ci(raw)
    contrasts = [
        _paired(agg, "learned", b, "mi_bits")
        for b in ("procrustes", "relative_rep", "vec2vec", "null")
    ]
    return {"per_method": agg, "paired_tests": contrasts}


def _eval_gtm(seeds: tuple[int, ...]) -> dict[str, Any]:
    raw = run_multi_seed(run_gtm_ablation, seeds=seeds)
    agg = _aggregate_with_ci(raw)
    contrasts = []
    for metric in ("mi_bits", "synchrony_index", "accuracy"):
        contrasts.append(_paired(agg, "gtm", "simple_gating", metric))
        contrasts.append(_paired(agg, "gtm", "null", metric))
    return {"per_method": agg, "paired_tests": contrasts}


def _eval_scale(seeds: tuple[int, ...]) -> dict[str, Any]:
    per_seed = [
        run_scale_robustness(sizes=SCALE_SIZES, seed=s) for s in seeds
    ]
    per_n: list[dict[str, Any]] = []
    for i, n in enumerate(SCALE_SIZES):
        real_cknna = [float(run["rows"][i]["cknna"]) for run in per_seed]
        null_cknna = [float(run["null_rows"][i]["cknna"]) for run in per_seed]
        real_hsic = [float(run["rows"][i]["hsic"]) for run in per_seed]
        null_hsic = [float(run["null_rows"][i]["hsic"]) for run in per_seed]
        ci_real_cknna = bootstrap_ci(real_cknna, n_resamples=1000, seed=0)
        ci_null_cknna = bootstrap_ci(null_cknna, n_resamples=1000, seed=0)
        ci_real_hsic = bootstrap_ci(real_hsic, n_resamples=1000, seed=0)
        ci_null_hsic = bootstrap_ci(null_hsic, n_resamples=1000, seed=0)
        paired_cknna = wilcoxon_paired(real_cknna, null_cknna)
        paired_hsic = wilcoxon_paired(real_hsic, null_hsic)
        per_n.append({
            "n": int(n),
            "real_cknna": {"values": real_cknna, **ci_real_cknna},
            "null_cknna": {"values": null_cknna, **ci_null_cknna},
            "real_hsic":  {"values": real_hsic,  **ci_real_hsic},
            "null_hsic":  {"values": null_hsic,  **ci_null_hsic},
            "paired_cknna": paired_cknna,
            "paired_hsic":  paired_hsic,
        })
    return {"per_n": per_n}


def run_extended_eval(
    *, seeds: tuple[int, ...] = tuple(range(10)),
) -> dict[str, Any]:
    """Run the three Plan A pilots over `seeds` and aggregate everything."""
    return {
        "config": {
            "seeds":       list(seeds),
            "scale_sizes": list(SCALE_SIZES),
        },
        "transducer":      _eval_transducer(seeds),
        "gtm_ablation":    _eval_gtm(seeds),
        "scale_robustness": _eval_scale(seeds),
    }


# ---------------------------------------------------------------------- #
# Report rendering                                                       #
# ---------------------------------------------------------------------- #


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _per_method_table(
    per_method: dict[str, dict[str, dict[str, Any]]],
) -> str:
    metrics = sorted({m for d in per_method.values() for m in d})
    lines = [
        "| method | metric | mean | std | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for method in per_method:
        for metric in metrics:
            if metric not in per_method[method]:
                continue
            s = per_method[method][metric]
            lines.append(
                f"| {method} | {metric} | {_fmt(s['mean'])} | "
                f"{_fmt(s['std'])} | "
                f"[{_fmt(s['ci95_low'])}, {_fmt(s['ci95_high'])}] |"
            )
    return "\n".join(lines)


def _paired_table(contrasts: list[dict[str, Any]]) -> str:
    lines = [
        "| contrast | metric | n | p-value | Cohen's d_z | median diff |",
        "|---|---|---|---|---|---|",
    ]
    for c in contrasts:
        lines.append(
            f"| {c['contrast']} | {c['metric']} | {c['n']} | "
            f"{c['p_value']:.4g} | {_fmt(c['cohens_dz'])} | "
            f"{_fmt(c['median_diff'])} |"
        )
    return "\n".join(lines)


def _scale_table(per_n: list[dict[str, Any]]) -> str:
    lines = [
        "| N | real CKNNA mean [CI] | null CKNNA mean [CI] | "
        "paired p | d_z | real HSIC mean [CI] | null HSIC mean [CI] | "
        "HSIC paired p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in per_n:
        rc, nc = row["real_cknna"], row["null_cknna"]
        rh, nh = row["real_hsic"], row["null_hsic"]
        pc, ph = row["paired_cknna"], row["paired_hsic"]
        lines.append(
            f"| {row['n']} | "
            f"{_fmt(rc['mean'])} [{_fmt(rc['ci95_low'])}, "
            f"{_fmt(rc['ci95_high'])}] | "
            f"{_fmt(nc['mean'])} [{_fmt(nc['ci95_low'])}, "
            f"{_fmt(nc['ci95_high'])}] | "
            f"{pc['p_value']:.4g} | {_fmt(pc['cohens_dz'])} | "
            f"{_fmt(rh['mean'])} [{_fmt(rh['ci95_low'])}, "
            f"{_fmt(rh['ci95_high'])}] | "
            f"{_fmt(nh['mean'])} [{_fmt(nh['ci95_low'])}, "
            f"{_fmt(nh['ci95_high'])}] | "
            f"{ph['p_value']:.4g} |"
        )
    return "\n".join(lines)


def _headline_bullets(results: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    tr = results["transducer"]
    tr_means = {m: tr["per_method"][m]["mi_bits"]["mean"] for m in tr["per_method"]}
    learned = tr_means["learned"]
    # Best non-learned alt
    alts = {k: v for k, v in tr_means.items() if k not in ("learned", "null")}
    best_alt = max(alts, key=lambda k: alts[k]) if alts else None
    if best_alt is not None:
        delta = learned - tr_means[best_alt]
        test = next(
            c for c in tr["paired_tests"]
            if c["contrast"] == f"learned vs {best_alt}"
        )
        sig = "significant" if test["p_value"] < 0.05 else "non-significant"
        bullets.append(
            f"Transducer: learned MI = {learned:.3f} bits beats the best "
            f"baseline ({best_alt}, {tr_means[best_alt]:.3f}) by "
            f"Δ={delta:+.3f} bits (Wilcoxon p={test['p_value']:.3g}, "
            f"d_z={test['cohens_dz']:+.2f}; {sig})."
        )
    null_test = next(
        c for c in tr["paired_tests"] if c["contrast"] == "learned vs null"
    )
    bullets.append(
        f"Transducer null check: learned vs null MI Δ_median="
        f"{null_test['median_diff']:+.3f} bits, p={null_test['p_value']:.3g}, "
        f"d_z={null_test['cohens_dz']:+.2f} — the learned arm is "
        f"{'clearly above' if null_test['p_value'] < 0.05 else 'not separated from'} chance."
    )
    gtm = results["gtm_ablation"]
    gtm_mi = gtm["per_method"]["gtm"]["mi_bits"]["mean"]
    sg_mi = gtm["per_method"]["simple_gating"]["mi_bits"]["mean"]
    gtm_vs_sg_mi = next(
        c for c in gtm["paired_tests"]
        if c["contrast"] == "gtm vs simple_gating" and c["metric"] == "mi_bits"
    )
    bullets.append(
        f"GTM ablation: gtm MI={gtm_mi:.3f} vs simple_gating MI={sg_mi:.3f} "
        f"(Δ={gtm_mi - sg_mi:+.3f}, p={gtm_vs_sg_mi['p_value']:.3g}, "
        f"d_z={gtm_vs_sg_mi['cohens_dz']:+.2f})."
    )
    gtm_sync = gtm["per_method"]["gtm"]["synchrony_index"]["mean"]
    sg_sync = gtm["per_method"]["simple_gating"]["synchrony_index"]["mean"]
    gtm_vs_sg_sync = next(
        c for c in gtm["paired_tests"]
        if c["contrast"] == "gtm vs simple_gating"
        and c["metric"] == "synchrony_index"
    )
    bullets.append(
        f"GTM synchrony: gtm={gtm_sync:.3f} vs simple_gating={sg_sync:.3f} "
        f"(Δ={gtm_sync - sg_sync:+.3f}, p={gtm_vs_sg_sync['p_value']:.3g}, "
        f"d_z={gtm_vs_sg_sync['cohens_dz']:+.2f})."
    )
    sc = results["scale_robustness"]["per_n"]
    n_sig = sum(1 for r in sc if r["paired_cknna"]["p_value"] < 0.05)
    largest_n_row = sc[-1]
    bullets.append(
        f"Scale-robustness: real vs null CKNNA paired tests significant at "
        f"{n_sig}/{len(sc)} sizes; at N={largest_n_row['n']}, "
        f"real={largest_n_row['real_cknna']['mean']:.3f} vs "
        f"null={largest_n_row['null_cknna']['mean']:.3f} "
        f"(p={largest_n_row['paired_cknna']['p_value']:.3g}, "
        f"d_z={largest_n_row['paired_cknna']['cohens_dz']:+.2f})."
    )
    return bullets


def render_markdown(results: dict[str, Any]) -> str:
    cfg = results["config"]
    bullets = _headline_bullets(results)
    out: list[str] = []
    out.append("# Extended multi-seed evaluation (2026-05-20)")
    out.append("")
    out.append("## Headline findings")
    out.append("")
    for b in bullets:
        out.append(f"- {b}")
    out.append("")
    out.append("## Configuration")
    out.append("")
    out.append(f"- Seeds: `{cfg['seeds']}` (n={len(cfg['seeds'])})")
    out.append(
        "- Runners: `run_transducer_benchmark`, `run_gtm_ablation`, "
        "`run_scale_robustness`"
    )
    out.append(f"- Scale-robustness sizes: `{cfg['scale_sizes']}`")
    out.append(
        "- Bootstrap CI: 1000 resamples, seed=0; Wilcoxon two-sided "
        "(scipy.stats), Cohen's d_z = mean(diff)/std(diff, ddof=1)."
    )
    out.append("")
    out.append("## Transducer baselines")
    out.append("")
    out.append("### Per-method (10 seeds)")
    out.append("")
    out.append(_per_method_table(results["transducer"]["per_method"]))
    out.append("")
    out.append("### Paired contrasts vs `learned`")
    out.append("")
    out.append(_paired_table(results["transducer"]["paired_tests"]))
    out.append("")
    out.append("## GTM ablation")
    out.append("")
    out.append("### Per-arm (10 seeds)")
    out.append("")
    out.append(_per_method_table(results["gtm_ablation"]["per_method"]))
    out.append("")
    out.append("### Paired contrasts")
    out.append("")
    out.append(_paired_table(results["gtm_ablation"]["paired_tests"]))
    out.append("")
    out.append("## Scale-robustness")
    out.append("")
    out.append(_scale_table(results["scale_robustness"]["per_n"]))
    out.append("")
    out.append("## Limitations")
    out.append("")
    out.append(
        "- Seeds=10 is the minimum recommended for paired Wilcoxon; the "
        "smallest two-sided p-value achievable with n=10 is ~0.002, so "
        "marginally non-significant results may simply be power-limited."
    )
    out.append(
        "- The transducer pilot uses synthetic codebook pairs with default "
        "`steps=2000`; absolute MI numbers are not directly comparable "
        "across runners or task configurations."
    )
    out.append(
        "- Bootstrap CIs are non-parametric over the 10 seed values; with "
        "n=10 the CI is wide and edge effects (ties at endpoints) are "
        "expected."
    )
    out.append(
        "- Scale-robustness only reports CKNNA + HSIC; other geometry "
        "indices are not evaluated."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    import time
    t0 = time.time()
    results = run_extended_eval()
    elapsed = time.time() - t0
    results["config"]["wall_clock_s"] = elapsed
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(results, indent=2, default=float)
    )
    REPORT_MD.write_text(render_markdown(results))
    print(f"Wrote {REPORT_MD} and {REPORT_JSON} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
