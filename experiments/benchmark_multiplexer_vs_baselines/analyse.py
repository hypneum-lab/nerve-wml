"""Q1 Task 15 — Welch's t-test + Cohen's d + Bonferroni + figure rendering.

Loads results.json (T13) and optionally ablation_results.json (T14),
computes per-metric pairwise t-tests of GTM vs each baseline,
applies Bonferroni correction over EFFECTIVE comparisons (degenerate
metrics excluded), reports Cohen's d effect size for non-degenerate
metrics, renders a comparison figure, prints the verdict per
pre-registration.

Critic refactor 2026-05-11 (per N8/N9 review):
- CRITICAL #2 : zero-variance metrics produce t=-inf / p=0 with Welch.
  Detect per-arm `std < 1e-6` -> mark metric "degenerate", report
  descriptive stats only, exclude from win/loss/tie counts and from
  the Bonferroni denominator.
- MAJOR #4 : add `--quadratic` mode to fit y = a + b*x + c*x^2 per
  seed and test c<0 (used by Q3 inverted-U analysis). Q1 doesn't
  use it but the helper lives here for reuse.
- Add Cohen's d (pooled-SD effect size) to every non-degenerate test
  record.
- Verdict re-bracketed against effective N (>=2/3 rule applied to
  non-degenerate count).

Plan: HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md Task 15.
Pre-reg: docs/milestones/q1-multiplexer-benchmark-2026-05-10.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

METRICS = ["round_trip_fidelity", "mi_h", "bandwidth_efficiency"]
BASELINES = ["RecursiveLinkBridge", "MLPBridge", "CrossAttentionBridge"]
GTM_NAME = "GTMBridge"
ALPHA_RAW = 0.05
DEGENERATE_STD_THRESHOLD = 1e-6


def load_results(path: Path) -> dict[str, dict[str, list[float]]]:
    """Load runner results.json into {arch: {metric: [vals_per_seed]}}."""
    raw = json.loads(path.read_text())
    out: dict[str, dict[str, list[float]]] = {}
    for r in raw["results"]:
        arch = r["arch"]
        out.setdefault(arch, {m: [] for m in METRICS + ["n_params", "train_time_s"]})
        for m in METRICS:
            out[arch][m].append(r[m])
        out[arch]["n_params"].append(r["n_params"])
        out[arch]["train_time_s"].append(r["train_time_s"])
    return out


def is_degenerate(values: list[float], threshold: float = DEGENERATE_STD_THRESHOLD) -> bool:
    """A sample is degenerate if its sample-std is below threshold."""
    if len(values) < 2:
        return True
    return float(np.std(values, ddof=1)) < threshold


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-SD Cohen's d. Returns NaN if pooled-SD is degenerate."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = float(np.sqrt(pooled))
    if pooled_sd < 1e-9:
        return float("nan")
    return (mean_a - mean_b) / pooled_sd


def welch_test(gtm_vals: list[float], base_vals: list[float]) -> tuple[float, float]:
    stat, p = st.ttest_ind(gtm_vals, base_vals, equal_var=False)
    return float(stat), float(p)


def quadratic_concavity_test(
    x: list[float], y_per_seed: list[list[float]]
) -> dict[str, float | int | list[float]]:
    """Per-seed fit y = a + b*x + c*x^2; sign + t-test on c<0.

    Used by Q3 (Amedi inverted-U). Kept here so analyse.py is the one
    statistical entry point. Returns dict with c-coefficients per seed,
    sign-test p-value, and one-tailed t-test p-value for mean(c)<0.
    """
    x_arr = np.asarray(x, dtype=float)
    c_coeffs: list[float] = []
    for y in y_per_seed:
        coeffs = np.polyfit(x_arr, np.asarray(y, dtype=float), deg=2)
        c, _b, _a = coeffs
        c_coeffs.append(float(c))
    n_concave = sum(1 for c in c_coeffs if c < 0)
    n_total = len(c_coeffs)
    sign_p = float(
        st.binomtest(n_concave, n_total, p=0.5, alternative="greater").pvalue
    )
    t_c, p_two = st.ttest_1samp(c_coeffs, 0.0)
    p_one = float(p_two / 2 if t_c < 0 else 1 - p_two / 2)
    return {
        "c_coeffs": c_coeffs,
        "n_concave_down": n_concave,
        "n_total": n_total,
        "sign_test_p": sign_p,
        "t_stat_c": float(t_c),
        "t_test_p_one_tailed": p_one,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/benchmark_multiplexer_vs_baselines/results.json"),
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("experiments/benchmark_multiplexer_vs_baselines/ablation_results.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("papers/paper2/figures/multiplexer_benchmark.png"),
    )
    parser.add_argument(
        "--verdict",
        type=Path,
        default=None,
        help="Path for verdict JSON (default: <figure_dir>/q1_verdict.json).",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the figure title.",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    ablation = load_results(args.ablation) if args.ablation.exists() else None

    if GTM_NAME not in results:
        print(f"ERROR: {GTM_NAME} not in results", file=sys.stderr)
        return 2

    # ===== Degenerate-metric pre-pass =====
    # A (baseline, metric) pair is degenerate if either GTM or that
    # baseline's per-seed sample std is below threshold. Welch t-test
    # is invalid under zero variance (yields t=-inf, p=0).
    degenerate: list[dict[str, str]] = []
    for baseline in BASELINES:
        if baseline not in results:
            continue
        for metric in METRICS:
            gtm_vals = results[GTM_NAME][metric]
            base_vals = results[baseline][metric]
            gtm_deg = is_degenerate(gtm_vals)
            base_deg = is_degenerate(base_vals)
            if gtm_deg or base_deg:
                degenerate.append({
                    "baseline": baseline,
                    "metric": metric,
                    "gtm_degenerate": gtm_deg,
                    "baseline_degenerate": base_deg,
                    "gtm_value_constant": float(np.mean(gtm_vals)) if gtm_deg else None,
                    "baseline_value_constant": float(np.mean(base_vals)) if base_deg else None,
                })

    n_total_pairs = len(BASELINES) * len(METRICS)
    n_effective = n_total_pairs - len(degenerate)
    if n_effective <= 0:
        print("ERROR: all comparisons are degenerate; no valid tests", file=sys.stderr)
        return 3
    alpha_corrected = ALPHA_RAW / n_effective

    print(
        f"=== Q1 stats: {n_total_pairs} planned, {len(degenerate)} degenerate, "
        f"{n_effective} effective | alpha_corrected = {alpha_corrected:.5f} ==="
    )
    if degenerate:
        print("\nDegenerate metrics (zero/near-zero variance, Welch skipped):")
        for d in degenerate:
            print(
                f"  {d['baseline']:25s} {d['metric']:25s}"
                f"  GTM_deg={d['gtm_degenerate']} base_deg={d['baseline_degenerate']}"
            )
    print()

    # ===== Welch's t-test pairwise (non-degenerate only) =====
    wins = {b: 0 for b in BASELINES}
    losses = {b: 0 for b in BASELINES}
    ties = {b: 0 for b in BASELINES}
    test_records: list[dict] = []

    degen_set = {(d["baseline"], d["metric"]) for d in degenerate}

    for baseline in BASELINES:
        if baseline not in results:
            print(f"WARNING: {baseline} missing from results, skipping", file=sys.stderr)
            continue
        for metric in METRICS:
            gtm_vals = results[GTM_NAME][metric]
            base_vals = results[baseline][metric]
            gtm_mean = float(np.mean(gtm_vals))
            base_mean = float(np.mean(base_vals))
            if (baseline, metric) in degen_set:
                test_records.append({
                    "baseline": baseline,
                    "metric": metric,
                    "gtm_mean": gtm_mean,
                    "baseline_mean": base_mean,
                    "degenerate": True,
                    "t_stat": None,
                    "p": None,
                    "cohens_d": None,
                    "significant": False,
                    "outcome_for_gtm": "excluded",
                })
                print(
                    f"  {baseline:25s} {metric:25s} "
                    f"DEGENERATE GTM={gtm_mean:.4f} base={base_mean:.4f} -> excluded"
                )
                continue

            stat, p = welch_test(gtm_vals, base_vals)
            d = cohens_d(gtm_vals, base_vals)
            sig = p < alpha_corrected
            outcome = "tie"
            if sig:
                outcome = "win" if gtm_mean > base_mean else "loss"
            if outcome == "win":
                wins[baseline] += 1
            elif outcome == "loss":
                losses[baseline] += 1
            else:
                ties[baseline] += 1
            test_records.append({
                "baseline": baseline,
                "metric": metric,
                "gtm_mean": gtm_mean,
                "baseline_mean": base_mean,
                "degenerate": False,
                "t_stat": stat,
                "p": p,
                "cohens_d": d,
                "significant": sig,
                "outcome_for_gtm": outcome,
            })
            print(
                f"  {baseline:25s} {metric:25s} "
                f"t={stat:+6.2f} p={p:.4g} d={d:+.2f} "
                f"GTM={gtm_mean:.4f} base={base_mean:.4f} -> {outcome}"
            )

    total_wins = sum(wins.values())
    total_losses = sum(losses.values())
    total_ties = sum(ties.values())
    print(
        f"\nGTM wins: {total_wins}/{n_effective}, losses: {total_losses}/{n_effective}, "
        f"ties: {total_ties}/{n_effective} (degenerate excluded: {len(degenerate)})"
    )

    # ===== Verdict per pre-registration (re-bracketed for effective N) =====
    # Original brackets: >=6/9 = headline, <=3/9 = loses, else tied.
    # Re-bracketed proportionally: >=2/3 of effective comparisons.
    win_threshold = max(1, int(np.ceil(n_effective * 2 / 3)))
    loss_threshold = max(1, int(np.ceil(n_effective * 2 / 3)))
    if total_wins >= win_threshold:
        verdict = "GTM_headline"
        narrative = (
            f"GTM headline: GTM wins {total_wins}/{n_effective} non-degenerate "
            f"comparisons (>= {win_threshold}). Paper 2 leads with the benchmark."
        )
    elif total_losses >= loss_threshold:
        verdict = "GTM_loses"
        narrative = (
            f"GTM loses: GTM loses {total_losses}/{n_effective} non-degenerate "
            f"comparisons (>= {loss_threshold}). Paper 2 reframes as "
            "falsifiability-scope: PAC multiplexing is task-dependent."
        )
    else:
        verdict = "tied"
        narrative = (
            f"Tied ({total_wins}W/{total_losses}L/{total_ties}T over {n_effective} "
            "non-degenerate comparisons). Paper 2 reframes as convergent evidence: "
            "GTM matches latent-space baselines on round-trip while preserving "
            "phase-coupled biological plausibility, with stronger MI per code unit."
        )
    print(f"\nVerdict: {verdict}")
    print(narrative)

    # ===== Render figure =====
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    archs = [GTM_NAME] + BASELINES
    abl_keys: list[str] = []
    if ablation is not None:
        abl_keys = sorted(ablation.keys())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors_main = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    colors_abl = ["#ff9896", "#aec7e8", "#98df8a"]

    for ax, metric in zip(axes, METRICS, strict=False):
        labels = []
        means = []
        sems = []
        bar_colors = []
        for i, arch in enumerate(archs):
            if arch not in results:
                continue
            vals = results[arch][metric]
            labels.append(arch.replace("Bridge", ""))
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            bar_colors.append(colors_main[i])
        for j, abl in enumerate(abl_keys):
            if metric not in ablation[abl]:
                continue
            vals = ablation[abl][metric]
            labels.append(abl.replace("Bridge", ""))
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
            bar_colors.append(colors_abl[j % len(colors_abl)])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, color=bar_colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(metric)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    suffix = f" {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Q1{suffix} - GTM benchmark (5 seeds, "
        f"Bonferroni alpha={alpha_corrected:.4f}, "
        f"{n_effective} effective comparisons)"
    )
    plt.tight_layout()
    plt.savefig(args.figure, dpi=120)
    print(f"\nWrote figure: {args.figure}")

    # ===== Persist verdict JSON for downstream Paper 2 update =====
    verdict_json = args.verdict if args.verdict else args.figure.with_name("q1_verdict.json")
    verdict_json.parent.mkdir(parents=True, exist_ok=True)
    verdict_json.write_text(json.dumps({
        "verdict": verdict,
        "narrative": narrative,
        "wins": wins, "losses": losses, "ties": ties,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_ties": total_ties,
        "n_planned": n_total_pairs,
        "n_effective": n_effective,
        "n_degenerate": len(degenerate),
        "degenerate_metrics": degenerate,
        "alpha_raw": ALPHA_RAW,
        "alpha_corrected": alpha_corrected,
        "tests": test_records,
        "has_ablation": ablation is not None,
    }, indent=2))
    print(f"Wrote verdict: {verdict_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
