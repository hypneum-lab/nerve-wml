"""Renf 4: massive-seed extended evaluation (50 seeds, multiprocessing).

Runs the three Plan A runners (transducer baselines, GTM ablation,
scale-robustness) on 50 seeds via a worker pool, computes bootstrap CIs
and paired Wilcoxon tests at higher statistical power than v1/v2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scripts._eval_lib import (
    aggregate,
    format_metric_table,
    paired_test_vs_reference,
    run_seeds_parallel,
)
from scripts.gtm_ablation_pilot import run_gtm_ablation
from scripts.scale_robustness_pilot import run_scale_robustness
from scripts.transducer_baselines_pilot import run_transducer_benchmark

_SEEDS = tuple(range(50))
_DOC = Path("docs/superpowers/research/2026-05-20-extended-eval-v3.md")
_JSON = Path("docs/superpowers/research/2026-05-20-extended-eval-v3.json")


def run_extended_v3(*, seeds: tuple[int, ...] = _SEEDS) -> dict[str, Any]:
    out: dict[str, Any] = {"seeds": list(seeds)}

    # Transducer
    t0 = time.perf_counter()
    per_seed = run_seeds_parallel(run_transducer_benchmark, seeds, steps=2000)
    agg = aggregate(per_seed)
    paired = paired_test_vs_reference(agg, reference_method="learned")
    out["transducer"] = {
        "aggregated": agg,
        "paired_vs_learned": paired,
        "wall_clock_s": time.perf_counter() - t0,
    }

    # GTM
    t0 = time.perf_counter()
    per_seed = run_seeds_parallel(run_gtm_ablation, seeds, steps=2000)
    agg = aggregate(per_seed)
    paired = paired_test_vs_reference(
        agg, reference_method="gtm", metric="synchrony_index",
    )
    out["gtm_ablation"] = {
        "aggregated": agg,
        "paired_vs_gtm_on_synchrony": paired,
        "wall_clock_s": time.perf_counter() - t0,
    }

    # Scale-robustness (per-N aggregation)
    t0 = time.perf_counter()
    per_seed_raw = run_seeds_parallel(
        run_scale_robustness, seeds, sizes=(64, 128, 256, 512, 1024),
    )
    # Convert {rows, null_rows} into a {method: {metric: float}} per-seed shape
    # so aggregate() can consume it. Use one "row" arm per N plus a "null" arm
    # per N. Method names: f"real_N{n}", f"null_N{n}".
    per_seed_flat: list[dict[str, dict[str, float]]] = []
    for run in per_seed_raw:
        flat: dict[str, dict[str, float]] = {}
        for row in run["rows"]:
            flat[f"real_N{row['n']}"] = {"cknna": row["cknna"], "hsic": row["hsic"]}
        for row in run["null_rows"]:
            flat[f"null_N{row['n']}"] = {"cknna": row["cknna"], "hsic": row["hsic"]}
        per_seed_flat.append(flat)
    agg = aggregate(per_seed_flat)
    out["scale_robustness"] = {
        "aggregated": agg,
        "wall_clock_s": time.perf_counter() - t0,
    }

    return out


def _write_report(out: dict[str, Any]) -> None:
    t = out["transducer"]
    g = out["gtm_ablation"]
    s = out["scale_robustness"]

    md = []
    md.append("# Extended multi-seed evaluation v3 (50 seeds)")
    md.append("")
    md.append(
        "Renf 4: massive-seed multiprocessing run to confirm v1/v2 findings "
        "at higher statistical power. Wilcoxon at n=50 reaches arbitrarily "
        "small p when effect sizes are large."
    )
    md.append("")
    md.append(f"- transducer wall-clock: {t['wall_clock_s']:.1f}s")
    md.append(f"- gtm wall-clock: {g['wall_clock_s']:.1f}s")
    md.append(f"- scale wall-clock: {s['wall_clock_s']:.1f}s")
    md.append("")
    md.append("## Transducer (MI in bits)")
    md.append("")
    md.append(format_metric_table(t["aggregated"], metric="mi_bits"))
    md.append("")
    md.append("### Paired tests vs learned (Wilcoxon)")
    md.append("")
    md.append("| method | p_value | cohens_dz | median_diff |")
    md.append("|---|---|---|---|")
    for method, res in t["paired_vs_learned"].items():
        md.append(
            f"| {method} | {res['p_value']:.4g} | "
            f"{res['cohens_dz']:+.2f} | {res['median_diff']:+.4f} |"
        )
    md.append("")
    md.append("## GTM ablation (synchrony_index)")
    md.append("")
    md.append(format_metric_table(g["aggregated"], metric="synchrony_index"))
    md.append("")
    md.append("### Paired tests vs gtm on synchrony (Wilcoxon)")
    md.append("")
    md.append("| method | p_value | cohens_dz | median_diff |")
    md.append("|---|---|---|---|")
    for method, res in g["paired_vs_gtm_on_synchrony"].items():
        md.append(
            f"| {method} | {res['p_value']:.4g} | "
            f"{res['cohens_dz']:+.2f} | {res['median_diff']:+.4f} |"
        )
    md.append("")
    md.append("## Scale-robustness (CKNNA)")
    md.append("")
    md.append(format_metric_table(s["aggregated"], metric="cknna"))
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append("- 50 seeds (0..49), multiprocessing pool.")
    md.append(
        "- transducer: steps=2000 (default). gtm: steps=2000. "
        "scale: sizes=(64,128,256,512,1024)."
    )
    md.append("- Bootstrap CI: 2000 resamples per metric.")
    md.append("")
    _DOC.parent.mkdir(parents=True, exist_ok=True)
    _DOC.write_text("\n".join(md))
    _JSON.write_text(json.dumps(out, indent=2, default=float))


def main() -> None:
    out = run_extended_v3()
    _write_report(out)
    print(f"wrote {_DOC} and {_JSON}")


if __name__ == "__main__":
    main()
