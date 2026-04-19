"""HTML renderer for the neuroletter semantics table.

Plain <table> + inline CSS. No JavaScript so it renders locally without
a server. Cluster membership is shown as a coloured dot per row.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# 8 cluster colours for up to 8 clusters. For higher n_clusters, they recycle.
_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _format_top_inputs(top: list[dict]) -> str:
    if not top:
        return "<em>(no mapped input)</em>"
    rows = []
    for s in top:
        rows.append(
            f"<span class='summary'>μ={s['mean']:.2f}, "
            f"‖·‖={s['norm']:.2f}, argmax={s['argmax_dim']}</span>"
        )
    return "<br/>".join(rows)


def render_html_report(
    table: dict[int, dict[str, Any]],
    clusters: Tensor,
    *,
    output_path: str,
    wml_id: int,
) -> None:
    """Render a semantics table + cluster labels to a single HTML file."""
    n_codes = len(table)
    active = sum(1 for c in range(n_codes) if table[c]["n_samples_mapped"] > 0)

    # Style + header.
    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        f"<title>WML #{wml_id} — neuroletter semantics</title>",
        "<style>",
        "body { font-family: ui-monospace, monospace; padding: 1em; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ccc; padding: 6px 10px; font-size: 12px; }",
        "th { background: #f0f0f0; }",
        ".dot { display: inline-block; width: 12px; height: 12px; ",
        "       border-radius: 50%; margin-right: 6px; }",
        ".summary { display: inline-block; margin-right: 0.5em; }",
        "tr[data-n='0'] { opacity: 0.45; }",
        "</style></head><body>",
        f"<h2>WML #{wml_id} — neuroletter semantics</h2>",
        f"<p>Total codes: {n_codes}. Active (n_samples_mapped &gt; 0): "
        f"<strong>{active}</strong>.</p>",
        "<table>",
        "<tr><th>code</th><th>cluster</th><th>n mapped</th>"
        "<th>centroid ‖·‖</th><th>top inputs</th><th>next argmax</th></tr>",
    ]

    for c in range(n_codes):
        entry       = table[c]
        cluster_id  = int(clusters[c].item())
        colour      = _PALETTE[cluster_id % len(_PALETTE)]
        centroid_n  = float(torch.as_tensor(entry["activation_centroid"]).norm().item())
        nd          = entry["next_codes_distribution"]
        next_code   = int(torch.as_tensor(nd).argmax().item())
        n_mapped    = int(entry["n_samples_mapped"])

        parts.append(
            f'<tr data-code="{c}" data-cluster="{cluster_id}" data-n="{n_mapped}">'
            f"<td>{c}</td>"
            f"<td><span class='dot' style='background:{colour}'></span>"
            f"cluster {cluster_id}</td>"
            f"<td>{n_mapped}</td>"
            f"<td>{centroid_n:.3f}</td>"
            f"<td>{_format_top_inputs(entry['top_inputs'])}</td>"
            f"<td>→ code {next_code}</td>"
            "</tr>"
        )

    parts.append("</table></body></html>")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(parts))
