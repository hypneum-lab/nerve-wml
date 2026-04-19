from pathlib import Path

import torch

from interpret.clustering import cluster_codes_by_activation
from interpret.code_semantics import build_semantics_table
from interpret.visualise import render_html_report
from track_w.mlp_wml import MlpWML


def test_render_html_report_creates_file(tmp_path):
    wml = MlpWML(id=0, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)
    centroids = torch.stack([table[c]["activation_centroid"] for c in range(64)])
    clusters = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)

    out = tmp_path / "report.html"
    render_html_report(table, clusters, output_path=str(out), wml_id=0)
    assert out.exists()
    text = out.read_text()

    # All 64 codes present as rows.
    for c in range(64):
        assert f'data-code="{c}"' in text
    # Every cluster id that appears is in [0, 8).
    import re
    cluster_ids = set(int(m) for m in re.findall(r'data-cluster="(\d+)"', text))
    assert all(0 <= cid < 8 for cid in cluster_ids), f"Invalid cluster IDs: {cluster_ids}"


def test_render_html_report_contains_summary_stats():
    """HTML includes a header with WML id, total codes, active codes."""
    import torch
    wml = MlpWML(id=7, d_hidden=16, seed=0)
    inputs = torch.randn(512, 16)
    table = build_semantics_table(wml, inputs, top_k_inputs=3)
    centroids = torch.stack([table[c]["activation_centroid"] for c in range(64)])
    clusters = cluster_codes_by_activation(centroids, n_clusters=8, seed=0)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        render_html_report(table, clusters, output_path=f.name, wml_id=7)
        text = Path(f.name).read_text()

    assert "WML #7" in text or "wml id 7" in text.lower()
    assert "64" in text  # total codes
    assert "<table" in text
    # Plain HTML — no script tags.
    assert "<script" not in text
