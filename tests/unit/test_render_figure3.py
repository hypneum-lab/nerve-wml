from pathlib import Path

import pytest


def test_render_p1_dead_curve_creates_pdf(tmp_path):
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_p1_dead_curve

    out = tmp_path / "fig3.pdf"
    render_p1_dead_curve(output_path=str(out), max_steps=2000, checkpoint_every=500)
    assert out.exists()
    assert out.stat().st_size > 1000


@pytest.mark.slow
def test_render_p1_dead_curve_at_paper_location():
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_p1_dead_curve

    out = "papers/paper1/figures/p1_dead_curve.pdf"
    render_p1_dead_curve(output_path=out, max_steps=2000, checkpoint_every=500)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000
