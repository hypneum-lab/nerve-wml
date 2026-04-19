"""Assertions that the v0.2 paper body has the new sections."""
from pathlib import Path


def _main_tex_text() -> str:
    return Path("papers/paper1/main.tex").read_text()


def test_paper_references_all_four_figures():
    text = _main_tex_text()
    for label in ("fig:cycle-trace", "fig:w4-forgetting", "fig:p1-dead-curve", "fig:w2-histogram"):
        assert f"\\label{{{label}}}" in text, f"Missing \\label{{{label}}} in main.tex"


def test_paper_has_ablation_table():
    text = _main_tex_text()
    assert "Ablation summary" in text or "ablation summary" in text.lower()
    # Structural: \begin{tabular} somewhere after the §5 Experiments title.
    assert "\\begin{tabular}" in text


def test_paper_has_threats_section():
    text = _main_tex_text()
    assert "\\section{Threats to validity}" in text or \
           "\\section{Threats to Validity}" in text


def test_paper_has_reproducibility_section():
    text = _main_tex_text()
    assert "\\section{Reproducibility}" in text
    # Must reference at least one concrete script by name (escaped underscores OK in LaTeX).
    assert "track" in text and "pilot" in text and "py" in text


def test_paper_inserts_fig_2_and_3_and_4_blocks():
    text = _main_tex_text()
    # Each figure must have an includegraphics.
    assert "figures/w4_forgetting.pdf" in text
    assert "figures/p1_dead_curve.pdf" in text
    assert "figures/w2_histogram.pdf" in text
