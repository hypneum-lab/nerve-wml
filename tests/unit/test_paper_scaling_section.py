"""Assert paper §5.1 Scaling is present with the measured numbers."""
from pathlib import Path


def _main_tex() -> str:
    return Path("papers/paper1/main.tex").read_text()


def test_paper_has_scaling_subsection():
    text = _main_tex()
    assert "Scaling behaviour" in text or "Scaling Behaviour" in text or \
           "Gate-Scale" in text


def test_paper_cites_scaling_numbers():
    text = _main_tex()
    # Must mention N=16 and N=32.
    assert "$N = 16$" in text or "N=16" in text or "N = 16" in text
    assert "$N = 32$" in text or "N=32" in text or "N = 32" in text
    # Must mention the router-sparsity diagnostic.
    assert "strongly connected" in text.lower() or "scc" in text.lower()


def test_paper_scaling_mentions_pilots():
    text = _main_tex()
    # At least one of the scaling pilot names must appear.
    pilots_mentioned = any(
        name in text for name in ["run_w1_n16", "run_w2_n16", "run_w4_n16",
                                   "run_w2_n32", "run\\_w1\\_n16", "run\\_w2\\_n16"]
    )
    assert pilots_mentioned, "§5.1 should cite at least one scaling pilot by name"
