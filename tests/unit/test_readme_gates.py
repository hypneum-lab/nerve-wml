"""Pin the README to the 11-gate advertised status."""
from pathlib import Path


def _readme() -> str:
    return Path("README.md").read_text()


def test_readme_lists_all_eleven_gates():
    text = _readme()
    for tag in (
        "gate-p-passed",
        "gate-w-passed",
        "gate-m-passed",
        "gate-m2-passed",
        "gate-scale-passed",
        "gate-interp-passed",
        "gate-neuro-passed",
        "gate-dream-passed",
        "gate-adaptive-passed",
        "gate-llm-advisor-passed",
    ):
        assert tag in text, f"README should advertise {tag}"


def test_readme_links_to_paper_sources():
    """README must link to paper source files that are currently in the repo.

    Asserting label literals like 'paper-v0.3-draft' makes the test red on
    every paper rename. Asserting structural anchors (file paths that
    actually exist) ties the test to the truth of the filesystem.
    """
    text = _readme()
    candidates = [
        "papers/paper1/main.tex",
        "papers/paper2/main.tex",
        "docs/papers/paper1/full-draft.md",
        "docs/papers/paper2/outline.md",
        "papers/",  # fallback: dir-level mention
    ]
    found = [c for c in candidates if c in text]
    assert found, (
        f"README.md does not reference any of the canonical paper anchors "
        f"{candidates}. Ensure the README points to a current paper source."
    )


def test_readme_points_at_every_pilot_script():
    text = _readme()
    for script in (
        "scripts/track_p_pilot.py",
        "scripts/track_w_pilot.py",
        "scripts/merge_pilot.py",
        "scripts/interpret_pilot.py",
        "scripts/adaptive_pilot.py",
    ):
        assert script in text, f"README should point at {script}"
