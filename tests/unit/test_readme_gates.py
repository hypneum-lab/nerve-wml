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


def test_readme_lists_paper_drafts():
    text = _readme()
    assert "paper-v0.2-draft" in text
    assert "paper-v0.3-draft" in text


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
