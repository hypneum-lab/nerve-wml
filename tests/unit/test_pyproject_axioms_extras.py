"""Pin the v1.8.0 version and the absence of a PyPI-incompatible extras group.

v1.8.0 originally had an ``[axioms]`` extras group referencing
``dreamofkiki`` via a git VCS URL. PyPI's upload server rejects
any package with direct VCS references in its metadata
(``400 Can't have direct dependency``). The extras group was dropped
for the PyPI release; consumers install the dream-of-kiki bridge
with a side-install:

    pip install "dreamofkiki @ git+https://github.com/hypneum-lab/dream-of-kiki@v0.9.1"
    pip install nerve-wml

The test below pins both the version and the absence of the extras
group so a future reintroduction of a PyPI-incompatible dep is
caught at CI time, before the GitHub release workflow fails.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return tomllib.loads(path.read_text())


def test_version_matches_changelog_top_entry():
    """Pyproject version must match the top non-Unreleased CHANGELOG section.

    Tautological version literals (e.g. test_version_is_1_8_0) drift on every
    bump and produce false-red CI signal. Deriving from CHANGELOG anchors
    the assertion to the release process: every bump updates both files.
    """
    declared = _load_pyproject()["project"]["version"]

    changelog_path = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
    changelog = changelog_path.read_text()
    # First "## [X.Y.Z] — DATE" line that is not Unreleased (em-dash U+2014)
    pattern = r"^## \[(\d+\.\d+\.\d+)\] — \d{4}-\d{2}-\d{2}"
    match = re.search(pattern, changelog, re.MULTILINE)
    assert match is not None, "No semver-tagged release header in CHANGELOG.md"
    top_release = match.group(1)
    assert declared == top_release, (
        f"pyproject version {declared} does not match top CHANGELOG release "
        f"{top_release}; one of the two was not updated during the release."
    )


def test_no_vcs_dependency_in_optional_deps():
    """PyPI upload rejects any dep containing 'git+' - pin the invariant."""
    extras = _load_pyproject()["project"].get("optional-dependencies", {})
    for name, deps in extras.items():
        for dep in deps:
            assert "git+" not in dep, (
                f"optional-dependencies.{name} contains a VCS URL: {dep!r}. "
                "PyPI upload will fail. Document the side-install in README."
            )
