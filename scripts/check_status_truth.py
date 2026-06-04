#!/usr/bin/env python3
"""Validate STATUS / CHANGELOG / CITATION claims against measurable truth.

Run as part of CI to fail fast when documentation drifts from
the underlying code/test state. Each check exits PASS / FAIL
and the script returns non-zero if any FAIL.

N4 Task 5 (2026-05-10) — created in response to N1-N3 audits
that surfaced systematic drift between docs and reality.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []

DRIFT_TOLERANCE = 0.05  # +/-5% on test-count claims


def report(name: str, passed: bool, message: str) -> None:
    results.append((name, passed, message))
    icon = "PASS" if passed else "FAIL"
    print(f"[{icon}] {name}: {message}")


def _load_pyproject_version() -> str | None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    data = tomllib.loads(pyproject_path.read_text())
    return data.get("project", {}).get("version")


def check_test_count() -> None:
    """STATUS.md / README.md test count vs `pytest --collect-only` reality."""
    candidates = [REPO_ROOT / "STATUS.md", REPO_ROOT / "README.md"]
    claim_src = None
    claimed = None
    for p in candidates:
        if not p.exists():
            continue
        text = p.read_text()
        m = re.search(r"(\d{2,5})\s+tests?\s+(?:collected|passing|pass\b)", text)
        if m:
            claim_src = p.name
            claimed = int(m.group(1))
            break
    if claimed is None:
        report("test_count", True, "no test-count claim found in STATUS.md / README.md (skip)")
        return

    out = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    actual_m = re.search(r"(\d+)\s+tests?\s+collected", out.stdout + out.stderr)
    if not actual_m:
        report(
            "test_count",
            False,
            f"could not parse pytest collect output (claimed in {claim_src}: {claimed})",
        )
        return
    actual = int(actual_m.group(1))
    drift = abs(actual - claimed) / max(claimed, 1)
    if drift > DRIFT_TOLERANCE:
        pct = (actual - claimed) / claimed * 100
        report(
            "test_count",
            False,
            f"{claim_src} claims {claimed} tests, actual {actual} ({pct:+.1f}% drift)",
        )
    else:
        report("test_count", True, f"{claim_src}={claimed} ~ pytest={actual}")


def check_pyproject_changelog_version() -> None:
    """pyproject.toml version matches CHANGELOG.md top dated semver entry."""
    declared = _load_pyproject_version()
    if declared is None:
        report("version_consistency", True, "no pyproject.toml version (skip)")
        return

    changelog_path = REPO_ROOT / "CHANGELOG.md"
    if not changelog_path.exists():
        report("version_consistency", True, f"no CHANGELOG.md (only checked pyproject={declared})")
        return

    changelog = changelog_path.read_text()
    pattern = r"^##\s+\[(\d+\.\d+\.\d+)\][\s—–\-]+\d{4}-\d{2}-\d{2}"  # noqa: RUF001
    match = re.search(pattern, changelog, re.MULTILINE)
    if not match:
        report(
            "version_consistency",
            True,
            f"no semver-tagged dated release header in CHANGELOG.md (pyproject={declared})",
        )
        return
    top_release = match.group(1)
    if declared == top_release:
        report("version_consistency", True, f"pyproject={declared} = CHANGELOG top {top_release}")
    else:
        report("version_consistency", False, f"pyproject={declared} != CHANGELOG top {top_release}")


def check_version_py_consistency() -> None:
    """pyproject.toml version matches src/<package>/_version.py.

    N6 Task 2 (2026-05-10): caught by N5 Task 6 — bouba_sens had
    pyproject 0.5.9 but _version.py 0.3.0 (4-minor stale).
    doc-truth now covers this 4th version source.
    """
    declared = _load_pyproject_version()
    if declared is None:
        report("version_py_consistency", True, "no pyproject.toml version (skip)")
        return

    candidates = list(REPO_ROOT.glob("src/**/_version.py")) + list(REPO_ROOT.glob("*/_version.py"))
    candidates = [c for c in candidates if ".venv" not in c.parts and "build" not in c.parts]
    if not candidates:
        report("version_py_consistency", True, "no _version.py (skip)")
        return

    version_file = candidates[0]
    text = version_file.read_text()
    m = re.search(r'__version__\s*=\s*["\']([\d.]+)["\']', text)
    if not m:
        report(
            "version_py_consistency",
            False,
            f"no __version__ literal in {version_file.relative_to(REPO_ROOT)}",
        )
        return
    file_ver = m.group(1)

    if declared == file_ver:
        report(
            "version_py_consistency",
            True,
            f"pyproject={declared} = _version.py {file_ver}",
        )
    else:
        report(
            "version_py_consistency",
            False,
            f"pyproject={declared} != _version.py {file_ver} "
            f"({version_file.relative_to(REPO_ROOT)})",
        )


def check_citation_version() -> None:
    """CITATION.cff top-level `version:` matches pyproject.toml version."""
    declared = _load_pyproject_version()
    cff_path = REPO_ROOT / "CITATION.cff"
    if not cff_path.exists() or declared is None:
        report("citation_version", True, "no CITATION.cff or pyproject version (skip)")
        return
    cff = cff_path.read_text()
    # Match top-level `version: "X.Y.Z"` or `version: X.Y.Z`
    # NOT inside identifiers blocks (those are indented).
    m = re.search(r'^version:\s*["\']?([0-9][^\s"\']+)["\']?\s*$', cff, re.MULTILINE)
    if not m:
        report(
            "citation_version",
            True,
            f"no top-level `version:` in CITATION.cff (skip; pyproject={declared})",
        )
        return
    cff_version = m.group(1)
    if cff_version == declared:
        report(
            "citation_version", True, f"CITATION.cff version={cff_version} = pyproject={declared}"
        )
    else:
        report(
            "citation_version", False, f"CITATION.cff version={cff_version} != pyproject={declared}"
        )


def main() -> int:
    print("=" * 60)
    print(f"check_status_truth.py -- {REPO_ROOT.name}")
    print("=" * 60)
    check_test_count()
    check_pyproject_changelog_version()
    check_version_py_consistency()
    check_citation_version()
    print("=" * 60)
    failed = [n for n, p, _ in results if not p]
    if failed:
        print(f"FAIL: {len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"PASS: {len(results)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
