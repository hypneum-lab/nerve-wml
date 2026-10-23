"""Shared hard guards for the registered N10-C campaign (no mlx import).

Used by ``run_campaign`` and by the CLI entry points of ``harness.train``
(registered 50M budget) and ``harness.evaluate`` (held-out levels >= 1000),
so the registered runs cannot be launched around the campaign runner.
Library functions ``train_run`` / ``evaluate_model`` stay unguarded.

Static binding only (no network): the OSF DOI must look like a real OSF
DOI (placeholders rejected); the manifest must live under
``experiments/procgen_multiagent_gtm_comm/manifest/``, say ``Status:
FILED`` (no ``DRAFT`` anywhere), contain that exact DOI and the expected
commit, be tracked in git, and match ``--manifest-sha256``. The git tree
must be clean and HEAD must be the expected commit, or a descendant whose
every intervening commit only touched ``docs/`` or the manifest folder.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MANIFEST_DIR = Path("experiments/procgen_multiagent_gtm_comm/manifest")
ALLOWED_AFTER_COMMIT = ("docs/", MANIFEST_DIR.as_posix() + "/")
DOI_RE = re.compile(r"^10\.17605/OSF\.IO/[A-Z0-9]{5,}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
STATUS_FILED_RE = re.compile(r"^\W*Status:\s*FILED\b", re.MULTILINE)
REGISTERED_BUDGET = 50_000_000
HELD_OUT_START = 1000


class GuardError(RuntimeError):
    """A hard guard refused the start."""


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def normalize_doi(doi: str | None) -> str | None:
    """Canonical OSF DOI, or None if missing, malformed or a placeholder."""
    if not doi:
        return None
    d = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi.strip())
    if not DOI_RE.match(d):
        return None
    suffix = d.rsplit("/", 1)[1]
    if len(set(suffix)) == 1 or "XXXXX" in suffix:
        return None
    return d


def _head_vs_commit(repo: Path, head: str, expect: str) -> list[str]:
    """HEAD may differ from the recorded code commit only by later commits
    that each touch nothing outside docs/ and the manifest folder."""
    anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                          expect, head], capture_output=True)
    if anc.returncode != 0:
        return [f"HEAD {head} != --expect-commit {expect} (not a descendant)"]
    out = _git(repo, "log", "-m", "--name-only", "--format=", f"{expect}..{head}")
    outside = sorted({p for p in out.splitlines()
                      if p and not p.startswith(ALLOWED_AFTER_COMMIT)})
    if outside:
        return [f"HEAD {head} != --expect-commit {expect} and code changed "
                f"since: {outside[:5]}"]
    return []


SWHID_RE = re.compile(r"^swh:1:(rev|rel|snp|dir|cnt):[0-9a-f]{40}$")
TAG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$")
ORIGIN = "origin"


def freeze_tag_info(repo: Path, tag: str) -> tuple[str | None, list[str]]:
    """Resolve a freeze tag: (peeled commit or None, problems).

    The tag must exist on ``origin`` (git ls-remote), be an *annotated* tag
    (a peeled ``^{}`` entry exists), and exist locally as the same tag
    object (so its message can be read).
    """
    bad: list[str] = []
    if not TAG_NAME_RE.match(tag or ""):
        return None, [f"--freeze-tag {tag!r} is not a valid tag name"]
    r = subprocess.run(
        ["git", "-C", str(repo), "ls-remote", "--tags", ORIGIN,
         f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, [f"git ls-remote {ORIGIN} failed: {r.stderr.strip()}"]
    refs = {ln.split("\t")[1]: ln.split("\t")[0]
            for ln in r.stdout.splitlines() if "\t" in ln}
    obj, peeled = refs.get(f"refs/tags/{tag}"), refs.get(f"refs/tags/{tag}^{{}}")
    if obj is None:
        return None, [f"tag {tag} does not exist on {ORIGIN}"]
    if peeled is None:
        return None, [f"tag {tag} is a lightweight tag: an annotated tag is required"]
    try:
        local = _git(repo, "rev-parse", "--verify", f"refs/tags/{tag}")
    except GuardError:
        return None, [f"tag {tag} not fetched locally (git fetch {ORIGIN} tag {tag})"]
    if local != obj:
        bad.append(f"local tag {tag} differs from {ORIGIN}'s tag object")
    return (peeled if not bad else None), bad


def _freeze_problems(*, repo: Path, manifest: Path | None,
                     manifest_sha256: str | None, manifest_doi: str | None,
                     freeze_tag: str) -> list[str]:
    bad: list[str] = []
    if manifest_doi:
        bad.append("use either --freeze-tag or --manifest-doi, not both")
    commit, probs = freeze_tag_info(repo, freeze_tag)
    bad += probs
    if manifest is None or not Path(manifest).is_file():
        return bad + ["--manifest missing or not a file"]
    mp = Path(manifest).resolve()
    if not mp.is_relative_to((Path(repo) / MANIFEST_DIR).resolve()):
        bad.append(f"manifest must be under {MANIFEST_DIR}/")
    data = mp.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if manifest_sha256 and digest != manifest_sha256.strip().lower():
        bad.append("manifest SHA-256 mismatch (--manifest-sha256)")
    text = data.decode("utf-8", "replace")
    if "DRAFT" in text or "DRAFT" in mp.name:
        bad.append("manifest is a DRAFT")
    frozen = re.compile(r"^\W*Status:\s*FROZEN \(tag " + re.escape(freeze_tag) + r"\)",
                        re.MULTILINE)
    if not (STATUS_FILED_RE.search(text) or frozen.search(text)):
        bad.append(f"manifest has no 'Status: FROZEN (tag {freeze_tag})' "
                   "(or 'Status: FILED') line")
    if commit is not None:
        try:
            msg = _git(repo, "tag", "-l", "--format=%(contents)", freeze_tag)
            m = re.search(r"manifest-sha256:\s*([0-9a-fA-F]{64})", msg)
            p = re.search(r"^manifest:\s*(\S+)", msg, re.MULTILINE)
            rel = mp.relative_to(Path(repo).resolve()).as_posix() \
                if mp.is_relative_to(Path(repo).resolve()) else None
            if not m:
                bad.append("tag message has no 'manifest-sha256: <64 hex>' line")
            elif m.group(1).lower() != digest:
                bad.append("manifest SHA-256 differs from the one in the tag message")
            if not p:
                bad.append("tag message has no 'manifest: <path>' line")
            elif rel and p.group(1) != rel:
                bad.append(f"tag message manifest path {p.group(1)} != {rel}")
        except GuardError as exc:
            bad.append(str(exc))
    try:
        if _git(repo, "status", "--porcelain"):
            bad.append("git tree is dirty (uncommitted or untracked files)")
        head = _git(repo, "rev-parse", "HEAD")
        if commit and head != commit:
            bad.extend(_head_vs_commit(repo, head, commit))
        r = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--error-unmatch", str(mp)],
            capture_output=True, text=True)
        if r.returncode != 0:
            bad.append("manifest file is not tracked in git (must be committed)")
    except GuardError as exc:
        bad.append(str(exc))
    return bad


def guard_problems(*, repo: Path, manifest: Path | None,
                   manifest_sha256: str | None, manifest_doi: str | None,
                   expect_commit: str | None = None,
                   freeze_tag: str | None = None) -> list[str]:
    """All reasons the registered start must be refused (empty list = go)."""
    if freeze_tag:
        return _freeze_problems(
            repo=repo, manifest=manifest, manifest_sha256=manifest_sha256,
            manifest_doi=manifest_doi, freeze_tag=freeze_tag)
    bad: list[str] = []
    doi = normalize_doi(manifest_doi)
    if doi is None:
        bad.append("OSF DOI (--manifest-doi) missing, not 10.17605/OSF.IO/<ID>, "
                   "or a placeholder: the stage-2 manifest must be filed on OSF first")
    if manifest is None or not Path(manifest).is_file():
        bad.append("--manifest missing or not a file")
        manifest = None
    if not manifest_sha256:
        bad.append("--manifest-sha256 missing")
    if expect_commit and HEX40.match(expect_commit.strip().lower()):
        expect_commit = expect_commit.strip().lower()
    else:
        bad.append("--expect-commit missing or not a full 40-hex commit")
        expect_commit = None
    if manifest is not None:
        mp = Path(manifest).resolve()
        if not mp.is_relative_to((Path(repo) / MANIFEST_DIR).resolve()):
            bad.append(f"manifest must be under {MANIFEST_DIR}/")
        data = mp.read_bytes()
        if manifest_sha256 and (
                hashlib.sha256(data).hexdigest() != manifest_sha256.strip().lower()):
            bad.append("manifest SHA-256 mismatch")
        text = data.decode("utf-8", "replace")
        if "DRAFT" in text or "DRAFT" in mp.name:
            bad.append("manifest is a DRAFT")
        if not STATUS_FILED_RE.search(text):
            bad.append("manifest has no 'Status: FILED' line")
        if doi is not None and doi not in text:
            bad.append("manifest does not contain the given DOI")
        if expect_commit and expect_commit not in text.lower():
            bad.append("--expect-commit not recorded in the manifest text")
    try:
        if _git(repo, "status", "--porcelain"):
            bad.append("git tree is dirty (uncommitted or untracked files)")
        head = _git(repo, "rev-parse", "HEAD")
        if expect_commit and head != expect_commit:
            bad.extend(_head_vs_commit(repo, head, expect_commit))
        if manifest is not None:
            r = subprocess.run(
                ["git", "-C", str(repo), "ls-files", "--error-unmatch",
                 str(Path(manifest).resolve())], capture_output=True, text=True)
            if r.returncode != 0:
                bad.append("manifest file is not tracked in git (must be committed)")
    except GuardError as exc:
        bad.append(str(exc))
    return bad


def add_guard_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", type=Path)
    p.add_argument("--manifest-sha256")
    p.add_argument("--manifest-doi")
    p.add_argument("--expect-commit")
    p.add_argument("--freeze-tag", help="annotated freeze tag on origin "
                   "(alternative to --manifest-doi)")


def require_guards(args: argparse.Namespace, repo: Path = REPO) -> None:
    """Raise GuardError unless every guard passes for the parsed CLI args."""
    bad = guard_problems(
        repo=repo, manifest=args.manifest, manifest_sha256=args.manifest_sha256,
        manifest_doi=args.manifest_doi, expect_commit=args.expect_commit,
        freeze_tag=getattr(args, "freeze_tag", None))
    if bad:
        raise GuardError("REFUSED:\n- " + "\n- ".join(bad))


def enforce_cli(args: argparse.Namespace, *, registered: bool, what: str) -> None:
    """Exit(2) unless ``registered`` is False (narrow dev exception: budget
    below the registered one, or training levels only) or every guard passes."""
    import sys

    if not registered:
        return
    try:
        require_guards(args)
    except GuardError as exc:
        sys.exit(f"{what} refused without valid campaign guards: {exc}")
