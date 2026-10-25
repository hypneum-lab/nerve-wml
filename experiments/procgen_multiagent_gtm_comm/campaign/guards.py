"""Shared hard guards for the registered N10-C campaign (no mlx import).

Commitment rule (amendment 2026-10-23, Filing): the commitment point is a
frozen, annotated git tag ``n10c-freeze-<YYYYMMDD>`` pushed to ``origin``.
The tag is MANDATORY for every registered launch and every held-out
evaluation. An OSF DOI is optional supplementary metadata (OSF is filed
AFTER the tag, so a DOI can never be required before the first run).

``guard_problems`` checks, without network access beyond ``git ls-remote``:

* the tag name, that the tag exists on ``origin`` and is annotated;
* the manifest and the amendment are read from the PEELED TAG COMMIT
  (``git show <tag>^{commit}:<path>``); the tagged manifest bytes must hash
  to the SHA-256 in the tag message, must not say DRAFT and must say
  ``Status: FROZEN (tag <tag>)``; the working-tree copies must be
  byte-identical to the tagged ones;
* a Software Heritage identifier (``--swhid``) is supplied and well formed;
* the git tree is clean and HEAD is the tag commit or a descendant whose
  every intervening commit only touched ``docs/`` or the manifest folder.

On success :func:`require_guards` returns a :class:`RegistrationContext`.
``harness.train.train_run`` and ``harness.evaluate`` refuse registered
budgets (>= 50M pair-steps) and held-out levels (>= 1000) unless they are
handed such a context (the execution boundary); without one they run in an
explicit development mode only (<= 1M pair-steps, training levels 0-15).
The tag's server-recorded GitHub push time is recorded when obtainable;
otherwise the record states it must be taken from GitHub / Software
Heritage (git author and committer dates are not evidence of time).

Registrations (N10-C-bis addendum 2026-10-25): every guard takes a
``registration`` name, ``n10c`` by default (unchanged behaviour: tag
prefix ``n10c-freeze-``, the N10-C amendment, manifest and cells) or
``n10c-bis`` (tag prefix ``n10c-bis-freeze-``, the addendum, the
``stage2-manifest-bis.md`` manifest, the four control arms in both
conditions, every seed on KXKM-AI). The issued ``RegistrationContext``
carries the name, so the execution boundary checks the right cells.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..harness.codes import (
    BIS_ARMS,
    CROSS_PAIRS,
    HOST_SETTINGS,
    PROCGEN_VERSION,
    PROCGEN_X86_WHEEL,
    SEED_HOST,
    SEED_HOST_BIS,
    SEEDS,
)

REPO = Path(__file__).resolve().parents[3]
MANIFEST_DIR = Path("experiments/procgen_multiagent_gtm_comm/manifest")
AMENDMENT_PATH = ("docs/milestones/"
                  "n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md")
ALLOWED_AFTER_COMMIT = ("docs/", MANIFEST_DIR.as_posix() + "/")
DOI_RE = re.compile(r"^10\.17605/OSF\.IO/[A-Z0-9]{5,}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REGISTERED_BUDGET = 50_000_000
HELD_OUT_START = 1000
HELD_OUT_RANGE = range(1000, 1200)
DEV_MAX_PAIR_STEPS = 1_000_000  # development / smoke cap (amendment: <= 1M)
# Learnability-gate dev runs (kind gate_dev) may run longer: the Adam study's
# 10M runs (PI decision 2026-10-25 13:33), matching devledger.GATE_MAX_STEPS.
GATE_DEV_MAX_PAIR_STEPS = 10_000_000


def dev_max_pair_steps(dev_kind: str = "dev_train") -> int:
    """Development cap of a training run of this ledger kind."""
    return GATE_DEV_MAX_PAIR_STEPS if dev_kind == "gate_dev" else DEV_MAX_PAIR_STEPS
DEV_LEVELS = range(0, 16)  # training levels only
REGISTERED_TRAIN_CONFIG = {  # frozen rollout / measurement configuration
    "n_pairs": 64, "horizon": 128, "log_every_pair_steps": 1_000_000,
    "ckpt_every_pair_steps": 1_000_000, "minibatch_pairs": 1024,
}
# Training-options preset of registered runs (harness.train.PRESETS). The
# tag n10c-freeze-20261024 froze the original configuration; the 2026-10-25
# amendment changes this only together with a NEW freeze tag.
REGISTERED_TRAIN_OPTIONS = "frozen-20261024"
# Pre-trained trunk of each registered seed (preset with pretrained_trunk):
# SHA-256 of ``<campaign_dir>/trunks/seed-<seed>/trunk.safetensors``, written
# by the registered pre-training run of that seed on its fixed host and
# pinned by the new freeze's manifest. Empty until then: a registered run of
# a pre-training preset is refused.
REGISTERED_TRUNK_SHA256: dict[int, str] = {}
PRETRAIN_BUDGET = 1_000_000  # single-agent steps per registered pre-training
TRUNK_DIR = "trunks"


def trunk_path(campaign_dir: Path, seed: int) -> Path:
    return Path(campaign_dir) / TRUNK_DIR / f"seed-{seed}" / "trunk.safetensors"
ARMS_BY_CONDITION = {
    "registered": ("gtm", "mlp", "atoc", "dial", "eng", "nocomm"),
    "closed": ("gtm", "dial", "eng", "nocomm"),
}
ORIGIN = "origin"
AMEND_STATUS_LINE_RE = re.compile(r"^\*\*Status:\*\*(.*)$", re.MULTILINE)


def amendment_status_line(tag: str) -> str:
    """The final Status line of the amendment for a freeze tag."""
    return (f"**Status:** FROZEN by tag {tag} (stage 1 + stage 2 manifest); "
            "OSF registration to follow")


def amendment_status_problem(text: str, tag: str) -> str | None:
    """None if the amendment's Status line is the final, tag-naming one."""
    m = AMEND_STATUS_LINE_RE.search(text)
    if not m:
        return "amendment has no '**Status:**' line"
    line = m.group(1)
    if "DRAFT" in line or not re.match(
            r"\s*FROZEN by tag " + re.escape(tag) + r"\b", line):
        return (f"amendment Status is not final: expected "
                f"'**Status:** FROZEN by tag {tag} ...', got {line.strip()[:80]!r}")
    return None
TAG_PREFIX = "n10c-freeze-"


@dataclass(frozen=True)
class Registration:
    """What differs between the N10-C and N10-C-bis registrations."""

    name: str
    tag_prefix: str
    amendment_path: str
    manifest_name: str
    arms_by_condition: dict[str, tuple[str, ...]]
    seed_host: dict[int, str]


ADDENDUM_PATH = "docs/milestones/2026-10-25-n10c-bis-control-arms.md"
ARMS_BY_CONDITION_BIS = {"registered": BIS_ARMS, "closed": BIS_ARMS}
REGISTRATIONS: dict[str, Registration] = {
    "n10c": Registration("n10c", TAG_PREFIX, AMENDMENT_PATH, "stage2-manifest.md",
                         ARMS_BY_CONDITION, SEED_HOST),
    "n10c-bis": Registration("n10c-bis", "n10c-bis-freeze-", ADDENDUM_PATH,
                             "stage2-manifest-bis.md", ARMS_BY_CONDITION_BIS,
                             SEED_HOST_BIS),
}
DEFAULT_REGISTRATION = "n10c"


def registration(name: str | None = None) -> Registration:
    """The registration record for ``name`` (``n10c`` when None)."""
    name = name or DEFAULT_REGISTRATION
    if name not in REGISTRATIONS:
        raise GuardError(f"unknown registration {name!r}; expected one of "
                         f"{sorted(REGISTRATIONS)}")
    return REGISTRATIONS[name]


SWHID_RE = re.compile(
    r"^swh:1:(snp|rev|dir):[0-9a-f]{40}(;[a-z]+=[^;\s]+)*$")
PUSH_TIME_NOTE = ("GitHub server-recorded push time of the tag not obtained "
                  "locally: take it from GitHub (and the archival time from "
                  "Software Heritage); git dates are not evidence of time")


class GuardError(RuntimeError):
    """A hard guard refused the start."""


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       timeout=60)
    return r.stdout if r.returncode == 0 else None


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


def valid_tag_name(tag: str | None,
                   registration_name: str = DEFAULT_REGISTRATION) -> bool:
    """``<prefix>YYYYMMDD`` with a real calendar date (``n10c-freeze-`` for
    N10-C, ``n10c-bis-freeze-`` for N10-C-bis)."""
    prefix = registration(registration_name).tag_prefix
    m = re.fullmatch(re.escape(prefix) + r"(\d{8})", tag or "")
    if not m:
        return False
    try:
        dt.datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return False
    return True


def _head_vs_commit(repo: Path, head: str, expect: str) -> list[str]:
    """HEAD may differ from the recorded code commit only by later commits
    that each touch nothing outside docs/ and the manifest folder."""
    anc = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                          expect, head], capture_output=True)
    if anc.returncode != 0:
        return [f"HEAD {head} != tag commit {expect} (not a descendant)"]
    out = _git(repo, "log", "-m", "--name-only", "--format=", f"{expect}..{head}")
    outside = sorted({p for p in out.splitlines()
                      if p and not p.startswith(ALLOWED_AFTER_COMMIT)})
    if outside:
        return [f"HEAD {head} != tag commit {expect} and code changed "
                f"since: {outside[:5]}"]
    return []


def freeze_tag_info(repo: Path, tag: str,
                    registration_name: str = DEFAULT_REGISTRATION,
                    ) -> tuple[str | None, list[str]]:
    """Resolve a freeze tag: (peeled commit or None, problems).

    The tag must have the registered name, exist on ``origin`` (git
    ls-remote), be an *annotated* tag (a peeled ``^{}`` entry exists), and
    exist locally as the same tag object (so its message can be read).
    """
    bad: list[str] = []
    if not valid_tag_name(tag, registration_name):
        prefix = registration(registration_name).tag_prefix
        return None, [f"--freeze-tag {tag!r} is not a valid freeze tag name "
                      f"({prefix}YYYYMMDD)"]
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
        local_commit = _git(repo, "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
    except GuardError:
        return None, [f"tag {tag} not fetched locally (git fetch {ORIGIN} tag {tag})"]
    if local != obj or local_commit != peeled:
        bad.append(f"local tag {tag} differs from {ORIGIN}'s tag object")
    return (peeled if not bad else None), bad


def _tag_message_fields(repo: Path, tag: str) -> tuple[str | None, str | None]:
    msg = _git(repo, "tag", "-l", "--format=%(contents)", tag)
    m = re.search(r"manifest-sha256:\s*([0-9a-fA-F]{64})", msg)
    p = re.search(r"^manifest:\s*(\S+)", msg, re.MULTILINE)
    return (p.group(1) if p else None), (m.group(1).lower() if m else None)


def _freeze_problems(*, repo: Path, manifest: Path | None,
                     manifest_sha256: str | None, manifest_doi: str | None,
                     expect_commit: str | None, freeze_tag: str | None,
                     swhid: str | None,
                     registration_name: str = DEFAULT_REGISTRATION) -> list[str]:
    repo = Path(repo)
    reg = registration(registration_name)
    amendment_path = reg.amendment_path
    bad: list[str] = []
    if not freeze_tag:
        return ["--freeze-tag missing: the annotated freeze tag is mandatory "
                "for every registered launch and held-out evaluation"]
    commit, probs = freeze_tag_info(repo, freeze_tag, registration_name)
    bad += probs
    if manifest_doi and normalize_doi(manifest_doi) is None:
        bad.append("--manifest-doi (optional supplementary metadata) is not a "
                   "valid OSF DOI 10.17605/OSF.IO/<ID>")
    if not swhid:
        bad.append("--swhid missing: the Software Heritage identifier of the "
                   "tagged state is required")
    elif not SWHID_RE.match(swhid):
        bad.append("--swhid is not a valid swh:1:(snp|rev|dir):<40 hex> identifier")
    if expect_commit:
        if not HEX40.match(expect_commit.strip().lower()):
            bad.append("--expect-commit is not a full 40-hex commit")
        elif commit and expect_commit.strip().lower() != commit:
            bad.append(f"--expect-commit differs from the tag commit {commit}")
    if commit is None:
        return bad
    try:
        rel, msg_sha = _tag_message_fields(repo, freeze_tag)
    except GuardError as exc:
        return bad + [str(exc)]
    if not msg_sha:
        bad.append("tag message has no 'manifest-sha256: <64 hex>' line")
    if not rel:
        bad.append("tag message has no 'manifest: <path>' line")
    if not rel or not msg_sha:
        return bad
    if not (rel.startswith(MANIFEST_DIR.as_posix() + "/") and ".." not in rel.split("/")):
        bad.append(f"manifest must be under {MANIFEST_DIR}/ (tag message: {rel})")
        return bad
    tagged = _git_bytes(repo, "show", f"{commit}:{rel}")
    if tagged is None:
        return bad + [f"manifest {rel} is not in the tagged commit {commit}"]
    digest = hashlib.sha256(tagged).hexdigest()
    if digest != msg_sha:
        bad.append("tagged manifest SHA-256 differs from the one in the tag message")
    if manifest_sha256 and manifest_sha256.strip().lower() != digest:
        bad.append("manifest SHA-256 mismatch (--manifest-sha256 vs tagged manifest)")
    text = tagged.decode("utf-8", "replace")
    if "DRAFT" in text or "DRAFT" in rel:
        bad.append("manifest is a DRAFT")
    frozen = re.compile(r"^\W*Status:\s*FROZEN \(tag " + re.escape(freeze_tag) + r"\)",
                        re.MULTILINE)
    if not frozen.search(text):
        bad.append(f"tagged manifest has no 'Status: FROZEN (tag {freeze_tag})' line")
    amend = _git_bytes(repo, "show", f"{commit}:{amendment_path}")
    if not amend:
        bad.append(f"amendment {amendment_path} is not in the tagged commit")
    else:
        prob = amendment_status_problem(amend.decode("utf-8", "replace"), freeze_tag)
        if prob:
            bad.append(prob + " (Status)")
    # working-tree copies must be the tagged ones (docs/ commits are allowed
    # after the tag, but they must not alter the committed documents)
    wt = repo / rel
    if manifest is not None:
        mp = Path(manifest).resolve()
        if mp != wt.resolve():
            bad.append(f"--manifest {manifest} is not the tagged manifest path {rel}")
    if not wt.is_file() or wt.read_bytes() != tagged:
        bad.append("working-tree manifest differs from the tagged manifest")
    wa = repo / amendment_path
    if amend and (not wa.is_file() or wa.read_bytes() != amend):
        bad.append("working-tree amendment differs from the tagged amendment")
    try:
        if _git(repo, "status", "--porcelain"):
            bad.append("git tree is dirty (uncommitted or untracked files)")
        head = _git(repo, "rev-parse", "HEAD")
        if head != commit:
            bad.extend(_head_vs_commit(repo, head, commit))
        r = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                            str(wt)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append("manifest file is not tracked in git (must be committed)")
    except GuardError as exc:
        bad.append(str(exc))
    return bad


def guard_problems(*, repo: Path, manifest: Path | None = None,
                   manifest_sha256: str | None = None,
                   manifest_doi: str | None = None,
                   expect_commit: str | None = None,
                   freeze_tag: str | None = None,
                   swhid: str | None = None,
                   registration: str = DEFAULT_REGISTRATION) -> list[str]:
    """All reasons the registered start must be refused (empty list = go).

    The freeze tag is mandatory; ``manifest_doi`` is optional metadata.
    """
    return _freeze_problems(
        repo=repo, manifest=manifest, manifest_sha256=manifest_sha256,
        manifest_doi=manifest_doi, expect_commit=expect_commit,
        freeze_tag=freeze_tag, swhid=swhid, registration_name=registration)


# -- registration context ---------------------------------------------------
_TOKEN = object()


@dataclass(frozen=True)
class RegistrationContext:
    """Proof that every registration guard passed (see :func:`require_guards`).

    Only guards issue it; ``train_run`` / ``evaluate_*`` demand one for
    registered budgets and held-out levels.
    """

    tag: str
    commit: str
    swhid: str
    manifest_path: str
    manifest_sha256: str
    amendment_sha256: str
    doi: str | None = None
    tag_push_time: str | None = None
    tag_push_time_note: str | None = None
    registration: str = DEFAULT_REGISTRATION  # "n10c" | "n10c-bis"
    _token: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise GuardError("a RegistrationContext can only be issued by "
                             "campaign.guards.require_guards")
        registration(self.registration)  # known name

    def pins(self) -> dict:
        return {
            "freeze_tag": self.tag, "freeze_commit": self.commit,
            "swhid": self.swhid, "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "amendment_path": registration(self.registration).amendment_path,
            "amendment_sha256": self.amendment_sha256,
            "manifest_doi": self.doi,
            "tag_push_time": self.tag_push_time,
            "tag_push_time_note": self.tag_push_time_note,
            "registration": self.registration,
        }


def _repo_slug(repo: Path) -> str | None:
    try:
        url = _git(repo, "remote", "get-url", ORIGIN)
    except GuardError:
        return None
    m = re.search(r"github\.com[:/]([^/\s]+/[^/\s]+?)(?:\.git)?$", url)
    return m.group(1) if m else None


def github_push_time(repo: Path, tag: str, run=subprocess.run) -> str | None:
    """Server-recorded push time of ``refs/tags/<tag>`` from the GitHub
    events API (``gh api``), or None when not obtainable. Best effort: the
    events feed is short-lived, so None is normal on a late start."""
    slug = _repo_slug(Path(repo))
    if not slug:
        return None
    try:
        r = run(["gh", "api", f"repos/{slug}/events?per_page=100"],
                capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        for ev in json.loads(r.stdout):
            if (ev.get("type") == "PushEvent"
                    and ev.get("payload", {}).get("ref") == f"refs/tags/{tag}"):
                return ev.get("created_at")
    except Exception:  # noqa: BLE001 - best effort, never blocks
        return None
    return None


def issue_context(*, repo: Path, manifest: Path | None = None,
                  manifest_sha256: str | None = None,
                  manifest_doi: str | None = None,
                  expect_commit: str | None = None,
                  freeze_tag: str | None = None, swhid: str | None = None,
                  push_time_fn=github_push_time,
                  registration: str = DEFAULT_REGISTRATION) -> RegistrationContext:
    """Run every guard; return the context or raise GuardError."""
    bad = guard_problems(
        repo=repo, manifest=manifest, manifest_sha256=manifest_sha256,
        manifest_doi=manifest_doi, expect_commit=expect_commit,
        freeze_tag=freeze_tag, swhid=swhid, registration=registration)
    if bad:
        raise GuardError("REFUSED:\n- " + "\n- ".join(bad))
    repo = Path(repo)
    reg = REGISTRATIONS[registration]
    commit, _ = freeze_tag_info(repo, freeze_tag, registration)
    rel, msg_sha = _tag_message_fields(repo, freeze_tag)
    amend = _git_bytes(repo, "show", f"{commit}:{reg.amendment_path}") or b""
    pt = push_time_fn(repo, freeze_tag)
    return RegistrationContext(
        tag=freeze_tag, commit=commit, swhid=swhid, manifest_path=rel,
        manifest_sha256=msg_sha, amendment_sha256=hashlib.sha256(amend).hexdigest(),
        doi=normalize_doi(manifest_doi), tag_push_time=pt,
        tag_push_time_note=None if pt else PUSH_TIME_NOTE,
        registration=registration, _token=_TOKEN)


def make_context_for_tests(**kw) -> RegistrationContext:
    """Forge a context WITHOUT guards. For unit tests only (never imported
    by the harness or the campaign runner)."""
    base = dict(tag="n10c-freeze-20260101", commit="0" * 40,
                swhid="swh:1:rev:" + "0" * 40,
                manifest_path=MANIFEST_DIR.as_posix() + "/m.md",
                manifest_sha256="0" * 64, amendment_sha256="0" * 64)
    return RegistrationContext(**{**base, **kw}, _token=_TOKEN)


def add_guard_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", type=Path,
                   help="optional: must be the tagged manifest path")
    p.add_argument("--manifest-sha256",
                   help="optional: must equal the tagged manifest's SHA-256")
    p.add_argument("--manifest-doi",
                   help="optional supplementary metadata (OSF is filed after "
                        "the tag); never a substitute for --freeze-tag")
    p.add_argument("--expect-commit",
                   help="optional: must equal the freeze tag's commit")
    p.add_argument("--freeze-tag",
                   help="annotated freeze tag n10c-freeze-YYYYMMDD on origin "
                        "(mandatory for registered runs)")
    p.add_argument("--swhid", help="Software Heritage id of the tagged state "
                   "(mandatory for registered runs)")
    p.add_argument("--registration", choices=sorted(REGISTRATIONS),
                   default=DEFAULT_REGISTRATION,
                   help="n10c (default; tag n10c-freeze-YYYYMMDD) or n10c-bis "
                        "(tag n10c-bis-freeze-YYYYMMDD, addendum 2026-10-25)")


def require_guards(args: argparse.Namespace,
                   repo: Path = REPO) -> RegistrationContext:
    """Return the registration context, or raise GuardError."""
    return issue_context(
        repo=repo, manifest=getattr(args, "manifest", None),
        manifest_sha256=getattr(args, "manifest_sha256", None),
        manifest_doi=getattr(args, "manifest_doi", None),
        expect_commit=getattr(args, "expect_commit", None),
        freeze_tag=getattr(args, "freeze_tag", None),
        swhid=getattr(args, "swhid", None),
        registration=getattr(args, "registration", None) or DEFAULT_REGISTRATION)


def enforce_cli(args: argparse.Namespace, *, registered: bool,
                what: str) -> RegistrationContext | None:
    """Exit unless ``registered`` is False (explicit development mode) or
    every guard passes; returns the context for a registered start."""
    import sys

    if not registered:
        return None
    try:
        return require_guards(args)
    except GuardError as exc:
        sys.exit(f"{what} refused without valid campaign guards: {exc}")


# -- runtime identity (actual host / backend / Procgen build) ---------------
def default_runtime_probe() -> dict:
    """Actual runtime: OS, CPU, MLX backend (metal / cuda / cpu)."""
    backend = "cpu"
    try:
        import mlx.core as mx

        if mx.metal.is_available():
            backend = "metal"
        elif getattr(getattr(mx, "cuda", None), "is_available", lambda: False)():
            backend = "cuda"
    except Exception:  # noqa: BLE001
        backend = "unknown"
    return {"system": platform.system(), "machine": platform.machine(),
            "backend": backend}


def default_procgen_probe() -> dict:
    """Installed procgen distribution identity (no procgen import)."""
    try:
        d = md.distribution("procgen")
        wheel = d.read_text("WHEEL") or ""
        tags = re.findall(r"^Tag:\s*(\S+)", wheel, re.MULTILINE)
        return {"version": d.version, "tags": tags}
    except md.PackageNotFoundError:
        return {"version": None, "tags": []}


_runtime_probe = default_runtime_probe
_procgen_probe = default_procgen_probe

HOST_RUNTIME = {  # registered host -> (system, machine, MLX backend)
    "macm1": ("Darwin", "arm64", "metal"),
    "kxkm": ("Linux", "x86_64", "cuda"),
}


def assert_runtime_host(host: str) -> None:
    """The actual runtime must be the registered host's (OS, CPU, backend)."""
    exp = HOST_RUNTIME[host]
    rt = _runtime_probe()
    got = (rt["system"], rt["machine"], rt["backend"])
    if got != exp:
        raise GuardError(
            f"runtime {got} is not host {host!r} {exp} (system, machine, "
            "MLX backend)")


def assert_x86_procgen_build() -> None:
    """Held-out evaluation requires the KXKM x86-64 Procgen wheel."""
    assert_runtime_host("kxkm")
    pg = _procgen_probe()
    parts = PROCGEN_X86_WHEEL.removesuffix(".whl").split("-")
    want_tags = {f"{parts[2]}-{parts[3]}-{plat}" for plat in parts[4].split(".")}
    if pg.get("version") != PROCGEN_VERSION:
        raise GuardError(f"procgen {pg.get('version')} != registered {PROCGEN_VERSION}")
    if set(pg.get("tags") or ()) != want_tags:
        raise GuardError(
            f"procgen wheel tags {pg.get('tags')} do not match the registered "
            f"x86-64 wheel {PROCGEN_X86_WHEEL}")


# -- execution-boundary checks (called by train_run / evaluate_*) -------------
def _need_ctx(ctx: Any, what: str) -> RegistrationContext:
    if not isinstance(ctx, RegistrationContext) or ctx._token is not _TOKEN:
        raise GuardError(
            f"{what} refused: a validated RegistrationContext (freeze-tag "
            "guards) is required")
    return ctx


def check_train_boundary(ctx: Any, *, arm: str, seed: int, condition: str,
                         total_pair_steps: int, settings: dict,
                         config: dict, has_env_factory: bool,
                         options_preset: str = REGISTERED_TRAIN_OPTIONS,
                         init_trunk_sha256: str | None = None,
                         pretrained_trunk: bool = False,
                         dev_kind: str = "dev_train",
                         ) -> str | None:
    """Gate for ``train_run``. Returns the registered host, or None in the
    explicit development mode (budget <= ``dev_max_pair_steps(dev_kind)``:
    1M, or 10M for gate_dev runs; no context).
    A registered run must use the registered training-options preset."""
    if ctx is None:
        cap = dev_max_pair_steps(dev_kind)
        if total_pair_steps > cap:
            raise GuardError(
                f"training with {total_pair_steps:,} pair-steps refused: the "
                f"development mode ({dev_kind}) is limited to {cap:,}; "
                "larger runs need the registered path (freeze-tag context)")
        return None
    _need_ctx(ctx, "registered training")
    reg = registration(ctx.registration)
    errs = registered_run_problems(
        arm=arm, seed=seed, condition=condition, total_pair_steps=total_pair_steps,
        settings=settings, config=config, registration=reg.name)
    if has_env_factory:
        errs.append("a registered run uses the real Procgen (no env_factory)")
    if options_preset != REGISTERED_TRAIN_OPTIONS:
        errs.append(f"training options {options_preset!r} != registered "
                    f"{REGISTERED_TRAIN_OPTIONS!r}")
    if pretrained_trunk:
        want = REGISTERED_TRUNK_SHA256.get(seed)
        if want is None or init_trunk_sha256 != want:
            errs.append(f"pre-trained trunk sha256 {init_trunk_sha256!r} != "
                        f"registered {want!r} for seed {seed}")
    if errs:
        raise GuardError("registered training refused: " + "; ".join(errs))
    host = reg.seed_host[seed]
    assert_runtime_host(host)
    return host


def check_pretrain_boundary(ctx: Any, *, seed: int, total_steps: int,
                            has_env: bool) -> str | None:
    """Gate for ``harness.pretrain``. None = development pre-training
    (ledgered, <= 1M steps); with a context, the registered pre-training
    of a registered seed on its fixed host, with the exact budget."""
    if ctx is None:
        if total_steps > DEV_MAX_PAIR_STEPS:
            raise GuardError(f"development pre-training is limited to "
                             f"{DEV_MAX_PAIR_STEPS:,} steps")
        return None
    _need_ctx(ctx, "registered pre-training")
    errs = []
    if seed not in SEEDS:
        errs.append(f"seed {seed} not in the registered seeds {list(SEEDS)}")
    if total_steps != PRETRAIN_BUDGET:
        errs.append(f"budget {total_steps:,} != registered {PRETRAIN_BUDGET:,}")
    if has_env:
        errs.append("registered pre-training uses the real Procgen")
    if errs:
        raise GuardError("registered pre-training refused: " + "; ".join(errs))
    host = SEED_HOST[seed]
    assert_runtime_host(host)
    return host


def registered_run_problems(*, arm: str, seed: int, condition: str,
                            total_pair_steps: int, settings: dict,
                            config: dict,
                            registration: str = DEFAULT_REGISTRATION) -> list[str]:
    """Registered run tuple, exact budget, frozen configuration and the
    per-host implementation settings (codes.HOST_SETTINGS) of the named
    registration (N10-C cells and seed hosts by default)."""
    reg = REGISTRATIONS[registration] if registration in REGISTRATIONS else None
    if reg is None:
        return [f"unknown registration {registration!r}"]
    arms_by_condition, seed_host = reg.arms_by_condition, reg.seed_host
    errs: list[str] = []
    if seed not in SEEDS:
        errs.append(f"seed {seed} not in the registered seeds {list(SEEDS)}")
    if condition not in arms_by_condition:
        errs.append(f"unknown condition {condition!r}")
    elif arm not in arms_by_condition[condition]:
        errs.append(f"arm {arm!r} is not run in condition {condition!r} "
                    f"(allowed: {arms_by_condition[condition]})")
    if total_pair_steps != REGISTERED_BUDGET:
        errs.append(f"budget {total_pair_steps:,} != registered {REGISTERED_BUDGET:,}")
    for k, v in REGISTERED_TRAIN_CONFIG.items():
        if config.get(k) != v:
            errs.append(f"{k}={config.get(k)!r} != registered {v!r}")
    if seed in seed_host:
        want = HOST_SETTINGS[seed_host[seed]]
        got = {k: settings.get(k) for k in want}
        if got != want:
            errs.append(f"per-host settings {got} != registered "
                        f"{want} for host {seed_host[seed]!r}")
    return errs


def check_eval_boundary(ctx: Any, levels: Any, *, arm: str, condition: str,
                        seeds: tuple[int | None, ...] = (),
                        has_env_factory: bool = False,
                        cross: bool = False) -> bool:
    """Gate for evaluation. Returns True for registered (held-out)
    evaluation, False for development (training levels 0-15, no context)."""
    lv = list(levels)
    held_out = any(j >= HELD_OUT_START for j in lv)
    if not held_out:
        if not set(lv) <= set(DEV_LEVELS):
            raise GuardError(
                "development evaluation is limited to the training levels "
                f"{DEV_LEVELS.start}-{DEV_LEVELS.stop - 1}")
        return False
    _need_ctx(ctx, "held-out evaluation")
    cells = registration(ctx.registration).arms_by_condition
    errs: list[str] = []
    if lv != list(HELD_OUT_RANGE):
        errs.append(f"levels must be exactly {HELD_OUT_RANGE.start}-"
                    f"{HELD_OUT_RANGE.stop - 1}")
    if condition not in cells or arm not in cells.get(condition, ()):
        errs.append(f"arm/condition {arm!r}/{condition!r} is not a registered cell")
    for s in seeds:
        if s is not None and s not in SEEDS:
            errs.append(f"seed {s} not in the registered seeds")
    if cross and len(seeds) == 2 and tuple(seeds) not in CROSS_PAIRS:
        errs.append(f"cross-play pair {tuple(seeds)} is not one of the fixed pairs")
    if has_env_factory:
        errs.append("a registered evaluation uses the real Procgen (no env_factory)")
    if errs:
        raise GuardError("held-out evaluation refused: " + "; ".join(errs))
    assert_x86_procgen_build()
    return True
