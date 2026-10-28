"""Prepare the freeze commit deterministically (no git side effects).

    python -m experiments.procgen_multiagent_gtm_comm.campaign.freeze_prepare \
        --tag n10c-freeze-YYYYMMDD [--registration n10c] [--manifest-name NAME]

The amendment and the manifest name come from the selected registration
(``guards.REGISTRATIONS``: for N10-C the 2026-10-25 amendment and
``stage2-manifest-fixc.md``); the files frozen by the earlier tag
(``guards.FROZEN_PATHS``) are never written.
Rewrites the Status line of the stage-1 amendment to the final, tag-naming
form (``**Status:** FROZEN by tag <tag> (stage 1 + stage 2 manifest); OSF
registration to follow``) and regenerates the stage-2 manifest with
``Status: FROZEN (tag <tag>)`` (run on macM1: it needs MLX and collects the
toolchain there). It then prints the tag message to use. The controller
commits both files, then creates the annotated tag on that commit:

    git add -A && git commit -F msg && git tag -a <tag> -F <printed message>
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from . import guards


def rewrite_amendment_status(text: str, tag: str) -> str:
    """Replace the (single) Status line; raise if absent or ambiguous."""
    if not guards.valid_tag_name(tag):
        raise ValueError(f"invalid freeze tag {tag!r}")
    if len(guards.AMEND_STATUS_LINE_RE.findall(text)) != 1:
        raise ValueError("amendment must have exactly one '**Status:**' line")
    return guards.AMEND_STATUS_LINE_RE.sub(
        lambda _m: guards.amendment_status_line(tag), text, count=1)


def prepare(repo: Path, tag: str, manifest_name: str | None = None,
            write_manifest: Callable[[Path, str], None] | None = None,
            registration: str = guards.DEFAULT_REGISTRATION) -> dict:
    """Rewrite the amendment status and regenerate the manifest in ``repo``
    (paths from the registration; never a path of ``guards.FROZEN_PATHS``)."""
    repo = Path(repo)
    reg = guards.registration(registration)
    if not guards.valid_tag_name(tag, reg.name):
        raise ValueError(f"invalid freeze tag {tag!r} for {reg.name}")
    manifest_name = manifest_name or reg.manifest_name
    if "DRAFT" in manifest_name:
        raise ValueError("a frozen manifest file name must not contain DRAFT")
    amend_rel = reg.amendment_path
    out_rel = (guards.MANIFEST_DIR / manifest_name).as_posix()
    for rel in (amend_rel, out_rel):
        if rel in guards.FROZEN_PATHS:
            raise ValueError(f"{rel} is frozen by an earlier tag and is never "
                             "written again")
    if write_manifest is None and reg.name != guards.DEFAULT_REGISTRATION:
        raise ValueError(f"no stage-2 manifest writer for {reg.name} yet")
    amend = repo / amend_rel
    amend.write_text(rewrite_amendment_status(amend.read_text(), tag))
    out = repo / out_rel
    if write_manifest is None:
        from ..harness.manifest import write_manifest as write_manifest_default

        def write_manifest(o: Path, t: str) -> None:
            write_manifest_default(o, freeze_tag=t)
    write_manifest(out, tag)
    text = out.read_text()
    if not re.search(r"^\W*Status:\s*FROZEN \(tag " + re.escape(tag) + r"\)",
                     text, re.MULTILINE) or "DRAFT" in text:
        raise ValueError("regenerated manifest is not a valid frozen manifest")
    rel = out.relative_to(repo).as_posix()
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"manifest": rel, "manifest_sha256": sha,
            "tag_message": f"freeze {tag}\n\nmanifest: {rel}\nmanifest-sha256: {sha}\n"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--registration", default=guards.DEFAULT_REGISTRATION,
                   choices=sorted(guards.REGISTRATIONS))
    p.add_argument("--manifest-name", default=None,
                   help="default: the registration's manifest name")
    p.add_argument("--repo", type=Path, default=guards.REPO)
    a = p.parse_args(argv)
    res = prepare(a.repo, a.tag, a.manifest_name, registration=a.registration)
    print(res["tag_message"], end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
