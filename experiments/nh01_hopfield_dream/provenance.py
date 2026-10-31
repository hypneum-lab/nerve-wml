"""Executed-code checks for the NH-01 size sweep.

Pre-registration: ``docs/milestones/2026-10-30-nh01-scale-sweep.md``
(§7.1, §8). ``check_loaded_modules`` classifies every loaded module by
the resolved path of its file (repository, environment, standard
library) and compares repository files with their SHA-256 at the tag.
``run_checked`` runs a task and repeats the check in a ``finally``
block of the same process, writing one JSON line per task to
``imports-<pid>.jsonl``, so that modules imported late inside a worker
are covered.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPECTED_PYTHON = (3, 12, 15)


class ImportCheckError(RuntimeError):
    """A loaded module is not allowed or differs from the tag."""


def version_ok(version_info: Any = None, expected: tuple[int, int, int] = EXPECTED_PYTHON) -> bool:
    """True iff major.minor.micro equal ``expected`` and the release is final."""
    vi = sys.version_info if version_info is None else version_info
    return tuple(vi[:3]) == tuple(expected) and vi[3] == "final"


@dataclass(frozen=True)
class ImportContext:
    repo_root: Path
    site_packages: Path
    stdlib_root: Path
    tag_hashes: Mapping[str, str]  # repository-relative path -> sha256 at the tag
    out_dir: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def check_loaded_modules(ctx: ImportContext, modules: Mapping[str, Any] | None = None) -> dict:
    mods = sys.modules if modules is None else modules
    repo = ctx.repo_root.resolve()
    site = ctx.site_packages.resolve()
    std = ctx.stdlib_root.resolve()
    report: dict = {"repository": {}, "environment": 0, "stdlib": 0, "violations": []}
    for name, mod in sorted(dict(mods).items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue  # built-in or frozen module
        p = Path(f).resolve()
        # environment first: a venv may live inside the repository tree
        if _under(p, site):
            report["environment"] += 1
        elif _under(p, std):
            report["stdlib"] += 1
        elif _under(p, repo):
            rel = p.relative_to(repo).as_posix()
            digest = _sha256(p)
            report["repository"][rel] = digest
            expected = ctx.tag_hashes.get(rel)
            if expected is None:
                report["violations"].append({"module": name, "path": rel, "reason": "not in tag"})
            elif expected != digest:
                report["violations"].append({"module": name, "path": rel, "reason": "hash differs"})
        else:
            report["violations"].append(
                {"module": name, "path": str(p), "reason": "outside allowed roots"}
            )
    return report


def run_checked(
    fn: Callable[[Any], Any],
    task: Any,
    ctx: ImportContext,
    modules: Mapping[str, Any] | None = None,
) -> Any:
    """Run ``fn(task)`` then re-check the loaded modules in this process."""
    failed = False
    try:
        return fn(task)
    except BaseException:
        failed = True
        raise
    finally:
        report = check_loaded_modules(ctx, modules)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        line = {"pid": os.getpid(), "task": repr(task), "task_failed": failed, **report}
        with (ctx.out_dir / f"imports-{os.getpid()}.jsonl").open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
        if report["violations"] and not failed:
            raise ImportCheckError(f"{len(report['violations'])} module violation(s)")
