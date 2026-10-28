"""Pin the pre-freeze registered trunks (freeze plan, phase M, step 3).

    python -m experiments.procgen_multiagent_gtm_comm.campaign.pin_trunks \
        OUT_DIR [OUT_DIR ...] [--repo R]

Each OUT_DIR is the output of one ``harness.pretrain --pre-freeze`` run
(``pretrain.json`` + ``trunk.safetensors``). Every record is checked (usable
trunk, pre-freeze procedure on the seed's fixed host with that host's
settings, exact budget, registered seed or control unit, SHA-256 of the
file), then copied into ``<repo>/<MANIFEST_DIR>/trunks/<name>/`` (the layout
of ``guards.trunk_path`` / ``guards.control_trunk_path``) so that the freeze
tag and its Software Heritage snapshot contain them (PI decision 2026-10-28),
and ``campaign/trunk_pins.py`` is rewritten from ALL trunks found there.
No GPU, no MLX. Prints what is still missing (6 shared + 10 control).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ..harness.codes import HOST_SETTINGS, PER_RUN_TRUNK_SEED, SEED_HOST, SEEDS
from . import guards

TRUNK_FILE = "trunk.safetensors"
PINS_REL = Path("experiments/procgen_multiagent_gtm_comm/campaign/trunk_pins.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_name(seed: int, unit: list[str] | None) -> str:
    if unit is None:
        return f"seed-{seed}"
    arm, cond = unit
    return f"seed-{seed}-{cond}-{arm}"


def control_units() -> list[tuple[str, str]]:
    return [(a, c) for c, arms in guards.ARMS_BY_CONDITION.items() for a in arms]


def check_record(src: Path) -> dict:
    """The validated pretrain record of ``src``; raise ValueError if not a
    usable pre-freeze registered trunk."""
    src = Path(src)
    rec = json.loads((src / "pretrain.json").read_text())
    errs = []
    seed, unit, pf = rec.get("seed"), rec.get("unit"), rec.get("pre_freeze")
    if seed not in SEEDS:
        errs.append(f"seed {seed} not registered")
    if not isinstance(pf, dict) or pf.get("mode") != "pre-freeze registered pre-training":
        errs.append("not a pre-freeze registered pre-training")
        pf = {}
    if not rec.get("usable"):
        errs.append("trunk not usable (one restart, then the PI decides)")
    if not (rec.get("total_steps") == rec.get("steps_done") == guards.PRETRAIN_BUDGET):
        errs.append("budget not completed exactly")
    proc = {k: rec.get(k) for k in guards.PRETRAIN_PROCEDURE}
    if proc != guards.PRETRAIN_PROCEDURE:
        errs.append(f"procedure {proc} != registered {guards.PRETRAIN_PROCEDURE}")
    every = guards.PRETRAIN_PROCEDURE["liveness_every"]
    steps = [m.get("step") for m in rec.get("liveness") or []]
    want = list(range(guards.PRETRAIN_BUDGET // every + 1))
    if (not steps or steps[0] != 0
            or [s // every for s in steps if isinstance(s, int)] != want
            or any(m.get("flagged") for m in rec.get("liveness") or [])):
        errs.append("liveness probes do not cover the schedule, or one is flagged")
    host = SEED_HOST.get(seed)
    if pf.get("host") != host:
        errs.append(f"host {pf.get('host')!r} is not the seed's host {host!r}")
    if host is not None:
        want = {k: HOST_SETTINGS[host][k] for k in ("micro_batch_pairs", "clear_cache")}
        if rec.get("settings") != want:
            errs.append(f"settings {rec.get('settings')} != {host} {want}")
    if unit is not None:
        if seed != PER_RUN_TRUNK_SEED or tuple(unit) not in control_units():
            errs.append(f"unit {unit} is not a control unit of seed {PER_RUN_TRUNK_SEED}")
    if pf.get("unit") != unit:
        errs.append("unit of the procedure differs from the record")
    f = src / (rec.get("trunk_file") or TRUNK_FILE)
    if not f.is_file():
        errs.append(f"trunk file {f.name} missing")
    elif _sha(f) != rec.get("trunk_sha256"):
        errs.append("trunk SHA-256 differs from pretrain.json")
    if errs:
        raise ValueError(f"{src}: " + "; ".join(errs))
    return rec


def collect(trunk_root: Path) -> tuple[dict[int, str], dict[str, str]]:
    """Pins from the trunks under ``trunk_root`` (re-checked)."""
    shared: dict[int, str] = {}
    control: dict[str, str] = {}
    for d in sorted(Path(trunk_root).glob("seed-*")):
        rec = check_record(d)
        if d.name != unit_name(rec["seed"], rec["unit"]):
            raise ValueError(f"{d}: directory name does not match its record")
        if rec["unit"] is None:
            shared[rec["seed"]] = rec["trunk_sha256"]
        else:
            arm, cond = rec["unit"]
            control[f"{cond}/{arm}"] = rec["trunk_sha256"]
    return shared, control


def pins_source(shared: dict[int, str], control: dict[str, str],
                repo: Path = guards.REPO) -> str:
    f = Path(repo) / PINS_REL
    doc = (f.read_text().split('"""')[1] if f.is_file()
           else "Pinned trunk SHA-256.\n")
    lines = ['"""' + doc + '"""', "SHARED: dict[int, str] = {"]
    lines += [f"    {s}: \"{h}\"," for s, h in sorted(shared.items())]
    lines += ["}", "CONTROL: dict[str, str] = {"]
    lines += [f"    \"{k}\": \"{h}\"," for k, h in sorted(control.items())]
    lines += ["}", ""]
    return "\n".join(lines)


def pin(sources: list[Path], repo: Path = guards.REPO) -> dict:
    repo = Path(repo)
    root = repo / guards.MANIFEST_DIR / "trunks"
    for src in sources:
        rec = check_record(src)
        dst = root / unit_name(rec["seed"], rec["unit"])
        if dst.exists():
            old = json.loads((dst / "pretrain.json").read_text())
            if old.get("trunk_sha256") != rec["trunk_sha256"]:
                raise ValueError(f"{dst} already pins another trunk")
            continue
        dst.mkdir(parents=True)
        shutil.copy2(Path(src) / (rec.get("trunk_file") or TRUNK_FILE), dst / TRUNK_FILE)
        shutil.copy2(Path(src) / "pretrain.json", dst / "pretrain.json")
    shared, control = collect(root)
    (repo / PINS_REL).write_text(pins_source(shared, control, repo))
    missing = [f"seed-{s}" for s in SEEDS if s not in shared]
    missing += [f"seed-{PER_RUN_TRUNK_SEED}-{c}-{a}" for a, c in control_units()
                if f"{c}/{a}" not in control]
    return {"shared": shared, "control": control, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--repo", type=Path, default=guards.REPO)
    a = p.parse_args(argv)
    res = pin(a.sources, a.repo)
    print(json.dumps(res, indent=1))
    return 0 if not res["missing"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
