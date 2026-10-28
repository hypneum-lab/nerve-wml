"""Fake pre-freeze registered trunk records (no MLX, no GPU) for the tests
of campaign.pin_trunks and of the stage-2 manifest's trunk table."""
import hashlib
import json
from pathlib import Path

from experiments.procgen_multiagent_gtm_comm.campaign import guards
from experiments.procgen_multiagent_gtm_comm.campaign.queue import control_queue
from experiments.procgen_multiagent_gtm_comm.harness.codes import (
    HOST_SETTINGS,
    SEED_HOST,
    SEEDS,
)


def units() -> list[tuple[int, list[str] | None]]:
    return [(s, None) for s in SEEDS] + [
        (r.seed, [r.arm, r.condition]) for r in control_queue()]


def name(seed: int, unit) -> str:
    return f"seed-{seed}" if unit is None else f"seed-{seed}-{unit[1]}-{unit[0]}"


def make(dst: Path, seed: int, unit=None, *, usable=True, host=None,
         settings=None, payload=None, commit="a" * 40) -> dict:
    """Write ``dst/{trunk.safetensors,pretrain.json}``; return the record."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    data = payload if payload is not None else f"trunk {seed} {unit}".encode()
    (dst / "trunk.safetensors").write_bytes(data)
    host = host or SEED_HOST[seed]
    hs = HOST_SETTINGS[SEED_HOST[seed]]
    rec = {
        "seed": seed, "total_steps": guards.PRETRAIN_BUDGET,
        "steps_done": guards.PRETRAIN_BUDGET, "usable": usable,
        "trunk_file": "trunk.safetensors",
        "trunk_sha256": hashlib.sha256(data).hexdigest(),
        "settings": settings if settings is not None else {
            "micro_batch_pairs": hs["micro_batch_pairs"],
            "clear_cache": hs["clear_cache"]},
        "unit": unit, "seed_sequence": [seed, 99] + ([0, 0] if unit else []),
        "coin_rate_curve": [[106496, 0.4], [1000000, 0.8]],
        "pre_freeze": {"mode": "pre-freeze registered pre-training",
                       "commit": commit, "host": host,
                       "registered_preset": "fix-c-coin-pretrain",
                       "budget": guards.PRETRAIN_BUDGET, "unit": unit},
    }
    (dst / "pretrain.json").write_text(json.dumps(rec))
    return rec


def make_all(root: Path) -> dict:
    """All 16 trunks under ``root``; returns {name: sha256}."""
    return {name(s, u): make(Path(root) / name(s, u), s, u)["trunk_sha256"]
            for s, u in units()}


def pin_into_guards(monkeypatch, shas: dict) -> None:
    for s, u in units():
        h = shas[name(s, u)]
        if u is None:
            monkeypatch.setitem(guards.REGISTERED_TRUNK_SHA256, s, h)
        else:
            monkeypatch.setitem(guards.REGISTERED_CONTROL_TRUNK_SHA256, tuple(u), h)
