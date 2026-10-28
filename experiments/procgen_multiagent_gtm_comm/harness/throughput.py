"""End-to-end throughput and smoke runs for the N10-C MLX harness.

Development tooling for the stage-2 manifest (amendment: smoke runs of at
most 1M steps, at most 3 per arm, disclosed; throughput on untrained
policies). Everything here uses TRAINING levels only (0-15); the
held-out levels are never touched. Run on macM1::

    python -m experiments.procgen_multiagent_gtm_comm.harness.throughput \
        throughput --out runs_dir/throughput.json
    python -m experiments.procgen_multiagent_gtm_comm.harness.throughput \
        smoke --out runs_dir/smoke_runs.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import devledger
from .codes import ARM_CODES, CONDITION_CODES, HOST_SETTINGS
from .options import options_for

PKG = Path(__file__).resolve().parent.parent

# Training levels are start_level=0, num_levels=16; the evaluation-cost
# proxy replays them (labelled as a proxy: not the 200 held-out levels).
TRAIN_LEVELS = range(0, 16)
EVAL_EPISODES_REGISTERED = 200  # per pass, registered protocol
EVAL_PASSES = 2  # intact + shuffled
SMOKE_PAIR_STEPS = 200_000
SMOKE_SEED = 9001  # not a registered seed: never shares a registered stream
SMOKE_RUNS: list[tuple[str, str]] = [
    (arm, "registered") for arm in ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
] + [("gtm", "closed")]
STEADY_SKIP = 2  # discard the first update cycles (mx.compile warm-up)

# Development allowance: see harness/devledger.py (one persistent per-arm
# ledger for every real-environment development run).
SMOKE_MAX_STEPS = devledger.MAX_STEPS
SMOKE_MAX_PER_ARM = devledger.MAX_PER_ARM
SMOKE_LEDGER = devledger.LEDGER
SmokeLimitError = devledger.SmokeLimitError
smoke_counts = devledger.counts


def reserve_smoke(arm: str, condition: str, steps: int,
                  path: Path = SMOKE_LEDGER) -> devledger.Reservation:
    return devledger.reserve(arm, condition, steps, kind="smoke", path=path)


def finish_smoke(res: devledger.Reservation, outcome: str) -> None:
    devledger.finish(res, outcome)


def _uptime() -> str:
    try:
        return subprocess.run(["uptime"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def eval_proxy_seconds(arm: str, *, levels: range = TRAIN_LEVELS,
                       env_factory: Callable[[int], Any] | None = None) -> dict:
    """Wall time of intact + shuffled passes over ``levels`` (training)."""
    from .evaluate import evaluate_model

    if max(levels) >= 1000:
        raise ValueError("evaluation proxy must use training levels only")
    t0 = time.time()
    rec = evaluate_model(None, arm, "registered", levels, seed=0,
                         env_factory=env_factory)
    dt = time.time() - t0
    episodes = EVAL_PASSES * len(levels)
    per_ep = dt / episodes
    steps = int(sum(rec.lengths_intact) + sum(rec.lengths_shuffled))
    return {
        "label": "PROXY: untrained policy on training levels "
                 f"{levels.start}-{levels.stop - 1}, not the held-out levels",
        "levels": [levels.start, levels.stop],
        "episodes": episodes,
        "seconds_total": dt,
        "seconds_per_episode": per_ep,
        "mean_episode_length": steps / episodes,
        "seconds_per_model_2x200": per_ep * EVAL_PASSES * EVAL_EPISODES_REGISTERED,
    }


def measure_throughput(arm: str, n_updates: int = 20, *,
                       env_factory: Callable[[int], Any] | None = None,
                       n_pairs: int = 64, horizon: int = 128,
                       eval_levels: range = TRAIN_LEVELS,
                       skip: int = STEADY_SKIP, n_workers: int = 1,
                       ledger: Path = SMOKE_LEDGER, learning: bool = False,
                       options: Any = None, init_trunk: Path | None = None,
                       micro_batch_pairs: int | None = None,
                       clear_cache: bool = False) -> dict:
    """Measure end-to-end training throughput of ``arm`` (untrained start).

    One job, real env (unless ``env_factory`` is given for tests), policy,
    bus and PPO update, on training levels only. The policy stays at its
    random initialisation (learning rate 0: "throughput measurements on
    random or untrained policies"); the run is ledgered as ``measurement``.
    ``learning=True`` (allowed by the amendment change of 2026-10-24) keeps
    learning on and ledgers the run as
    ``throughput_learning``.
    Steady-state rate uses the per-update wall clock after ``skip`` warm-up cycles.
    ``options`` / ``init_trunk`` / ``micro_batch_pairs`` / ``clear_cache``: a
    training preset (e.g. the registered fix C, with its pre-trained trunk)
    and per-host settings, for the fix-C throughput of the freeze manifest
    (freeze plan, C7); default: the frozen preset, full minibatch.
    """
    from .train import train_run

    if n_updates * n_pairs * horizon > SMOKE_MAX_STEPS:
        raise ValueError(
            f"throughput measurements are limited to {SMOKE_MAX_STEPS:,} "
            "pair-steps (development mode)")
    stamps: list[float] = []

    def hook(update: int, buf: Any) -> None:  # after rollout, before update
        stamps.append(time.time())

    uptime_before = _uptime()
    total = n_updates * n_pairs * horizon
    res = None
    if env_factory is None:  # real environment: ledgered before it starts
        res = devledger.reserve(
            arm, "registered", total, path=ledger,
            kind="throughput_learning" if learning else "measurement",
            inspected=["throughput"] if learning else None)
    outcome = "crashed"
    try:
        with tempfile.TemporaryDirectory() as td:
            t0 = time.time()
            rec = train_run(
                arm, SMOKE_SEED, "registered", total,
                Path(td), env_factory=env_factory, n_pairs=n_pairs,
                horizon=horizon, log_every_pair_steps=n_pairs * horizon,
                ckpt_every_pair_steps=10**12, _rollout_hook=hook,
                n_workers=n_workers, frozen_policy=not learning,
                dev_reservation=res, dev_ledger=ledger, options=options,
                init_trunk=init_trunk, micro_batch_pairs=micro_batch_pairs,
                clear_cache=clear_cache,
            )
            wall = time.time() - t0
        outcome = "failed_non_finite" if rec.training_failure else "completed"
    finally:
        if res is not None:
            devledger.finish(res, outcome)
    deltas = np.diff(stamps)  # cycle u = update u + rollout u+1
    steady = deltas[skip:] if len(deltas) > skip else deltas
    per_cycle = float(np.mean(steady)) if len(steady) else float("nan")
    ps_per_s = n_pairs * horizon / per_cycle if len(steady) else float("nan")
    out = {
        "arm": arm,
        "n_updates": n_updates,
        "n_pairs": n_pairs,
        "n_workers": n_workers,  # implementation setting, not registered
        "preset": getattr(options, "preset", "frozen-20261024"),
        "micro_batch_pairs": micro_batch_pairs, "clear_cache": clear_cache,
        "horizon": horizon,
        "pair_steps": rec.pair_steps_done,
        "wall_s_incl_warmup": wall,
        "pair_steps_per_s_incl_warmup": rec.pair_steps_done / wall,
        "steady_cycles": int(len(steady)),
        "steady_s_per_update_cycle": per_cycle,
        "steady_s_per_update_cycle_std": float(np.std(steady)) if len(steady) else None,
        "pair_steps_per_s": ps_per_s,
        "agent_samples_per_s": 2 * ps_per_s,
        "training_failure": rec.training_failure,
        "uptime_before": uptime_before,
        "uptime_after": _uptime(),
        "scope": "one job; real procgen env + policy + bus + PPO update; "
                 "training levels only; untrained start",
    }
    out["eval_proxy"] = eval_proxy_seconds(arm, levels=eval_levels,
                                           env_factory=env_factory)
    return out


def run_smoke(arm: str, condition: str, out_dir: Path, *,
              total: int = SMOKE_PAIR_STEPS,
              env_factory: Callable[[int], Any] | None = None,
              n_pairs: int = 64, horizon: int = 128,
              log_every: int = 25_600,
              ledger: Path = SMOKE_LEDGER) -> dict:
    """One smoke run; a crash is recorded, not hidden.

    Raises :class:`SmokeLimitError` BEFORE anything starts when the run
    would exceed 1M steps or the 3-runs-per-arm allowance; the run is
    entered in the persistent ledger first (training levels only).
    """
    from .train import train_run

    res = reserve_smoke(arm, condition, total, ledger)

    entry: dict[str, Any] = {
        "arm": arm, "condition": condition, "seed": SMOKE_SEED,
        "codes": {"arm": ARM_CODES[arm], "condition": CONDITION_CODES[condition]},
        "pair_steps_requested": total, "levels": "training (start_level=0, "
        "num_levels=16)", "outcome": None,
    }
    t0 = time.time()
    try:
        rec = train_run(
            arm, SMOKE_SEED, condition, total, out_dir / f"{arm}_{condition}",
            env_factory=env_factory, n_pairs=n_pairs, horizon=horizon,
            log_every_pair_steps=log_every, ckpt_every_pair_steps=10**12,
            dev_reservation=res, dev_ledger=ledger,
        )
        losses = np.array(rec.losses, dtype=float)
        curve = [r for _, r in rec.success_curve]
        entry.update({
            "outcome": "failed_non_finite" if rec.training_failure else "completed",
            "failure_reason": rec.failure_reason,
            "pair_steps_done": rec.pair_steps_done,
            "n_updates": rec.n_updates,
            "wall_time_s": rec.wall_time_s,
            "pair_steps_per_s_incl_warmup": rec.pair_steps_per_s,
            "using_stub": rec.using_stub,
            "all_losses_finite": bool(np.isfinite(losses).all()),
            "loss_first": float(losses[0]) if len(losses) else None,
            "loss_last": float(losses[-1]) if len(losses) else None,
            "completed_episodes": rec.completed_episodes,
            "final_success_rate_last100": curve[-1] if curve else None,
            "success_curve": rec.success_curve,
            "bits_logged": rec.bits_logged,
            "max_first_minibatch_ratio_dev": max(rec.first_mb_ratio_dev, default=None),
        })
    except Exception as exc:  # noqa: BLE001 - disclosed, not hidden
        entry.update({"outcome": "crashed", "error": repr(exc),
                      "traceback": traceback.format_exc(),
                      "wall_time_s": time.time() - t0})
    finish_smoke(res, entry["outcome"])
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="N10-C throughput / smoke runs")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--arms", nargs="+", default=["gtm", "nocomm"])
    t.add_argument("--n-updates", type=int, default=20)
    t.add_argument("--n-workers", type=int, default=1)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--preset", default=None,
                   help="training preset (default: the frozen one); fix C "
                        "needs --init-trunk")
    t.add_argument("--init-trunk", type=Path, default=None)
    t.add_argument("--host-settings", choices=sorted(HOST_SETTINGS), default=None,
                   help="apply that host's registered HOST_SETTINGS "
                        "(micro-batch, clear_cache, n_workers)")
    t.add_argument("--learning", action="store_true",
                   help="learning on (ledgered as throughput_learning)")
    s = sub.add_parser("smoke")
    s.add_argument("--runs-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--steps", type=int, default=SMOKE_PAIR_STEPS)
    s.add_argument("--ledger", type=Path, default=SMOKE_LEDGER,
                   help="persistent smoke-run accounting (under manifest/)")
    a = p.parse_args()
    if a.cmd == "smoke" and a.steps > SMOKE_MAX_STEPS:
        p.error(f"--steps is capped at {SMOKE_MAX_STEPS:,} (development "
                "allowance); larger runs only via the registered path")
    if a.cmd == "throughput":
        res = {"measured_on": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "results": {}}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for arm in a.arms:
            hs = HOST_SETTINGS[a.host_settings] if a.host_settings else {}
            res["results"][arm] = measure_throughput(
                arm, a.n_updates, n_workers=hs.get("n_workers", a.n_workers),
                learning=a.learning,
                options=None if a.preset is None else options_for(a.preset),
                init_trunk=a.init_trunk,
                micro_batch_pairs=hs.get("micro_batch_pairs"),
                clear_cache=hs.get("clear_cache", False))
            a.out.write_text(json.dumps(res, indent=2))  # incremental
            print(json.dumps({arm: res["results"][arm]}), flush=True)
    else:
        runs = []
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for arm, cond in SMOKE_RUNS:
            try:
                e = run_smoke(arm, cond, a.runs_dir, total=a.steps, ledger=a.ledger)
            except SmokeLimitError as exc:
                print(json.dumps({"arm": arm, "condition": cond,
                                  "skipped": str(exc)}), flush=True)
                continue
            runs.append(e)
            a.out.write_text(json.dumps(
                {"smoke_seed": SMOKE_SEED, "pair_steps_each": a.steps,
                 "runs": runs}, indent=2))
            print(json.dumps({k: e.get(k) for k in
                              ("arm", "condition", "outcome", "wall_time_s")}),
                  flush=True)


if __name__ == "__main__":
    main()
