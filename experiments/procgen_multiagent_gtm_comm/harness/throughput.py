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
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .codes import ARM_CODES, CONDITION_CODES

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
                       skip: int = STEADY_SKIP) -> dict:
    """Measure end-to-end training throughput of ``arm`` (untrained start).

    One job, real env (unless ``env_factory`` is given for tests), policy,
    bus and PPO update, on training levels only. Steady-state rate uses
    the per-update wall clock after ``skip`` warm-up cycles.
    """
    from .train import train_run

    stamps: list[float] = []

    def hook(update: int, buf: Any) -> None:  # after rollout, before update
        stamps.append(time.time())

    uptime_before = _uptime()
    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        rec = train_run(
            arm, SMOKE_SEED, "registered", n_updates * n_pairs * horizon,
            Path(td), env_factory=env_factory, n_pairs=n_pairs,
            horizon=horizon, log_every_pair_steps=n_pairs * horizon,
            ckpt_every_pair_steps=10**12, _rollout_hook=hook,
        )
        wall = time.time() - t0
    deltas = np.diff(stamps)  # cycle u = update u + rollout u+1
    steady = deltas[skip:] if len(deltas) > skip else deltas
    per_cycle = float(np.mean(steady)) if len(steady) else float("nan")
    ps_per_s = n_pairs * horizon / per_cycle if len(steady) else float("nan")
    out = {
        "arm": arm,
        "n_updates": n_updates,
        "n_pairs": n_pairs,
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
              log_every: int = 25_600) -> dict:
    """One smoke run; never raises: a crash is recorded, not hidden."""
    from .train import train_run

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
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="N10-C throughput / smoke runs")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("throughput")
    t.add_argument("--arms", nargs="+", default=["gtm", "nocomm"])
    t.add_argument("--n-updates", type=int, default=20)
    t.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--runs-dir", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--steps", type=int, default=SMOKE_PAIR_STEPS)
    a = p.parse_args()
    if a.cmd == "throughput":
        res = {"measured_on": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               "results": {}}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for arm in a.arms:
            res["results"][arm] = measure_throughput(arm, a.n_updates)
            a.out.write_text(json.dumps(res, indent=2))  # incremental
            print(json.dumps({arm: res["results"][arm]}), flush=True)
    else:
        runs = []
        a.out.parent.mkdir(parents=True, exist_ok=True)
        for arm, cond in SMOKE_RUNS:
            e = run_smoke(arm, cond, a.runs_dir, total=a.steps)
            runs.append(e)
            a.out.write_text(json.dumps(
                {"smoke_seed": SMOKE_SEED, "pair_steps_each": a.steps,
                 "runs": runs}, indent=2))
            print(json.dumps({k: e.get(k) for k in
                              ("arm", "condition", "outcome", "wall_time_s")}),
                  flush=True)


if __name__ == "__main__":
    main()
