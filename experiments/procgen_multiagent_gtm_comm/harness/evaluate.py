"""Evaluation and message-shuffle probe (pre-registered protocol).

Per held-out level j (``num_levels=1, start_level=j``): one cooperative
episode, actions sampled from the policy softmax for agent 0 then agent 1
with ``default_rng([20261023, j])`` (identical for both passes and every
model). Messages are deterministic (GTM argmax symbols, other arms hard
bits, no noise).

* Intact pass: records every emitted message; the bus enforces {0,1}.
* Shuffled pass: fixed derangement pi (``default_rng(20261023)``) of the
  evaluated levels; at step t of level j agent i receives the message the
  other role emitted at step min(t, T_pi(j) - 1) of intact level pi(j).
  Messages computed in the shuffled pass are not delivered.

S = R_intact - R_shuffled; bandwidth = plug-in entropy (bits) of the
16-bit messages over all intact steps. A checkpoint of ``None`` (training
failed before any finite checkpoint) evaluates the untrained policy built
from the run's seed stream. No torch / track_p / nerve_wml import.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from experiments.procgen_multiagent_gtm_comm.coop_env import ProcgenCoopEnv

from .bus import MessageBus
from .codes import ARM_CODES, CONDITION_CODES, init_key_int
from .policy import PairPolicy
from .train import EngCounters

EVAL_SEED = 20261023
HELD_OUT = range(1000, 1200)
MAX_EPISODE_STEPS = 5000  # safety net; procgen times out far earlier
_WEIGHTS = (1 << np.arange(15, -1, -1)).astype(np.int64)


@dataclass
class EvalRecord:
    arm: str
    condition: str
    seed: int | None
    ckpt: str | None
    untrained: bool
    levels: list[int]  # [start, stop)
    R_intact: float
    R_shuffled: float
    S: float
    completion_time: float | None
    bandwidth_bits: float
    returns_intact: list[float]
    returns_shuffled: list[float]
    lengths_intact: list[int]
    lengths_shuffled: list[int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def level_rng(j: int) -> np.random.Generator:
    return np.random.default_rng([EVAL_SEED, j])


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of range(n) without fixed points (rejection)."""
    if n < 2:
        raise ValueError("a derangement needs at least 2 levels")
    ident = np.arange(n)
    while True:
        p = rng.permutation(n)
        if not (p == ident).any():
            return p


def donor_received(donor_msgs: np.ndarray, t: int) -> np.ndarray:
    """Messages delivered at step t from an intact donor episode.

    ``donor_msgs`` [T, 2, 16] (row r = sender role r). Agent i receives
    role (1 - i)'s message at step min(t, T - 1). Returns [2, 16].
    """
    m = donor_msgs[min(t, donor_msgs.shape[0] - 1)]
    return np.stack([m[1], m[0]])


def bandwidth_entropy(messages: np.ndarray) -> float:
    """Plug-in entropy (bits) of the 16-bit messages, [..., 16] binary."""
    ints = (np.asarray(messages).reshape(-1, 16).astype(np.int64) * _WEIGHTS).sum(1)
    _, counts = np.unique(ints, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log2(p)).sum())
    return 0.0 if h == 0.0 else h  # avoid -0.0


def _sample_actions(logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Categorical draw per agent (inverse CDF, one uniform each, in order)."""
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    cdf = np.cumsum(p / p.sum(-1, keepdims=True), axis=-1)
    out = np.empty(2, dtype=np.int32)
    for i in range(2):
        out[i] = min(int(np.searchsorted(cdf[i], rng.random(), side="right")),
                     logits.shape[-1] - 1)
    return out


def _run_level(model, arm: str, team_summary: bool, j: int,
               env_factory, donor: np.ndarray | None, bus: MessageBus | None):
    """One episode on level j. Returns (return, length, emitted msgs [T,2,16])."""
    env = ProcgenCoopEnv(
        seed=j, team_summary=team_summary, num_levels=1, start_level=j,
        env_factory=env_factory,
    )  # allow_stub stays False
    try:
        rng = level_rng(j)
        use_eng = arm == "eng"
        counters = EngCounters(1)
        obs = np.stack(env._last_obs)[None]
        summ = np.zeros((1, 2, 2), np.float32)
        if team_summary:
            summ[0] = np.stack(env.team_summaries())
        key = mx.random.key(0)  # unused: evaluation is deterministic
        msgs: list[np.ndarray] = []
        t = 0
        while True:
            eng = None
            if use_eng:
                w = [env.window_view(i) for i in range(2)]
                eng = mx.array(counters.bits(
                    np.array([[w[0][0], w[1][0]]]),
                    np.array([[w[0][1], w[1][1]]]),
                ))
            kw = {}
            if donor is not None:
                kw["recv_bits"] = mx.array(donor_received(donor, t)[None])
            out = model(mx.array(obs).astype(mx.float32), mx.array(summ), eng,
                        train=False, key=key, **kw)
            bits = np.array(out.hard_bits)
            if donor is None:
                bus.deliver(bits, phase="eval")  # enforce {0,1}; intact only
                msgs.append(bits[0])
            actions = _sample_actions(np.array(out.logits)[0], rng)
            res = env.step([int(actions[0]), int(actions[1])])
            t += 1
            if res.done:
                return float(res.team_reward), t, (
                    np.stack(msgs) if donor is None else None)
            if t >= MAX_EPISODE_STEPS:
                raise RuntimeError(f"episode on level {j} exceeded {t} steps")
            counters.update(actions[None], np.array([False]),
                            np.array([res.info["finished"]]))
            obs = np.stack(res.obs_per_agent)[None]
            if team_summary:
                summ[0] = np.stack(res.summary_per_agent)
    finally:
        env.close()


def evaluate_policy(model, arm: str, condition: str, levels: range, *,
                    env_factory: Callable[[int], Any] | None = None,
                    seed: int | None = None, ckpt: str | None = None,
                    untrained: bool = False) -> EvalRecord:
    levels = list(levels)
    team_summary = condition == "registered"
    bus = MessageBus(arm)
    intact = []
    for j in levels:
        intact.append(_run_level(model, arm, team_summary, j, env_factory,
                                 None, bus))
    pi = derangement(len(levels), np.random.default_rng(EVAL_SEED))
    shuffled = [
        _run_level(model, arm, team_summary, j, env_factory,
                   intact[int(pi[k])][2], None)
        for k, j in enumerate(levels)
    ]
    r_i = [r for r, _, _ in intact]
    r_s = [r for r, _, _ in shuffled]
    l_i = [n for _, n, _ in intact]
    l_s = [n for _, n, _ in shuffled]
    succ = [n for r, n in zip(r_i, l_i, strict=False) if r > 0.5]
    r_int, r_shu = float(np.mean(r_i)), float(np.mean(r_s))
    return EvalRecord(
        arm=arm, condition=condition, seed=seed, ckpt=ckpt, untrained=untrained,
        levels=[levels[0], levels[-1] + 1], R_intact=r_int, R_shuffled=r_shu,
        S=r_int - r_shu,
        completion_time=float(np.mean(succ)) if succ else None,
        bandwidth_bits=bandwidth_entropy(np.concatenate([m for _, _, m in intact])),
        returns_intact=r_i, returns_shuffled=r_s,
        lengths_intact=l_i, lengths_shuffled=l_s,
    )


def evaluate_model(ckpt: Path | None, arm: str, condition: str,
                   levels: range = HELD_OUT, *, seed: int = 0,
                   env_factory: Callable[[int], Any] | None = None) -> EvalRecord:
    """Evaluate a checkpoint, or the run's untrained init if ``ckpt`` is None."""
    if arm not in ARM_CODES or condition not in CONDITION_CODES:
        raise ValueError(f"unknown arm/condition {arm!r}/{condition!r}")
    model = PairPolicy(arm, mx.random.key(init_key_int(seed, arm, condition)))
    if ckpt is not None:
        model.load_weights(str(ckpt))  # strict: every parameter must match
    mx.eval(model.parameters())
    return evaluate_policy(
        model, arm, condition, levels, env_factory=env_factory, seed=seed,
        ckpt=None if ckpt is None else Path(ckpt).name, untrained=ckpt is None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="N10-C evaluation + shuffle probe")
    p.add_argument("--arm", required=True, choices=sorted(ARM_CODES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", default="registered",
                   choices=sorted(CONDITION_CODES))
    p.add_argument("--ckpt", type=Path, default=None,
                   help="omit to evaluate the untrained init (training failure)")
    p.add_argument("--levels", type=int, nargs=2, default=[1000, 1200],
                   metavar=("START", "STOP"))
    p.add_argument("--out", type=Path, required=True)
    from experiments.procgen_multiagent_gtm_comm.campaign import guards

    guards.add_guard_args(p)
    a = p.parse_args()
    guards.enforce_cli(a, registered=a.levels[1] > guards.HELD_OUT_START,
                       what="held-out evaluation")
    rec = evaluate_model(a.ckpt, a.arm, a.condition, range(*a.levels),
                         seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(rec.to_json())
    print(json.dumps({k: v for k, v in asdict(rec).items()
                      if not k.startswith(("returns", "lengths"))}))


if __name__ == "__main__":
    main()
