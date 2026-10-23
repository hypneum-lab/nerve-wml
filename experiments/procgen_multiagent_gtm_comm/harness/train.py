"""PPO trainer for the N10-C MLX harness (pre-registered experiment).

One run = one (arm, seed, condition). All training randomness (init,
env level seeds, PPO shuffling, message / action sampling keys) is
derived from ``SeedSequence([seed, arm_code, cond_code])``. Every number
logged here feeds a registered endpoint:

* ``success_curve``: every ``log_every_pair_steps`` pair-steps, the mean
  team outcome over the last 100 completed coop episodes.
* ``time_to_half``: first logged step with rate >= 0.5 (window full).
* ``bits_logged``: from the message bus (enforced on every train step).

Checkpoints are safetensors, written only if every parameter is finite.
A non-finite loss or parameter stops the run (``training_failure``) and
the last finite checkpoint is kept (registered failure rule).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .bus import MessageBus
from .codes import ARM_CODES, CONDITION_CODES, init_key_int, run_seed_sequence
from .comm import eng_bits_from_window, plasticity_scale
from .policy import PairPolicy
from .rollout import RolloutBuffer
from .vec_env import VecCoopEnv

LR = 5e-4
CLIP = 0.2
GAMMA = 0.999
LAMBDA = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
EPOCHS = 3
MINIBATCH_PAIRS = 1024  # = 2048 agent-samples
SUCCESS_WINDOW = 100
RIGHT_ACTIONS = (6, 7, 8)
LEFT_ACTIONS = (0, 1, 2)


@dataclass
class RunRecord:
    arm: str
    seed: int
    condition: str
    codes: dict[str, int]
    total_pair_steps: int
    pair_steps_done: int = 0
    n_updates: int = 0
    wall_time_s: float = 0.0
    pair_steps_per_s: float = 0.0
    samples_per_s: float = 0.0  # agent-samples (2 per pair-step) per second
    restarts: int = 0
    using_stub: bool = False
    training_failure: bool = False
    failure_reason: str | None = None
    success_curve: list[list[float | None]] = field(default_factory=list)
    success_curve_n: list[int] = field(default_factory=list)  # episodes in window
    time_to_half: int | None = None
    bits_logged: int = 0
    last_checkpoint: str | None = None
    completed_episodes: int = 0
    losses: list[float] = field(default_factory=list)  # mean loss per update
    optimizer_steps: int = 0
    first_mb_ratio_dev: list[float] = field(default_factory=list)


class EngCounters:
    """Per-agent (#right - #left) action counters for the engineered sender.

    Counts only actions actually applied (an agent that already finished
    is frozen and its actions are ignored by the env), reset when that
    pair's coop episode ends, and builds the 16-bit message from the
    wrapper's window view.
    """

    def __init__(self, n_pairs: int) -> None:
        self.n = np.zeros((n_pairs, 2), dtype=np.int64)
        self.finished = np.zeros((n_pairs, 2), dtype=bool)

    def update(self, actions: np.ndarray, done: np.ndarray,
               finished: np.ndarray | None = None) -> None:
        a = np.asarray(actions)
        live = ~self.finished
        self.n += (np.isin(a, RIGHT_ACTIONS) & live).astype(np.int64)
        self.n -= (np.isin(a, LEFT_ACTIONS) & live).astype(np.int64)
        self.finished = (
            np.zeros_like(self.finished) if finished is None
            else np.asarray(finished, bool).copy()
        )
        d = np.asarray(done, bool)
        self.n[d] = 0
        self.finished[d] = False

    def bits(self, in_window: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n.shape[0], 2, 16), dtype=np.float32)
        for k in range(self.n.shape[0]):
            for i in range(2):
                out[k, i] = eng_bits_from_window(
                    bool(in_window[k, i]), int(remaining[k, i]), int(self.n[k, i])
                )
        return out


def _assert_not_stub(vec: VecCoopEnv) -> None:
    if vec.using_stub():
        raise RuntimeError(
            "procgen stub in use: a scientific run must use real procgen "
            "(infrastructure failure, run not started)"
        )


def _key(rng: np.random.Generator) -> mx.array:
    return mx.random.key(int(rng.integers(0, 2**31 - 1)))


def _params_finite(model: nn.Module) -> bool:
    flags = [mx.isfinite(p).all() for _, p in tree_flatten(model.parameters())]
    return bool(mx.all(mx.stack(flags)).item())


def _save_ckpt(model: nn.Module, out_dir: Path, step: int) -> str:
    final = out_dir / f"ckpt_{step}.safetensors"
    tmp = out_dir / f"ckpt_{step}.tmp.safetensors"
    mx.save_safetensors(str(tmp), dict(tree_flatten(model.parameters())))
    tmp.replace(final)
    return final.name


def _make_steps(model: PairPolicy, opt: optim.Optimizer, use_eng: bool,
                plastic: bool):
    def logp_value(model, obs, summary, eng, actions, noise, key):
        out = model(
            obs.astype(mx.float32), summary, eng if use_eng else None,
            train=True, key=key, noise=noise if model.noise_dim else None,
        )
        logp_all = nn.log_softmax(out.logits, axis=-1)  # [B, 2, 15]
        logp = mx.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]
        return out, logp_all, logp

    def loss_fn(model, obs, summary, eng, actions, noise, old_logp, adv, ret, key):
        out, logp_all, logp = logp_value(model, obs, summary, eng, actions,
                                         noise, key)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = mx.exp(logp - old_logp)
        pg = -mx.minimum(
            ratio * adv, mx.clip(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv
        ).mean()
        vf = ((out.value - ret) ** 2).mean()
        ent = -(mx.softmax(out.logits, axis=-1) * logp_all).sum(-1).mean()
        return pg + VALUE_COEF * vf - ENTROPY_COEF * ent

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=[model.state], outputs=[model.state])
    def grad_step(obs, summary, eng, actions, noise, old_logp, adv, ret, key,
                  scale):
        loss, grads = grad_fn(model, obs, summary, eng, actions, noise,
                              old_logp, adv, ret, key)
        if plastic:  # GTM constellation plasticity schedule
            grads["comm"]["constellation"] = grads["comm"]["constellation"] * scale
        grads, norm = optim.clip_grad_norm(grads, MAX_GRAD_NORM)
        return loss, norm, grads

    @partial(mx.compile, inputs=state, outputs=state)
    def apply_step(grads):
        opt.update(model, grads)
        return grads

    def ratio_dev(obs, summary, eng, actions, noise, old_logp, key):
        _, _, logp = logp_value(model, obs, summary, eng, actions, noise, key)
        return mx.abs(mx.exp(logp - old_logp) - 1.0).max()

    return grad_step, apply_step, ratio_dev


def train_run(
    arm: str,
    seed: int,
    condition: str,
    total_pair_steps: int,
    out_dir: Path,
    *,
    env_factory: Callable[[int], Any] | None = None,
    n_pairs: int = 64,
    horizon: int = 128,
    log_every_pair_steps: int = 1_000_000,
    ckpt_every_pair_steps: int = 1_000_000,
    minibatch_pairs: int = MINIBATCH_PAIRS,
    restarts: int = 0,
    n_workers: int = 1,
    _fault_hook: Callable[[int, PairPolicy], None] | None = None,
    _rollout_hook: Callable[[int, RolloutBuffer], None] | None = None,
) -> RunRecord:
    """Train one (arm, seed, condition) run; write ``out_dir/run.json``.

    Trains exactly ``total_pair_steps`` pair-steps (the last rollout is
    truncated). Success is measured at the exact boundaries
    k * ``log_every_pair_steps`` (a multiple of ``n_pairs``); checkpoints are
    written after the first update ending at or after each
    ``ckpt_every_pair_steps`` boundary (labelled with the actual step) and
    once at the exact end. ``_fault_hook`` / ``_rollout_hook`` are test-only.
    """
    if total_pair_steps % n_pairs or log_every_pair_steps % n_pairs:
        raise ValueError("total and log interval must be multiples of n_pairs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = run_seed_sequence(seed, arm, condition)
    env_ss, init_ss, shuf_ss, key_ss = ss.spawn(4)
    shuf_rng = np.random.default_rng(shuf_ss)
    key_rng = np.random.default_rng(key_ss)

    vec = VecCoopEnv(
        n_pairs, env_ss, team_summary=(condition == "registered"),
        env_factory=env_factory, n_workers=n_workers,
    )
    try:
        _assert_not_stub(vec)  # raises before any training
        return _train(
            vec, arm, seed, condition, total_pair_steps, out_dir, n_pairs,
            horizon, log_every_pair_steps, ckpt_every_pair_steps,
            minibatch_pairs, restarts, init_ss, shuf_rng, key_rng,
            _fault_hook, _rollout_hook,
        )
    finally:
        vec.close()


def _train(vec, arm, seed, condition, total, out_dir, n_pairs, horizon,
           log_every, ckpt_every, minibatch_pairs, restarts, init_ss,
           shuf_rng, key_rng, fault_hook, rollout_hook) -> RunRecord:
    rec = RunRecord(
        arm=arm, seed=seed, condition=condition,
        codes={"arm": ARM_CODES[arm], "condition": CONDITION_CODES[condition]},
        total_pair_steps=total, restarts=restarts, using_stub=False,
    )
    model = PairPolicy(arm, mx.random.key(init_key_int(seed, arm, condition)))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=LR)
    opt.init(model.trainable_parameters())
    mx.eval(opt.state)
    use_eng = arm == "eng"
    grad_step, apply_step, ratio_dev = _make_steps(
        model, opt, use_eng, plastic=(arm == "gtm")
    )

    bus = MessageBus(arm)
    nd = model.noise_dim
    counters = EngCounters(n_pairs)
    recent: deque[float] = deque(maxlen=SUCCESS_WINDOW)
    total_env_steps = total // n_pairs
    total_updates = max(1, -(-total_env_steps // horizon))
    zeros_eng = np.zeros((n_pairs, 2, 16), np.float32)

    po = vec.reset()
    obs, summ = po.obs, po.summary
    win = vec.window()
    step = 0  # pair-steps
    env_steps = 0
    t0 = time.time()

    def eng_now() -> np.ndarray:
        return counters.bits(*win) if use_eng else zeros_eng

    def mx_eng(e):
        return mx.array(e) if use_eng else None

    eng = eng_now()
    update = 0
    while env_steps < total_env_steps and not rec.training_failure:
        update += 1
        t_len = min(horizon, total_env_steps - env_steps)
        buf = RolloutBuffer(n_pairs, t_len, noise_dim=nd)
        for _ in range(t_len):
            k_pol, k_act, k_noise = _key(key_rng), _key(key_rng), _key(key_rng)
            noise = model.sample_noise(n_pairs, k_noise)
            out = model(
                mx.array(obs).astype(mx.float32), mx.array(summ), mx_eng(eng),
                train=True, key=k_pol, noise=noise,
            )
            act = mx.random.categorical(out.logits.reshape(-1, 15), key=k_act)
            act = act.reshape(n_pairs, 2)
            logp_all = nn.log_softmax(out.logits, axis=-1)
            logp = mx.take_along_axis(logp_all, act[..., None], axis=-1)[..., 0]
            to_eval = [act, logp, out.value, out.hard_bits]
            if noise is not None:
                to_eval.append(noise)
            mx.eval(*to_eval)
            actions = np.array(act)
            bus.deliver(np.array(out.hard_bits), phase="train")  # enforce + log
            res = vec.step(actions.astype(np.int32))
            buf.add(obs, summ, eng, actions, np.array(logp), np.array(out.value),
                    res.team_reward, res.done,
                    noise=None if noise is None else np.array(noise))
            for k in np.flatnonzero(res.done):
                recent.append(float(res.team_reward[k]))
                rec.completed_episodes += 1
            counters.update(actions, res.done, res.finished)
            obs, summ, win = res.obs, res.summary, res.window
            eng = eng_now()
            env_steps += 1
            step += n_pairs
            if step % log_every == 0:  # exact boundary
                rate = float(np.mean(recent)) if recent else None
                rec.success_curve.append([step, rate])
                rec.success_curve_n.append(len(recent))
                if (rec.time_to_half is None and rate is not None
                        and len(recent) == SUCCESS_WINDOW and rate >= 0.5):
                    rec.time_to_half = step
        k_last, k_lnoise = _key(key_rng), _key(key_rng)
        last = model(
            mx.array(obs).astype(mx.float32), mx.array(summ), mx_eng(eng),
            train=True, key=k_last, noise=model.sample_noise(n_pairs, k_lnoise),
        )
        mx.eval(last.value)
        buf.finish(np.array(last.value), GAMMA, LAMBDA)
        if rollout_hook is not None:
            rollout_hook(update, buf)

        scale = mx.array(plasticity_scale(update, total_updates), dtype=mx.float32)
        losses: list[float] = []
        first_mb = True
        for _ in range(EPOCHS):
            for mb in buf.minibatches(shuf_rng, minibatch_pairs):
                args = (mb["obs"], mb["summary"], mb["eng_bits"], mb["actions"],
                        mb["noise"])
                if first_mb:  # unchanged policy: ratio must be 1
                    dev = ratio_dev(*args, mb["logp"], _key(key_rng))
                    rec.first_mb_ratio_dev.append(float(dev.item()))
                    first_mb = False
                loss, norm, grads = grad_step(
                    *args, mb["logp"], mb["advantages"], mb["returns"],
                    _key(key_rng), scale,
                )
                mx.eval(loss, norm)
                lv = float(loss.item())
                if not (np.isfinite(lv) and np.isfinite(float(norm.item()))):
                    rec.training_failure = True
                    break  # detected before applying: no optimizer step
                losses.append(lv)
                apply_step(grads)
                mx.eval(model.state, opt.state)
                rec.optimizer_steps += 1
                if not _params_finite(model):
                    rec.training_failure = True
                    break
            if rec.training_failure:
                break
        rec.n_updates = update
        rec.pair_steps_done = step
        if rec.training_failure:
            rec.failure_reason = (
                f"non-finite loss or parameter at update {update} "
                f"(pair-step {step})"
            )
            break
        rec.losses.append(float(np.mean(losses)))
        prev = step - t_len * n_pairs
        if step // ckpt_every > prev // ckpt_every:
            rec.last_checkpoint = _save_ckpt(model, out_dir, step)
        if fault_hook is not None:
            fault_hook(update, model)

    if not rec.training_failure and rec.last_checkpoint != f"ckpt_{step}.safetensors":
        rec.last_checkpoint = _save_ckpt(model, out_dir, step)  # exact end

    rec.wall_time_s = time.time() - t0
    if rec.wall_time_s > 0:
        rec.pair_steps_per_s = rec.pair_steps_done / rec.wall_time_s
        rec.samples_per_s = 2 * rec.pair_steps_per_s
    rec.bits_logged = bus.bits_logged
    (out_dir / "run.json").write_text(json.dumps(asdict(rec), indent=2))
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description="N10-C MLX PPO trainer")
    p.add_argument("--arm", required=True, choices=sorted(ARM_CODES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--condition", default="registered",
                   choices=sorted(CONDITION_CODES))
    p.add_argument("--steps", type=int, default=50_000_000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-pairs", type=int, default=64)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--log-every", type=int, default=1_000_000)
    p.add_argument("--ckpt-every", type=int, default=1_000_000)
    p.add_argument("--n-workers", type=int, default=1)
    a = p.parse_args()
    rec = train_run(
        a.arm, a.seed, a.condition, a.steps, a.out, n_pairs=a.n_pairs,
        horizon=a.horizon, log_every_pair_steps=a.log_every,
        ckpt_every_pair_steps=a.ckpt_every, n_workers=a.n_workers,
    )
    print(json.dumps({k: v for k, v in asdict(rec).items() if k != "losses"}))


if __name__ == "__main__":
    main()
