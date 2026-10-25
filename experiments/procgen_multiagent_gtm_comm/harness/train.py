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
import hashlib
import json
import sys
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
from mlx.utils import tree_flatten, tree_map

from ..campaign import guards
from . import devledger
from .bus import MessageBus
from .codes import (
    ARM_CODES,
    CONDITION_CODES,
    HOST_SETTINGS,
    HOSTS,
    SEED_HOST,
    init_key_int,
    run_seed_sequence,
)
from .comm import eng_bits_from_window, plasticity_scale
from .options import (  # noqa: F401 - re-exported for callers and tests
    FROZEN_PRESET,
    PRESETS,
    TrainOptions,
    options_for,
)
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

# Trunk-liveness guard (amendment 2026-10-25). A probe on frames of the
# current rollout flags the trunk as dead when the positive fraction of the
# last IMPALA block's residual stream, or the fraction of the 256 h units
# active on at least one probe frame, falls below LIVENESS_MIN_FRAC, or
# when the policy logits or the values do not vary across the probe batch.
LIVENESS_MIN_FRAC = 0.01
LIVENESS_STD_EPS = 1e-6
LIVENESS_PROBE_PAIRS = 128  # pair-steps per probe (= 256 agent frames)


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
    # implementation settings (never registered quantities)
    micro_batch_pairs: int | None = None
    clear_cache: bool = False
    n_workers: int = 1
    peak_memory_bytes: int | None = None
    # provenance of a registered run (None in the development mode)
    registration: dict | None = None
    runtime: dict | None = None
    # amendment 2026-10-25: training options and diagnostics
    train_options: dict | None = None
    init_trunk_sha256: str | None = None
    coin_rate_curve: list[list[float | None]] = field(default_factory=list)
    entropy_curve: list[list[float]] = field(default_factory=list)  # [step, mean, min]
    liveness: list[dict] = field(default_factory=list)
    liveness_flagged: bool = False
    liveness_abort: bool = False


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


def normalize_adv(adv: mx.array, live: mx.array | None = None) -> mx.array:
    """Advantage normalisation over the full minibatch (before splitting).

    With ``live`` (1 = agent not yet finished), mean and std are taken over
    live samples only and finished samples get advantage 0."""
    if live is None:
        return (adv - adv.mean()) / (adv.std() + 1e-8)
    n = mx.maximum(live.sum(), 1.0)
    mean = (adv * live).sum() / n
    std = mx.sqrt((((adv - mean) ** 2) * live).sum() / n)
    return (adv - mean) / (std + 1e-8) * live


def live_scale(live: mx.array) -> mx.array:
    """1 / (live fraction of the FULL minibatch): turns a micro-batch mean
    of masked terms into the masked mean over the minibatch's live samples
    (micro weights = micro size / minibatch size)."""
    return live.size / mx.maximum(live.sum(), 1.0)


def micro_slices(n: int, micro: int | None) -> list[tuple[int, int, float]]:
    """(start, stop, weight=size/n) micro-batches of a minibatch of n pairs."""
    if micro is None or micro >= n:
        return [(0, n, 1.0)]
    return [(a, min(a + micro, n), (min(a + micro, n) - a) / n)
            for a in range(0, n, micro)]


def accumulate_minibatch(grad_step, mb: dict, micro: int | None, key,
                         mask: bool = False):
    """Weighted gradient of the mean loss over one registered minibatch.

    Returns ``(loss, grads, finite)``; ``grads`` is None (and ``finite``
    False) as soon as one micro loss or gradient is non-finite, so the
    caller can stop before applying anything.
    """
    n = mb["actions"].shape[0]
    if mask:
        live = mb["live"]
        adv = normalize_adv(mb["advantages"], live)  # live samples only
        inv = live_scale(live)
    else:
        live = mx.ones(mb["actions"].shape, dtype=mx.float32)
        adv = normalize_adv(mb["advantages"])  # full minibatch statistics
        inv = mx.array(1.0, dtype=mx.float32)
    acc, total = None, 0.0
    for a, b, w in micro_slices(n, micro):
        loss, norm, g = grad_step(
            mb["obs"][a:b], mb["summary"][a:b], mb["eng_bits"][a:b],
            mb["actions"][a:b], mb["noise"][a:b], mb["logp"][a:b], adv[a:b],
            mb["returns"][a:b], key, mx.array(w, dtype=mx.float32),
            live[a:b], inv,
        )
        acc = g if acc is None else tree_map(lambda x, y: x + y, acc, g)
        mx.eval(loss, norm, acc)
        total += float(loss.item())
        if not (np.isfinite(total) and np.isfinite(float(norm.item()))):
            return float("nan"), None, False
    return total, acc, True


def finalize_grads(grads, scale, plastic: bool = False):
    """Plasticity scale (GTM) then clip the ACCUMULATED gradient norm."""
    if plastic:
        grads["comm"]["constellation"] = grads["comm"]["constellation"] * scale
    return optim.clip_grad_norm(grads, MAX_GRAD_NORM)


def _make_steps(model: PairPolicy, opt: optim.Optimizer, use_eng: bool,
                plastic: bool, mask_finished: bool = False):
    def logp_value(model, obs, summary, eng, actions, noise, key):
        out = model(
            obs.astype(mx.float32), summary, eng if use_eng else None,
            train=True, key=key, noise=noise if model.noise_dim else None,
        )
        logp_all = nn.log_softmax(out.logits, axis=-1)  # [B, 2, 15]
        logp = mx.take_along_axis(logp_all, actions[..., None], axis=-1)[..., 0]
        return out, logp_all, logp

    def loss_fn(model, obs, summary, eng, actions, noise, old_logp, adv, ret,
                key, live, inv_live):
        out, logp_all, logp = logp_value(model, obs, summary, eng, actions,
                                         noise, key)
        ratio = mx.exp(logp - old_logp)
        surr = mx.minimum(
            ratio * adv, mx.clip(ratio, 1.0 - CLIP, 1.0 + CLIP) * adv
        )
        ent_each = -(mx.softmax(out.logits, axis=-1) * logp_all).sum(-1)
        vf = ((out.value - ret) ** 2).mean()
        if mask_finished:  # finished agents: no policy / entropy term
            pg = -(surr * live).mean() * inv_live
            ent = (ent_each * live).mean() * inv_live
        else:  # frozen-tag expression (live / inv_live unused)
            pg = -surr.mean()
            ent = ent_each.mean()
        return pg + VALUE_COEF * vf - ENTROPY_COEF * ent

    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    # Micro grad step: ``adv`` is already normalised over the FULL
    # minibatch; ``weight`` = micro_size / minibatch_size. Returns the
    # weighted loss and gradient (unclipped; plasticity scale and clipping
    # act on the accumulated gradient in ``finalize_grads``). Shapes are
    # fixed per micro size; a smaller last micro recompiles once.
    @partial(mx.compile, inputs=[model.state], outputs=[model.state])
    def grad_step(obs, summary, eng, actions, noise, old_logp, adv, ret, key,
                  weight, live, inv_live):
        loss, grads = grad_fn(model, obs, summary, eng, actions, noise,
                              old_logp, adv, ret, key, live, inv_live)
        grads = tree_map(lambda g: g * weight, grads)
        sq = sum((g * g).sum() for _, g in tree_flatten(grads))
        return loss * weight, mx.sqrt(sq), grads

    @partial(mx.compile, inputs=state, outputs=state)
    def apply_step(grads):
        opt.update(model, grads)
        return grads

    def ratio_dev(obs, summary, eng, actions, noise, old_logp, key):
        _, _, logp = logp_value(model, obs, summary, eng, actions, noise, key)
        return mx.abs(mx.exp(logp - old_logp) - 1.0).max()

    return grad_step, apply_step, ratio_dev


def trunk_activity(trunk, x: mx.array) -> dict[str, float]:
    """Liveness of the IMPALA trunk on agent frames ``x`` [N, 64, 64, 3] in
    [0, 1]: positive fraction of the last block's residual stream, fraction
    of h units active on >= 1 frame, and overall h nonzero fraction."""
    y = x
    for block in trunk.blocks:
        y = block(y)
    h = nn.relu(trunk.fc(nn.relu(y).reshape(y.shape[0], -1)))
    act = h > 0
    out = (((y > 0).mean()), act.any(axis=0).mean(), act.mean())
    mx.eval(*out)
    return {"block3_pos_frac": float(out[0].item()),
            "live_h_frac": float(out[1].item()),
            "h_nonzero_frac": float(out[2].item())}


def liveness_probe(model: PairPolicy, obs: np.ndarray, summary: np.ndarray,
                   eng: np.ndarray | None, key: mx.array) -> dict:
    """Trunk-liveness metrics on a batch of pair-steps (uint8 obs
    [B, 2, 64, 64, 3]) and the flag of the guard (see LIVENESS_*)."""
    o = mx.array(obs).astype(mx.float32)
    m = trunk_activity(model.trunk, o.reshape(-1, *o.shape[2:]) / 255.0)
    out = model(o, mx.array(summary), None if eng is None else mx.array(eng),
                train=False, key=key)
    logits = out.logits.reshape(-1, out.logits.shape[-1])
    lstd = logits.std(axis=0).mean()
    vstd = out.value.reshape(-1).std()
    mx.eval(lstd, vstd)
    m["logit_std"] = float(lstd.item())
    m["value_std"] = float(vstd.item())
    reasons = [k for k in ("block3_pos_frac", "live_h_frac")
               if m[k] < LIVENESS_MIN_FRAC]
    reasons += [k for k in ("logit_std", "value_std") if m[k] <= LIVENESS_STD_EPS]
    m["flagged"] = bool(reasons)
    m["reasons"] = reasons
    return m


def make_optimizer(options: TrainOptions,
                   frozen_policy: bool = False) -> optim.Adam:
    """Adam with the preset's epsilon and bias correction (off, the MLX
    default, in the frozen tag and every preset except the ``-bc`` one)."""
    return optim.Adam(learning_rate=0.0 if frozen_policy else LR,
                      eps=options.adam_eps,
                      bias_correction=options.adam_bias_correction)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trunk(model: PairPolicy, path: Path) -> None:
    """Initialise ``model.trunk`` from a pre-training trunk checkpoint
    (strict: every trunk parameter, nothing else)."""
    model.trunk.load_weights(list(mx.load(str(path)).items()), strict=True)
    mx.eval(model.trunk.parameters())


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
    micro_batch_pairs: int | None = None,
    clear_cache: bool = False,
    restarts: int = 0,
    n_workers: int = 1,
    registration: guards.RegistrationContext | None = None,
    frozen_policy: bool = False,
    dev_reservation: devledger.Reservation | None = None,
    dev_ledger: Path = devledger.LEDGER,
    options: TrainOptions | None = None,
    init_trunk: Path | None = None,
    dev_kind: str = "dev_train",
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

    ``micro_batch_pairs`` (None = off) splits each registered minibatch into
    gradient-accumulated micro-batches: same update, lower peak memory; it
    is an implementation setting, not a registered quantity.
    ``clear_cache`` calls ``mx.clear_cache()`` once per rollout update.

    Execution boundary: a registered run (``registration`` = the context
    issued by ``campaign.guards``) must be exactly the registered tuple
    (seed in SEEDS on its fixed host, arm allowed in the condition, 50M
    pair-steps, frozen rollout / measurement configuration) with exactly the
    per-host ``HOST_SETTINGS``, on the actual host and backend. Without a
    context only the development mode is available (<= 1M pair-steps).

    Every REAL-environment development run (``env_factory`` None, no
    registration) is accounted in the persistent per-arm ledger
    (``harness/devledger.py``) before it starts: kind ``dev_train`` counts
    against the 3-per-arm allowance; ``frozen_policy=True`` (learning rate 0:
    the policy stays at its random initialisation, for throughput
    measurements) is ledgered as ``measurement`` and not counted. A
    ``dev_reservation`` made by the caller (smoke / throughput paths) is used
    instead of making a second entry; the caller then finishes it.

    ``options`` (amendment 2026-10-25, default: the frozen-tag preset) selects
    a registered ``TrainOptions`` preset; ``init_trunk`` is the pre-trained
    trunk checkpoint, required exactly when the preset asks for one.
    ``dev_kind`` is the ledger kind of a real-environment development run
    (``dev_train``, or ``gate_dev`` for the learnability gate).
    """
    options = PRESETS[FROZEN_PRESET] if options is None else options
    if PRESETS.get(options.preset) != options:
        raise ValueError("training options must be one of the named PRESETS")
    if options.pretrained_trunk != (init_trunk is not None):
        raise ValueError(f"preset {options.preset!r} "
                         + ("needs" if options.pretrained_trunk else "refuses")
                         + " an init_trunk checkpoint")
    if dev_kind not in ("dev_train", "gate_dev"):
        raise ValueError(f"unknown development kind {dev_kind!r}")
    trunk_sha = None if init_trunk is None else file_sha256(init_trunk)
    reg_host = guards.check_train_boundary(
        registration, arm=arm, seed=seed, condition=condition,
        total_pair_steps=total_pair_steps,
        settings={"micro_batch_pairs": micro_batch_pairs,
                  "clear_cache": clear_cache, "n_workers": n_workers},
        config={"n_pairs": n_pairs, "horizon": horizon,
                "log_every_pair_steps": log_every_pair_steps,
                "ckpt_every_pair_steps": ckpt_every_pair_steps,
                "minibatch_pairs": minibatch_pairs},
        has_env_factory=env_factory is not None,
        options_preset=options.preset, init_trunk_sha256=trunk_sha,
        pretrained_trunk=options.pretrained_trunk)
    own_res = None
    if registration is not None:
        if frozen_policy or dev_reservation is not None:
            raise guards.GuardError(
                "a registered run cannot use frozen_policy or a development "
                "reservation")
    elif env_factory is None:  # real Procgen development run: account it
        kind = "measurement" if frozen_policy else dev_kind
        if dev_reservation is None:
            own_res = devledger.reserve(arm, condition, total_pair_steps,
                                        kind=kind, path=dev_ledger, seed=seed)
            devledger.claim(own_res, arm, condition, total_pair_steps)
        else:
            if not isinstance(dev_reservation, devledger.Reservation):
                raise devledger.SmokeLimitError(
                    "not a genuine development-ledger reservation")
            ok_kind = (dev_reservation.kind == "measurement") if frozen_policy \
                else dev_reservation.kind in devledger.LEARNING_KINDS
            if not ok_kind:
                raise devledger.SmokeLimitError(
                    f"reservation kind {dev_reservation.kind!r} does not match "
                    "this run")
            # exactly once, atomically: claimed or completed ones are rejected
            devledger.claim(dev_reservation, arm, condition, total_pair_steps)
    if micro_batch_pairs is not None and micro_batch_pairs < 1:
        raise ValueError("micro_batch_pairs must be >= 1")
    if total_pair_steps % n_pairs or log_every_pair_steps % n_pairs:
        raise ValueError("total and log interval must be multiples of n_pairs")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ss = run_seed_sequence(seed, arm, condition)
    env_ss, init_ss, shuf_ss, key_ss = ss.spawn(4)
    shuf_rng = np.random.default_rng(shuf_ss)
    key_rng = np.random.default_rng(key_ss)

    outcome = "crashed"
    vec = None
    try:
        vec = VecCoopEnv(
            n_pairs, env_ss, team_summary=(condition == "registered"),
            env_factory=env_factory, n_workers=n_workers,
        )
        _assert_not_stub(vec)  # raises before any training
        rec = _train(
            vec, arm, seed, condition, total_pair_steps, out_dir, n_pairs,
            horizon, log_every_pair_steps, ckpt_every_pair_steps,
            minibatch_pairs, restarts, init_ss, shuf_rng, key_rng,
            _fault_hook, _rollout_hook, micro_batch_pairs, clear_cache,
            n_workers, registration, reg_host, frozen_policy, options,
            init_trunk,
        )
        outcome = ("liveness_abort" if rec.liveness_abort else
                   "failed_non_finite" if rec.training_failure else "completed")
        return rec
    finally:
        if vec is not None:
            vec.close()
        if own_res is not None:
            devledger.finish(own_res, outcome)


def _train(vec, arm, seed, condition, total, out_dir, n_pairs, horizon,
           log_every, ckpt_every, minibatch_pairs, restarts, init_ss,
           shuf_rng, key_rng, fault_hook, rollout_hook, micro_pairs=None,
           clear_cache=False, n_workers=1, registration=None,
           reg_host=None, frozen_policy=False, options=None,
           init_trunk=None) -> RunRecord:
    options = PRESETS[FROZEN_PRESET] if options is None else options
    rec = RunRecord(
        arm=arm, seed=seed, condition=condition,
        codes={"arm": ARM_CODES[arm], "condition": CONDITION_CODES[condition]},
        total_pair_steps=total, restarts=restarts, using_stub=False,
        micro_batch_pairs=micro_pairs, clear_cache=clear_cache,
        n_workers=n_workers, train_options=asdict(options),
    )
    if registration is not None:
        rec.registration = {k: registration.pins()[k] for k in (
            "freeze_tag", "freeze_commit", "swhid", "manifest_sha256")}
        rec.registration["host"] = reg_host
        rec.runtime = guards._runtime_probe()
    model = PairPolicy(arm, mx.random.key(init_key_int(seed, arm, condition)))
    mx.eval(model.parameters())
    if init_trunk is not None:
        rec.init_trunk_sha256 = file_sha256(init_trunk)  # re-read: must match
        if registration is not None and (
                rec.init_trunk_sha256
                != guards.REGISTERED_TRUNK_SHA256.get(seed)):
            raise guards.GuardError("pre-trained trunk changed after the guard")
        load_trunk(model, init_trunk)
    # frozen_policy (throughput measurements): learning rate 0 keeps the
    # policy at its random initialisation while the full update cost is paid
    opt = make_optimizer(options, frozen_policy)
    opt.init(model.trainable_parameters())
    mx.eval(opt.state)
    use_eng = arm == "eng"
    plastic = arm == "gtm"
    grad_step, apply_step, ratio_dev = _make_steps(
        model, opt, use_eng, plastic=plastic,
        mask_finished=options.mask_finished,
    )
    mask = options.mask_finished
    coin_r = np.float32(options.coin_reward)
    probe_rng = np.random.default_rng(seed)  # probe subsampling only

    bus = MessageBus(arm)
    nd = model.noise_dim
    counters = EngCounters(n_pairs)
    recent: deque[float] = deque(maxlen=SUCCESS_WINDOW)
    total_env_steps = total // n_pairs
    total_updates = max(1, -(-total_env_steps // horizon))
    zeros_eng = np.zeros((n_pairs, 2, 16), np.float32)

    mx.reset_peak_memory()  # peak_memory_bytes covers this run only
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
    fin_prev = np.zeros((n_pairs, 2), dtype=bool)  # finished BEFORE the action
    ep_coin = np.zeros((n_pairs, 2), dtype=bool)
    recent_coin: deque[float] = deque(maxlen=2 * SUCCESS_WINDOW)

    def probe(obs_b, summ_b, eng_b) -> bool:
        if options.liveness == "off":
            return False
        k = probe_rng.choice(obs_b.shape[0], min(LIVENESS_PROBE_PAIRS,
                                                 obs_b.shape[0]), replace=False)
        m = liveness_probe(model, obs_b[k], summ_b[k],
                           eng_b[k] if use_eng else None,
                           mx.random.key(int(probe_rng.integers(0, 2**31 - 1))))
        m["step"] = step
        rec.liveness.append(m)
        if m["flagged"]:
            rec.liveness_flagged = True
        return m["flagged"] and options.liveness == "abort"

    probe(obs, summ, eng)  # step 0 (initialisation) reference
    update = 0
    while env_steps < total_env_steps and not rec.training_failure:
        update += 1
        t_len = min(horizon, total_env_steps - env_steps)
        buf = RolloutBuffer(n_pairs, t_len, noise_dim=nd)
        ent_sum, ent_n, ent_min = 0.0, 0, float("inf")
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
            ent = -(mx.exp(logp_all) * logp_all).sum(-1)
            to_eval = [act, logp, out.value, out.hard_bits, ent]
            if noise is not None:
                to_eval.append(noise)
            mx.eval(*to_eval)
            actions = np.array(act)
            live = ~fin_prev
            ent_np = np.array(ent)[live]
            if ent_np.size:
                ent_sum += float(ent_np.sum())
                ent_n += int(ent_np.size)
                ent_min = min(ent_min, float(ent_np.min()))
            bus.deliver(np.array(out.hard_bits), phase="train")  # enforce + log
            res = vec.step(actions.astype(np.int32))
            coin = (np.zeros((n_pairs, 2), bool) if res.coin is None
                    else np.asarray(res.coin, bool))
            buf.add(obs, summ, eng, actions, np.array(logp), np.array(out.value),
                    res.team_reward, res.done,
                    noise=None if noise is None else np.array(noise),
                    agent_reward=coin_r * coin.astype(np.float32), live=live)
            ep_coin |= coin
            for k in np.flatnonzero(res.done):
                recent.append(float(res.team_reward[k]))
                recent_coin.extend(float(c) for c in ep_coin[k])
                rec.completed_episodes += 1
            ep_coin[res.done] = False
            fin_prev = np.asarray(res.finished, bool) & ~res.done[:, None]
            counters.update(actions, res.done, res.finished)
            obs, summ, win = res.obs, res.summary, res.window
            eng = eng_now()
            env_steps += 1
            step += n_pairs
            if step % log_every == 0:  # exact boundary
                rate = float(np.mean(recent)) if recent else None
                rec.success_curve.append([step, rate])
                rec.success_curve_n.append(len(recent))
                rec.coin_rate_curve.append(
                    [step, float(np.mean(recent_coin)) if recent_coin else None])
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
        if ent_n:
            rec.entropy_curve.append([step, ent_sum / ent_n, ent_min])
        if rollout_hook is not None:
            rollout_hook(update, buf)

        scale = mx.array(plasticity_scale(update, total_updates), dtype=mx.float32)
        losses: list[float] = []
        first_mb = True
        for _ in range(EPOCHS):
            for mb in buf.minibatches(shuf_rng, minibatch_pairs):
                if first_mb:  # unchanged policy: ratio must be 1
                    n_mb = mb["actions"].shape[0]
                    devs = []
                    k_dev = _key(key_rng)
                    for a, b, _ in micro_slices(n_mb, micro_pairs):
                        d = ratio_dev(
                            mb["obs"][a:b], mb["summary"][a:b],
                            mb["eng_bits"][a:b], mb["actions"][a:b],
                            mb["noise"][a:b], mb["logp"][a:b], k_dev,
                        )
                        devs.append(float(d.item()))
                    rec.first_mb_ratio_dev.append(max(devs))
                    first_mb = False
                lv, acc, finite = accumulate_minibatch(
                    grad_step, mb, micro_pairs, _key(key_rng), mask=mask
                )
                if not finite:
                    rec.training_failure = True
                    break  # detected before applying: no optimizer step
                grads, norm = finalize_grads(acc, scale, plastic)
                mx.eval(norm)
                if not np.isfinite(float(norm.item())):
                    rec.training_failure = True
                    break
                losses.append(lv)
                apply_step(grads)
                mx.eval(model.state, opt.state)
                rec.optimizer_steps += 1
                if not _params_finite(model):
                    rec.training_failure = True
                    break
            if rec.training_failure:
                break
        if clear_cache:
            mx.clear_cache()
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
        if step // options.liveness_every > prev // options.liveness_every:
            t_obs = buf.obs.reshape(-1, 2, 64, 64, 3)
            if probe(t_obs, buf.summary.reshape(-1, 2, 2),
                     buf.eng_bits.reshape(-1, 2, 16)):
                rec.liveness_abort = True
                rec.training_failure = True
                rec.failure_reason = (
                    f"trunk liveness guard at update {update} (pair-step "
                    f"{step}): {rec.liveness[-1]['reasons']}")
                rec.last_checkpoint = _save_ckpt(model, out_dir, step)
                break
        if fault_hook is not None:
            fault_hook(update, model)

    if not rec.training_failure and rec.last_checkpoint != f"ckpt_{step}.safetensors":
        rec.last_checkpoint = _save_ckpt(model, out_dir, step)  # exact end

    rec.wall_time_s = time.time() - t0
    if rec.wall_time_s > 0:
        rec.pair_steps_per_s = rec.pair_steps_done / rec.wall_time_s
        rec.samples_per_s = 2 * rec.pair_steps_per_s
    rec.bits_logged = bus.bits_logged
    rec.peak_memory_bytes = int(mx.get_peak_memory())
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
    p.add_argument("--micro-batch-pairs", type=int, default=None,
                   help="gradient-accumulation micro-batch (pair-steps); "
                        "implementation setting, default off")
    p.add_argument("--clear-cache", action="store_true",
                   help="mx.clear_cache() once per rollout update (after its "
                        "optimizer steps), not per minibatch")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--host", choices=list(HOSTS),
                   help="registered-budget runs: the host whose registered "
                        "settings (codes.HOST_SETTINGS) apply; must be the "
                        "seed's fixed host")
    p.add_argument("--preset", default=None, choices=sorted(PRESETS),
                   help="training-options preset (amendment 2026-10-25); "
                        "default: the registered preset "
                        "(campaign.guards.REGISTERED_TRAIN_OPTIONS)")
    p.add_argument("--init-trunk", type=Path, default=None,
                   help="pre-trained trunk checkpoint (harness.pretrain), "
                        "required by presets with pretrained_trunk")
    p.add_argument("--dev-kind", default="dev_train",
                   choices=["dev_train", "gate_dev"],
                   help="development-ledger kind of a development run")
    guards.add_guard_args(p)
    a = p.parse_args()
    preset = a.preset or guards.REGISTERED_TRAIN_OPTIONS
    registered = a.steps > guards.DEV_MAX_PAIR_STEPS
    if registered and a.steps != guards.REGISTERED_BUDGET:
        sys.exit(f"training refused: {a.steps:,} pair-steps is neither the "
                 f"development mode (<= {guards.DEV_MAX_PAIR_STEPS:,}) nor the "
                 f"exact registered budget ({guards.REGISTERED_BUDGET:,})")
    settings = {"micro_batch_pairs": a.micro_batch_pairs,
                "clear_cache": a.clear_cache, "n_workers": a.n_workers or 1}
    ctx = None
    if registered:
        given = [f for f, v in (("--micro-batch-pairs", a.micro_batch_pairs),
                                ("--clear-cache", a.clear_cache or None),
                                ("--n-workers", a.n_workers)) if v is not None]
        if given:
            sys.exit("registered-budget run refuses setting overrides "
                     f"{given}: settings come from codes.HOST_SETTINGS")
        if not a.host:
            sys.exit("registered-budget training refused: --host is needed to "
                     "select the registered per-host settings")
        settings = dict(HOST_SETTINGS[a.host])
        errs = guards.registered_run_problems(
            arm=a.arm, seed=a.seed, condition=a.condition,
            total_pair_steps=a.steps, settings=settings,
            config={"n_pairs": a.n_pairs, "horizon": a.horizon,
                    "log_every_pair_steps": a.log_every,
                    "ckpt_every_pair_steps": a.ckpt_every,
                    "minibatch_pairs": MINIBATCH_PAIRS})
        if a.seed in SEED_HOST and SEED_HOST[a.seed] != a.host:
            errs.append(f"--host {a.host} is not seed {a.seed}'s fixed host "
                        f"{SEED_HOST[a.seed]}")
        if errs:
            sys.exit("registered-budget training refused: " + "; ".join(errs))
        ctx = guards.enforce_cli(a, registered=True,
                                 what="registered-budget training")
    rec = train_run(
        a.arm, a.seed, a.condition, a.steps, a.out, n_pairs=a.n_pairs,
        horizon=a.horizon, log_every_pair_steps=a.log_every,
        ckpt_every_pair_steps=a.ckpt_every, registration=ctx,
        options=options_for(preset), init_trunk=a.init_trunk,
        dev_kind=a.dev_kind, **settings,
    )
    print(json.dumps({k: v for k, v in asdict(rec).items()
                      if k not in ("losses", "liveness", "entropy_curve")}))


if __name__ == "__main__":
    main()
