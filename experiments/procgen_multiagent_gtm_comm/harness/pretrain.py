"""Single-agent CoinRun trunk pre-training (amendment 2026-10-25, option 2).

Trains the N10-C IMPALA trunk (``policy.Impala``) with a temporary
single-agent policy / value head by PPO on single-agent CoinRun, training
levels 0-15 only (``num_levels=16``, ``start_level=0``, ``easy``), with the
coin as the only reward (+1 when reached; procgen's 10 is mapped to 1) and
the same PPO hyper-parameters as the N10-C trainer (Adam eps 1e-5). Only
the trunk is kept: ``trunk.safetensors`` initialises ``PairPolicy.trunk``
of a coop run (``train.load_trunk``); the temporary heads are discarded.

All randomness derives from ``SeedSequence([seed, PRETRAIN_CODE])``: the
trunk depends on the seed only, not on the arm or condition, so every arm of
one seed starts from the same trunk.

The trunk-liveness probe (``train.trunk_activity``) runs at step 0 and after
the first update ending at or after every ``liveness_every`` steps; the run
aborts at the first flagged probe (a dead trunk is never written as a
usable checkpoint: ``pretrain.json`` then has ``usable: false``).

Per-host implementation settings (``codes.HOST_SETTINGS``) apply as in the
coop trainer: on KXKM-AI the PPO minibatch of 2048 agent-samples is split
into gradient-accumulated micro-batches of ``2 x micro_batch_pairs`` samples
and ``mx.clear_cache()`` runs once per rollout update (the full-minibatch
update fails on MLX CUDA with ``cudaGraphAddKernelNode ... invalid
argument``, 2026-10-25). The accumulated gradient equals the full-minibatch
gradient; advantages are normalised over the full minibatch.

Development runs (no registration context) are accounted in the
development ledger (kind ``pretrain_dev``) and are limited to
``devledger.MAX_STEPS`` steps. A registered pre-training run (context issued
by ``campaign.guards``) must be a registered seed, on that seed's fixed host,
with exactly ``guards.PRETRAIN_BUDGET`` steps; it is not ledgered and its
record carries the freeze pins.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from ..campaign import guards
from . import devledger
from .codes import ARM_CODES, CONDITION_CODES, HOST_SETTINGS
from .policy import N_ACTIONS, Impala
from .rollout import compute_gae
from .train import (
    CLIP,
    ENTROPY_COEF,
    EPOCHS,
    GAMMA,
    LAMBDA,
    LIVENESS_MIN_FRAC,
    LR,
    MAX_GRAD_NORM,
    VALUE_COEF,
    file_sha256,
    trunk_activity,
)

PRETRAIN_CODE = 99  # stream code of the pre-training run (not an arm code)
# Registered pre-training procedure (amendment 2026-10-25, section 3.2;
# guards.PRETRAIN_PROCEDURE): enforced for registered and pre-freeze runs.
REGISTERED_PROCEDURE = guards.PRETRAIN_PROCEDURE


def pretrain_seed_entropy(seed: int,
                          unit: tuple[str, str] | None = None) -> list[int]:
    """SeedSequence entropy of a pre-training run: ``[seed, 99]`` for the
    seed's shared trunk, ``[seed, 99, arm_code, condition_code]`` for the
    own trunk of one run of the per-run-pretraining control."""
    if unit is None:
        return [seed, PRETRAIN_CODE]
    arm, cond = unit
    if arm not in ARM_CODES or cond not in CONDITION_CODES:
        raise ValueError(f"unknown arm/condition {arm!r}/{cond!r}")
    return [seed, PRETRAIN_CODE, ARM_CODES[arm], CONDITION_CODES[cond]]
ADAM_EPS = 1e-5
MINIBATCH = 2048  # agent-samples, as in the coop trainer (1024 pairs x 2)


class SinglePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = Impala()
        self.pi = nn.Linear(256, N_ACTIONS)
        self.v = nn.Linear(256, 1)

    def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        h = self.trunk(obs.astype(mx.float32) / 255.0)
        return self.pi(h), self.v(h)[:, 0]


@dataclass
class PretrainRecord:
    seed: int
    total_steps: int
    n_envs: int
    horizon: int
    steps_done: int = 0
    n_updates: int = 0
    wall_time_s: float = 0.0
    episodes: int = 0
    coin_rate_curve: list[list[float | None]] = field(default_factory=list)
    liveness: list[dict] = field(default_factory=list)
    usable: bool = False
    failure_reason: str | None = None
    trunk_file: str | None = None
    trunk_sha256: str | None = None
    registration: dict | None = None
    settings: dict = field(default_factory=dict)
    levels: str = "training (start_level=0, num_levels=16, easy)"
    # per-run-pretraining control: the (arm, condition) whose own trunk
    # this is, and the entropy of its SeedSequence (None = shared trunk)
    unit: list[str] | None = None
    seed_sequence: list[int] = field(default_factory=list)
    # pre-freeze registered pre-training (freeze plan, option A): commit,
    # host, preset and budget of the procedure (None otherwise)
    pre_freeze: dict | None = None
    liveness_every: int | None = None


def make_env(n_envs: int, rand_seed: int) -> Any:
    from procgen import ProcgenGym3Env

    return ProcgenGym3Env(num=n_envs, env_name="coinrun", rand_seed=rand_seed,
                          num_levels=16, start_level=0,
                          distribution_mode="easy")


def _obs(o: Any) -> np.ndarray:
    return np.asarray(o["rgb"] if isinstance(o, dict) else o, dtype=np.uint8)


def pretrain_trunk(seed: int, total_steps: int, out_dir: Path, *,
                   n_envs: int = 64, horizon: int = 128,
                   liveness_every: int = 100_000,
                   env: Any | None = None,
                   dev_ledger: Path = devledger.LEDGER,
                   registration: Any = None,
                   micro_batch_pairs: int | None = None,
                   clear_cache: bool = False,
                   unit: tuple[str, str] | None = None,
                   pre_freeze: bool = False,
                   repo: Path = guards.REPO) -> PretrainRecord:
    """Pre-train the trunk for ``total_steps`` single-agent steps.

    ``env`` (tests only) replaces procgen: a gym3-style vector env with
    ``act`` / ``observe`` / ``close`` over ``n_envs`` copies.
    ``micro_batch_pairs`` / ``clear_cache``: per-host implementation
    settings; a registered run always uses its host's ``HOST_SETTINGS``.

    ``unit`` = (arm, condition): pre-train the OWN trunk of that run of the
    per-run-pretraining control, seeded ``SeedSequence([seed, PRETRAIN_CODE,
    arm_code, condition_code])`` (``pretrain_seed_entropy``); None = the
    seed's shared trunk, ``SeedSequence([seed, PRETRAIN_CODE])``.

    ``pre_freeze``: the registered procedure BEFORE the new freeze tag
    (PI decision 2026-10-28): ``guards.check_prefreeze_pretrain`` (clean,
    pushed commit; registered seed or control unit; exact budget; the
    seed's host and its settings), ledger kind ``pretrain_registered``
    (one usable trunk per seed / unit, one restart of an unusable one)."""
    pf = None
    if pre_freeze or registration is not None:
        got = {"n_envs": n_envs, "horizon": horizon,
               "liveness_every": liveness_every}
        if got != REGISTERED_PROCEDURE:
            raise ValueError(f"registered pre-training uses {REGISTERED_PROCEDURE}"
                             f", not {got}")
    if pre_freeze:
        if registration is not None:
            raise ValueError("pre_freeze and a registration are exclusive")
        pf = guards.check_prefreeze_pretrain(
            seed=seed, total_steps=total_steps, has_env=env is not None,
            unit=unit, repo=repo)
        host = pf["host"]
    else:
        host = guards.check_pretrain_boundary(registration, seed=seed,
                                              total_steps=total_steps,
                                              has_env=env is not None,
                                              unit=unit)
    if host is not None:
        micro_batch_pairs = HOST_SETTINGS[host]["micro_batch_pairs"]
        clear_cache = HOST_SETTINGS[host]["clear_cache"]
    if micro_batch_pairs is not None and micro_batch_pairs < 1:
        raise ValueError("micro_batch_pairs must be >= 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entropy = pretrain_seed_entropy(seed, unit)
    ss = np.random.SeedSequence(entropy)
    env_ss, init_ss, shuf_ss, key_ss, probe_ss = ss.spawn(5)
    shuf_rng = np.random.default_rng(shuf_ss)
    key_rng = np.random.default_rng(key_ss)
    probe_rng = np.random.default_rng(probe_ss)  # probes never touch shuf_rng
    res = None
    if env is None and registration is None:
        if total_steps > devledger.MAX_STEPS:
            raise devledger.SmokeLimitError(
                f"pre-training is limited to {devledger.MAX_STEPS:,} steps")
        res = devledger.reserve(
            devledger.PRETRAIN_ARM, devledger.PRETRAIN_CONDITION, total_steps,
            kind=devledger.PREFREEZE_KIND if pre_freeze else "pretrain_dev",
            path=dev_ledger, seed=seed, unit=unit if pre_freeze else None)
        devledger.claim(res, devledger.PRETRAIN_ARM,
                        devledger.PRETRAIN_CONDITION, total_steps)
    rec = PretrainRecord(seed, total_steps, n_envs, horizon)
    rec.seed_sequence = entropy
    rec.unit = None if unit is None else list(unit)
    rec.pre_freeze = pf
    rec.liveness_every = liveness_every
    rec.settings = {"micro_batch_pairs": micro_batch_pairs,
                    "clear_cache": clear_cache}
    if registration is not None:
        rec.registration = {k: registration.pins()[k] for k in (
            "freeze_tag", "freeze_commit", "swhid", "manifest_sha256")}
        rec.registration["host"] = host
    outcome = "crashed"
    try:
        try:
            if env is None:  # inside try: a failed env build is ledgered
                env = make_env(n_envs,
                               int(env_ss.generate_state(1)[0] % (2**31 - 1)))
            _pretrain(rec, env, out_dir, init_ss, shuf_rng, key_rng, probe_rng,
                      liveness_every, micro_batch_pairs, clear_cache)
            outcome = "completed" if rec.usable else "liveness_abort"
        finally:
            if env is not None:
                env.close()
    finally:  # ledger and record even if the env build or close fails
        try:
            (out_dir / "pretrain.json").write_text(
                json.dumps(asdict(rec), indent=2))
        finally:
            if res is not None:
                devledger.finish(res, outcome)
    return rec


def micro_slices(n: int, micro: int | None) -> list[tuple[int, int, float]]:
    """(start, stop, weight=size/n) micro-batches of a minibatch of n samples."""
    if micro is None or micro >= n:
        return [(0, n, 1.0)]
    return [(a, min(a + micro, n), (min(a + micro, n) - a) / n)
            for a in range(0, n, micro)]


def _pretrain(rec, env, out_dir, init_ss, shuf_rng, key_rng, probe_rng,
              liveness_every, micro_batch_pairs=None, clear_cache=False) -> None:
    micro = None if micro_batch_pairs is None else 2 * micro_batch_pairs
    mx.random.seed(int(init_ss.generate_state(1)[0] % (2**31 - 1)))
    model = SinglePolicy()
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=LR, eps=ADAM_EPS)
    opt.init(model.trainable_parameters())

    def loss_fn(model, obs, act, old_logp, adv, ret):
        logits, value = model(obs)
        lp_all = nn.log_softmax(logits, axis=-1)
        lp = mx.take_along_axis(lp_all, act[:, None], axis=-1)[:, 0]
        ratio = mx.exp(lp - old_logp)
        pg = -mx.minimum(ratio * adv,
                         mx.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
        vf = ((value - ret) ** 2).mean()
        ent = -(mx.exp(lp_all) * lp_all).sum(-1).mean()
        return pg + VALUE_COEF * vf - ENTROPY_COEF * ent

    grad_fn = nn.value_and_grad(model, loss_fn)
    n, t_h = rec.n_envs, rec.horizon
    _, o, _ = env.observe()
    obs = _obs(o)
    recent: deque[float] = deque(maxlen=100)
    coin_now = np.zeros(n, bool)

    def probe(frames: np.ndarray) -> bool:
        k = probe_rng.choice(frames.shape[0], min(256, frames.shape[0]),
                             replace=False)
        m = trunk_activity(model.trunk,
                           mx.array(frames[k]).astype(mx.float32) / 255.0)
        m["step"] = rec.steps_done
        m["flagged"] = (m["block3_pos_frac"] < LIVENESS_MIN_FRAC
                        or m["live_h_frac"] < LIVENESS_MIN_FRAC)
        rec.liveness.append(m)
        return m["flagged"]

    t0 = time.time()
    if probe(obs):
        rec.failure_reason = "dead trunk at initialisation"
        return
    while rec.steps_done < rec.total_steps:
        t_len = min(t_h, (rec.total_steps - rec.steps_done) // n)
        if t_len < 1:
            break
        b_obs = np.zeros((t_len, n, 64, 64, 3), np.uint8)
        b_act = np.zeros((t_len, n), np.int64)
        b_lp = np.zeros((t_len, n), np.float32)
        b_v = np.zeros((t_len, n, 1), np.float32)
        b_r = np.zeros((t_len, n), np.float32)
        b_d = np.zeros((t_len, n), bool)
        for t in range(t_len):
            logits, value = model(mx.array(obs))
            k = mx.random.key(int(key_rng.integers(0, 2**31 - 1)))
            a = mx.random.categorical(logits, key=k)
            lp = mx.take_along_axis(nn.log_softmax(logits, axis=-1),
                                    a[:, None], axis=-1)[:, 0]
            mx.eval(a, lp, value)
            act = np.array(a)
            env.act(act.astype(np.int32))
            r, o, first = env.observe()
            coin = np.asarray(r) > 0.5
            done = np.asarray(first, bool)
            b_obs[t], b_act[t], b_lp[t] = obs, act, np.array(lp)
            b_v[t, :, 0] = np.array(value)
            b_r[t], b_d[t] = coin.astype(np.float32), done
            coin_now |= coin
            for i in np.flatnonzero(done):
                recent.append(float(coin_now[i]))
                rec.episodes += 1
            coin_now[done] = False
            obs = _obs(o)
        prev = rec.steps_done
        rec.steps_done += t_len * n
        last_v = np.array(model(mx.array(obs))[1])[:, None]
        adv, ret = compute_gae(b_r, b_v, b_d, last_v, GAMMA, LAMBDA)
        flat_obs = b_obs.reshape(-1, 64, 64, 3)
        flat = (b_act.reshape(-1), b_lp.reshape(-1), adv.reshape(-1),
                ret.reshape(-1))
        for _ in range(EPOCHS):
            perm = shuf_rng.permutation(flat_obs.shape[0])
            for s in range(0, perm.size, MINIBATCH):
                idx = perm[s:s + MINIBATCH]
                ad = mx.array(flat[2][idx])  # normalised over the FULL minibatch
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                g = None
                for a, b, w in micro_slices(idx.size, micro):
                    j = idx[a:b]
                    _, gm = grad_fn(model, mx.array(flat_obs[j]),
                                    mx.array(flat[0][j]), mx.array(flat[1][j]),
                                    ad[a:b], mx.array(flat[3][j]))
                    if w != 1.0:
                        gm = tree_map(lambda x, w=w: x * w, gm)
                    g = gm if g is None else tree_map(mx.add, g, gm)
                    mx.eval(g)
                g, _ = optim.clip_grad_norm(g, MAX_GRAD_NORM)
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
        rec.n_updates += 1
        if clear_cache:
            mx.clear_cache()
        if rec.steps_done // liveness_every > prev // liveness_every:
            rec.coin_rate_curve.append(
                [rec.steps_done, float(np.mean(recent)) if recent else None])
            if probe(flat_obs):
                rec.failure_reason = (f"trunk liveness guard at step "
                                      f"{rec.steps_done}")
                rec.wall_time_s = time.time() - t0
                return
    rec.wall_time_s = time.time() - t0
    if not all(bool(mx.isfinite(p).all().item())
               for _, p in tree_flatten(model.trunk.parameters())):
        rec.failure_reason = "non-finite trunk parameters"
        return
    path = out_dir / "trunk.safetensors"
    mx.save_safetensors(str(path), dict(tree_flatten(model.trunk.parameters())))
    rec.trunk_file = path.name
    rec.trunk_sha256 = file_sha256(path)
    rec.usable = True


def main() -> None:
    p = argparse.ArgumentParser(description="N10-C single-agent trunk "
                                "pre-training (training levels 0-15)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--registered", action="store_true",
                   help="registered pre-training of a registered seed "
                        "(requires the freeze-tag guards)")
    p.add_argument("--micro-batch-pairs", type=int, default=None,
                   help="development only: gradient-accumulation micro-batch "
                        "(pairs; 2 agent-samples each); registered runs use "
                        "their host's HOST_SETTINGS")
    p.add_argument("--clear-cache", action="store_true",
                   help="development only: mx.clear_cache() per update")
    p.add_argument("--pre-freeze", action="store_true",
                   help="the registered pre-training of a registered seed (or "
                        "control unit) BEFORE the new freeze tag (PI decision "
                        "2026-10-28): clean pushed commit, exact budget, the "
                        "seed's host and settings; ledger kind "
                        "pretrain_registered")
    p.add_argument("--per-run", nargs=2, metavar=("ARM", "CONDITION"),
                   default=None,
                   help="own trunk of one run of the per-run-pretraining "
                        "control (seed 101): SeedSequence([seed, 99, "
                        "arm_code, condition_code])")
    guards.add_guard_args(p)
    a = p.parse_args()
    ctx = guards.enforce_cli(a, registered=a.registered,
                             what="registered pre-training")
    if a.registered and a.pre_freeze:
        p.error("--registered and --pre-freeze are exclusive")
    if (a.registered or a.pre_freeze) and (
            a.micro_batch_pairs is not None or a.clear_cache):
        p.error("registered pre-training uses its host's HOST_SETTINGS; "
                "do not pass --micro-batch-pairs / --clear-cache")
    if a.pre_freeze and a.steps != guards.PRETRAIN_BUDGET:
        p.error(f"--pre-freeze needs --steps {guards.PRETRAIN_BUDGET}")
    rec = pretrain_trunk(a.seed, a.steps, a.out, registration=ctx,
                         micro_batch_pairs=a.micro_batch_pairs,
                         clear_cache=a.clear_cache,
                         unit=None if a.per_run is None else tuple(a.per_run),
                         pre_freeze=a.pre_freeze)
    print(json.dumps({k: v for k, v in asdict(rec).items()
                      if k not in ("liveness",)}))


if __name__ == "__main__":
    main()
