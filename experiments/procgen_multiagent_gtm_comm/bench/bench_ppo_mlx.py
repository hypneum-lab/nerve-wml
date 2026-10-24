"""PPO (IMPALA CNN) throughput in MLX on Procgen-shaped data (N10-C).

Measures the neural-network cost of one PPO iteration — rollout
inference over 64 envs x 256 steps and a 3-epoch update with 2048-sample
minibatches — on synthetic 64x64x3 observations, then extrapolates to a
full N10-C run (2 agents x 50M steps). Procgen env stepping and the
GTM message path are not included.

Usage: python bench_ppo_mlx.py  (MLX 0.31.1, see requirements-mlx.lock)
"""
from __future__ import annotations

import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

N_ENV, T, MINIBATCH, EPOCHS = 64, 256, 2048, 3
STEPS_PER_RUN = 2 * 50_000_000  # 2 agents, parameter sharing


class Residual(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.c2(nn.relu(self.c1(nn.relu(x))))


class Block(nn.Module):
    def __init__(self, c_in: int, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.r1 = Residual(c)
        self.r2 = Residual(c)

    def __call__(self, x: mx.array) -> mx.array:
        return self.r2(self.r1(self.pool(self.conv(x))))


class Impala(nn.Module):
    def __init__(self, n_actions: int = 15) -> None:
        super().__init__()
        self.blocks = [Block(3, 16), Block(16, 32), Block(32, 32)]
        self.fc = nn.Linear(8 * 8 * 32, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        for block in self.blocks:
            x = block(x)
        h = nn.relu(self.fc(nn.relu(x).reshape(x.shape[0], -1)))
        return self.pi(h), self.v(h).squeeze(-1)


def ppo_loss(model, obs, act, adv, ret, old_logp):
    logits, value = model(obs)
    logp_all = nn.log_softmax(logits)
    logp = logp_all[mx.arange(act.shape[0]), act]
    ratio = mx.exp(logp - old_logp)
    pg = -mx.minimum(ratio * adv, mx.clip(ratio, 0.8, 1.2) * adv).mean()
    entropy = -(mx.softmax(logits) * logp_all).sum(-1).mean()
    return pg + 0.5 * ((value - ret) ** 2).mean() - 0.01 * entropy


def main() -> None:
    rng = np.random.default_rng(0)
    model = Impala()
    opt = optim.Adam(learning_rate=5e-4)
    mx.eval(model.parameters())
    grad_fn = nn.value_and_grad(model, ppo_loss)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(obs, act, adv, ret, old_logp):
        loss, grads = grad_fn(model, obs, act, adv, ret, old_logp)
        opt.update(model, grads)
        return loss

    obs64 = mx.array(rng.random((N_ENV, 64, 64, 3), dtype=np.float32))
    for _ in range(5):
        mx.eval(model(obs64))
    t0 = time.time()
    n_inf = 200
    for _ in range(n_inf):
        mx.eval(model(obs64))
    inference_s = (time.time() - t0) / n_inf

    batch = N_ENV * T
    data = {
        "obs": mx.array(rng.random((batch, 64, 64, 3), dtype=np.float32)),
        "act": mx.array(rng.integers(0, 15, batch)),
        "adv": mx.array(rng.standard_normal(batch, dtype=np.float32)),
        "ret": mx.array(rng.standard_normal(batch, dtype=np.float32)),
        "old_logp": mx.array(np.full(batch, -2.7, np.float32)),
    }
    keys = ("obs", "act", "adv", "ret", "old_logp")

    def update() -> None:
        for _ in range(EPOCHS):
            perm = rng.permutation(batch)
            for i in range(0, batch, MINIBATCH):
                idx = mx.array(perm[i : i + MINIBATCH])
                loss = train_step(*(data[k][idx] for k in keys))
                mx.eval(loss, state)

    update()  # compile + warm-up
    t0 = time.time()
    update()
    update_s = time.time() - t0

    rollout_s = T * inference_s
    per_iter = rollout_s + update_s
    samples_per_s = batch / per_iter
    print(f"inference 64 obs: {inference_s * 1e3:.2f} ms | "
          f"update ({EPOCHS} epochs x {batch}): {update_s:.2f} s")
    print(f"per iteration: rollout NN {rollout_s:.2f} s + update "
          f"{update_s:.2f} s = {per_iter:.2f} s")
    print(f"NN throughput: {samples_per_s:.0f} samples/s")
    print(f"estimate per run (NN only): "
          f"{STEPS_PER_RUN / samples_per_s / 3600:.1f} h")


if __name__ == "__main__":
    main()
