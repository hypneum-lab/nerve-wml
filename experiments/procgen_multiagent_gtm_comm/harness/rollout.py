# harness — pair-step rollout buffer and GAE for the N10-C PPO loop.
#
# Sample unit = pair-step: one entry holds BOTH agents' data at one env
# step of one coop pair. The team reward is credited to both agents.
# Storage stays in numpy (obs uint8); mx conversion happens per
# minibatch; /255 scaling is the policy's job.
#
# Advantages/returns are stored by ``finish(last_value, ...)`` (which calls
# ``compute_gae``); ``minibatches`` raises until that was done.
#
# Amendment 2026-10-25 (training options, see train.TrainOptions):
# ``agent_reward`` [T, n, 2] is an optional per-agent reward added to the
# team reward (individual coin reward), and ``live`` [T, n, 2] marks the
# agents that were NOT yet finished when the action was taken (a finished
# agent is frozen and its action is ignored by the environment). Both
# default to the frozen behaviour (no extra reward, every agent live).
#
# N10-C-bis (addendum 2026-10-25, arm ``gtm_mis``): ``recv_bits`` stores
# the 16-bit message each agent actually received at each pair-step so the
# PPO update replays it (``PairPolicy.__call__(recv_bits=...)``);
# ``derangement`` / ``deranged_recv`` are the per-rollout derangement of
# the pairs and the delivery it induces (also used by the message bus).
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep MLX lazy (absent on Linux CI)
    import mlx.core as mx


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of range(n) without fixed points (rejection).

    Used by the evaluation shuffle probe (levels, ``evaluate.py``) and by
    the ``gtm_mis`` training delivery (coop pairs, one draw per rollout).
    """
    if n < 2:
        raise ValueError("a derangement needs at least 2 levels")
    ident = np.arange(n)
    while True:
        p = rng.permutation(n)
        if not (p == ident).any():
            return p


def deranged_recv(bits: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Deranged delivery: ``out[e, i] = bits[sigma[e], 1 - i]`` for hard
    messages ``bits`` [n, 2, 16] (agent i of pair e receives what role
    1 - i of pair sigma(e) emitted at the same step)."""
    bits = np.asarray(bits)
    out = np.empty_like(bits)
    out[:, 0] = bits[sigma, 1]
    out[:, 1] = bits[sigma, 0]
    return out


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float = 0.999,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over [T, n] team rewards (or [T, n, 2] per-agent rewards) and
    [T, n, 2] per-agent values.

    ``dones[t]`` means the episode ended at step t: no bootstrap from
    V(t+1) across it. ``last_value`` is V after the final step, [n, 2].
    Returns (advantages, returns), both [T, n, 2] float32.
    """
    rewards = np.asarray(rewards, np.float64)
    values = np.asarray(values, np.float64)
    nonterm = 1.0 - np.asarray(dones, np.float64)  # [T, n]
    t_len = rewards.shape[0]
    next_values = np.concatenate(
        [values[1:], np.asarray(last_value, np.float64)[None]], axis=0
    )
    nt = nonterm[..., None]
    r = rewards[..., None] if rewards.ndim == 2 else rewards
    delta = r + gamma * next_values * nt - values
    adv = np.zeros_like(values)
    gae = np.zeros_like(values[0])
    for t in reversed(range(t_len)):
        gae = delta[t] + gamma * lam * nt[t] * gae
        adv[t] = gae
    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)


class RolloutBuffer:
    def __init__(self, n_pairs: int, horizon: int, noise_dim: int = 0):
        self.n_pairs = n_pairs
        self.horizon = horizon
        t_len, n = horizon, n_pairs
        self.obs = np.zeros((t_len, n, 2, 64, 64, 3), np.uint8)
        self.summary = np.zeros((t_len, n, 2, 2), np.float32)
        self.eng_bits = np.zeros((t_len, n, 2, 16), np.float32)
        self.actions = np.zeros((t_len, n, 2), np.int64)
        self.logp = np.zeros((t_len, n, 2), np.float32)
        self.value = np.zeros((t_len, n, 2), np.float32)
        self.team_reward = np.zeros((t_len, n), np.float32)
        self.done = np.zeros((t_len, n), bool)
        self.agent_reward = np.zeros((t_len, n, 2), np.float32)
        self.live = np.ones((t_len, n, 2), bool)
        self.noise_dim = noise_dim
        self.noise = np.zeros((t_len, n, 2, noise_dim), np.float32)
        # message each agent received (gtm_mis replay; zeros for other arms)
        self.recv_bits = np.zeros((t_len, n, 2, 16), np.float32)
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None
        self.t = 0

    def add(self, obs, summary, eng_bits, actions, logp, value,
            team_reward, done, noise=None, agent_reward=None,
            live=None, recv_bits=None) -> None:
        if self.t >= self.horizon:
            raise RuntimeError("rollout buffer full")
        t = self.t
        self.obs[t] = obs
        self.summary[t] = summary
        self.eng_bits[t] = eng_bits
        self.actions[t] = actions
        self.logp[t] = logp
        self.value[t] = value
        self.team_reward[t] = team_reward
        self.done[t] = done
        if noise is not None:
            self.noise[t] = noise
        if agent_reward is not None:
            self.agent_reward[t] = agent_reward
        if live is not None:
            self.live[t] = live
        if recv_bits is not None:
            self.recv_bits[t] = recv_bits
        self.t += 1

    def finish(self, last_value, gamma: float = 0.999, lam: float = 0.95):
        if self.t != self.horizon:
            raise RuntimeError("rollout not full")
        rewards = self.team_reward[..., None] + self.agent_reward
        self.advantages, self.returns = compute_gae(
            rewards, self.value, self.done, last_value, gamma, lam
        )

    def reset(self) -> None:
        self.t = 0
        self.advantages = self.returns = None

    def minibatches(self, rng: np.random.Generator,
                    size_pairs: int = 1024) -> Iterator[dict[str, mx.array]]:
        """One pass: each flattened (T*n) pair-step exactly once, shuffled."""
        if self.advantages is None or self.t != self.horizon:
            raise RuntimeError("call finish() on a full buffer first")
        import mlx.core as mx  # noqa: F811  (lazy, after lifecycle check)

        n_total = self.horizon * self.n_pairs
        flat = {
            "obs": self.obs.reshape(n_total, 2, 64, 64, 3),
            "summary": self.summary.reshape(n_total, 2, 2),
            "eng_bits": self.eng_bits.reshape(n_total, 2, 16),
            "actions": self.actions.reshape(n_total, 2),
            "logp": self.logp.reshape(n_total, 2),
            "value": self.value.reshape(n_total, 2),
            "advantages": self.advantages.reshape(n_total, 2),
            "returns": self.returns.reshape(n_total, 2),
            "noise": self.noise.reshape(n_total, 2, self.noise_dim),
            "live": self.live.reshape(n_total, 2).astype(np.float32),
            "recv_bits": self.recv_bits.reshape(n_total, 2, 16),
        }
        perm = rng.permutation(n_total)
        for s in range(0, n_total, size_pairs):
            idx = perm[s:s + size_pairs]
            mb = {k: mx.array(v[idx]) for k, v in flat.items()}
            mb["index"] = mx.array(idx)
            yield mb
