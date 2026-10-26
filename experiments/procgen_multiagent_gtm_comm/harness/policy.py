"""Pair policy network for the N10-C MLX PPO harness.

Per agent: IMPALA (16, 32, 32) -> 256-d h; Linear(256 -> 64) -> comm.send;
partner swap (agent i receives the partner's msg_for_receiver);
comm.receive(msg, h) -> 64; concat [h, recv, summary] (256 + 64 + 2 = 322)
-> Linear(322 -> 256) -> ReLU -> policy (15) / value (1). Parameters are
shared across both agents. Inputs: own image, received message, team
summary only -- never rewards, done flags or partner actions.
No torch / track_p / nerve_wml import.
"""
from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .comm import HID_DIM, MSG_DIM, N_BITS, make_comm

N_ACTIONS = 15
SUMMARY_DIM = 2


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
    """IMPALA trunk: [N, 64, 64, 3] in [0, 1] -> 256-d hidden h."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = [Block(3, 16), Block(16, 32), Block(32, 32)]
        self.fc = nn.Linear(8 * 8 * 32, HID_DIM)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return nn.relu(self.fc(nn.relu(x).reshape(x.shape[0], -1)))


class PolicyOut(NamedTuple):
    logits: mx.array  # [B, 2, 15]
    value: mx.array  # [B, 2]
    hard_bits: mx.array  # [B, 2, 16]
    # auxiliary sender loss (N10-C-bis ``vq`` arm, training only); None for
    # every other arm and at evaluation, so the N10-C loss is unchanged
    aux_loss: mx.array | None = None
    # the same loss per sender row [B, 2] (amendment 2026-10-25: masked for
    # finished agents under mask_finished, as the policy and entropy terms)
    aux_rows: mx.array | None = None


class Encoded(NamedTuple):
    """Trunk + sender output of ``PairPolicy.encode`` (row = 2 * pair + agent)."""

    h: mx.array  # [2B, 256]
    msg: mx.array  # [2B, 16], or the GTM ST one-hot (64) / VQ ST rows (32)
    hard_bits: mx.array  # [B, 2, 16]
    x_msg: mx.array  # [2B, 64] sender input (``to_msg(h)``)


class PairPolicy(nn.Module):
    def __init__(self, arm: str, key: mx.array) -> None:
        super().__init__()
        self.arm = arm
        self.comm = make_comm(arm, key)  # also seeds the global init RNG
        self.trunk = Impala()
        self.to_msg = nn.Linear(HID_DIM, MSG_DIM)
        self.fuse = nn.Linear(HID_DIM + MSG_DIM + SUMMARY_DIM, 256)
        self.pi = nn.Linear(256, N_ACTIONS)
        self.v = nn.Linear(256, 1)

    @property
    def noise_dim(self) -> int:
        return self.comm.noise_dim

    def sample_noise(self, b: int, key: mx.array) -> mx.array | None:
        """Base message noise for ``b`` pairs, [b, 2, noise_dim] (or None).

        Pass it back as ``noise=`` to replay the same stochastic message.
        """
        nz = self.comm.sample_noise(b * 2, key)
        return None if nz is None else nz.reshape(b, 2, -1)

    def encode(
        self,
        obs: mx.array,
        eng_bits: mx.array | None,
        *,
        train: bool,
        key: mx.array,
        noise: mx.array | None = None,
    ) -> Encoded:
        """Trunk and sender for both agents (the first half of ``__call__``).

        The ``gtm_mis`` rollout calls it alone, delivers the hard bits
        deranged across pairs through the bus, then calls ``heads`` with
        what each agent received: one trunk pass, as for every other arm.
        """
        b = obs.shape[0]
        x = obs.reshape(b * 2, *obs.shape[2:]) / 255.0  # row = 2 * pair + agent
        h = self.trunk(x)
        eb = None if eng_bits is None else eng_bits.reshape(b * 2, N_BITS)
        nz = None if noise is None else noise.reshape(b * 2, -1)
        x_msg = self.to_msg(h)
        msg, hard = self.comm.send(x_msg, train=train, key=key, eng_bits=eb, noise=nz)
        return Encoded(h, msg, hard.reshape(b, 2, N_BITS), x_msg)

    def heads(self, enc: Encoded, summary: mx.array, msg_in: mx.array, *,
              train: bool) -> PolicyOut:
        """Receive ``msg_in`` ([B, 2, ·], row i = what agent i receives) and
        apply fuse, policy and value heads (the second half of ``__call__``)."""
        b = enc.hard_bits.shape[0]
        msg_in = msg_in.astype(enc.msg.dtype).reshape(b * 2, -1)
        recv = self.comm.receive(msg_in, enc.h)
        z = mx.concatenate([enc.h, recv, summary.reshape(b * 2, SUMMARY_DIM)], axis=-1)
        z = nn.relu(self.fuse(z))
        aux = aux_rows = None
        if train and hasattr(self.comm, "aux_loss"):
            aux = self.comm.aux_loss(enc.x_msg)
            aux_rows = self.comm.aux_loss_rows(enc.x_msg).reshape(b, 2)
        return PolicyOut(
            self.pi(z).reshape(b, 2, N_ACTIONS),
            self.v(z).reshape(b, 2),
            enc.hard_bits,
            aux,
            aux_rows,
        )

    def __call__(
        self,
        obs: mx.array,
        summary: mx.array,
        eng_bits: mx.array | None,
        *,
        train: bool,
        key: mx.array,
        noise: mx.array | None = None,
        recv_bits: mx.array | None = None,
    ) -> PolicyOut:
        """``recv_bits`` [B, 2, 16] replaces the message each agent receives
        (row i = what agent i receives): the evaluation probes, and the
        ``gtm_mis`` PPO update replaying the deranged delivery stored in the
        rollout; default None keeps the partner swap, so the N10-C arms
        are unchanged."""
        enc = self.encode(obs, eng_bits, train=train, key=key, noise=noise)
        if recv_bits is None:
            b = obs.shape[0]
            msg_p = enc.msg.reshape(b, 2, -1)  # 16 bits, or the GTM ST one-hot (64)
            msg_in = mx.stack([msg_p[:, 1], msg_p[:, 0]], axis=1)
        else:
            msg_in = recv_bits
        return self.heads(enc, summary, msg_in, train=train)


def param_count(module: nn.Module) -> int:
    return int(sum(p.size for _, p in tree_flatten(module.parameters())))
