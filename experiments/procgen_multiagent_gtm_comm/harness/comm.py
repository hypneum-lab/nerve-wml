"""Message arms for the N10-C MLX harness (pre-registered message spec).

Every arm sends m in {0,1}^16 at evaluation. `send` returns
`(msg_for_receiver, hard_bits)`: msg_for_receiver carries the
straight-through gradient in training and equals hard_bits at evaluation
(DIAL: noisy sigmoid in training). No torch / track_p / nerve_wml import.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gtm_mlx import GammaThetaMultiplexerMLX, GTMConfigMLX

K_SYMBOLS = 4
ALPHABET = 16
N_BITS = 16
MSG_DIM = 64
HID_DIM = 256

# CODE_TABLE[a] = 4-bit binary of a, MSB first (ruling R2). [16, 4]
CODE_TABLE = mx.array(
    [[(a >> (3 - j)) & 1 for j in range(4)] for a in range(ALPHABET)], dtype=mx.float32
)
_BIT_WEIGHTS = mx.array([8, 4, 2, 1], dtype=mx.float32)


def bits_to_indices(bits: mx.array) -> mx.array:
    """bits [B, 16] -> symbol indices [B, 4] int32 (MSB-first nibbles)."""
    b = (bits > 0.5).astype(mx.float32).reshape(bits.shape[0], K_SYMBOLS, 4)
    return (b @ _BIT_WEIGHTS).astype(mx.int32)


def onehot_from_bits(bits: mx.array) -> mx.array:
    """Differentiable per-symbol one-hot from (ST) bits (ruling R2).

    onehot[a] = prod_j (b_j if CODE_TABLE[a, j] == 1 else 1 - b_j); exact for
    hard bits. bits [B, 16] -> [B, 4, 16].
    """
    b = bits.reshape(bits.shape[0], K_SYMBOLS, 1, 4)
    return mx.prod(b * CODE_TABLE + (1.0 - b) * (1.0 - CODE_TABLE), axis=-1)


def _adaptive_pool_matrix(n_in: int, n_out: int) -> mx.array:
    """Matrix equivalent to adaptive_avg_pool1d (floor/ceil windows)."""
    m = np.zeros((n_in, n_out), dtype=np.float32)
    for i in range(n_out):
        s, e = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
        m[s:e, i] = 1.0 / (e - s)
    return mx.array(m)


def plasticity_scale(update: int, total_updates: int, floor: float = 0.1) -> float:
    """Cosine schedule 1.0 -> floor over the run's total updates.

    The trainer multiplies this into the constellation gradient.
    """
    u = min(max(update / max(total_updates, 1), 0.0), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * u))


def eng_bits_from_window(in_window: bool, remaining: int, right_minus_left: int) -> np.ndarray:
    """Engineered 16-bit message: bit0 in_window; bits1-5 remaining (unsigned,
    MSB first); bits 6-15 clip(n, -512, 511) as 10-bit two's complement."""
    bits = np.zeros(16, dtype=np.float32)
    bits[0] = 1.0 if in_window else 0.0
    rem = int(remaining)
    if not 0 <= rem <= 31:
        raise ValueError(f"remaining out of 5-bit range: {rem}")
    for j in range(5):
        bits[1 + j] = (rem >> (4 - j)) & 1
    n = int(np.clip(right_minus_left, -512, 511)) & 0x3FF
    for j in range(10):
        bits[6 + j] = (n >> (9 - j)) & 1
    return bits


class Comm(nn.Module):
    """Base class. Subclasses implement `send` and `receive`."""

    # Width of the per-row base noise the sender consumes in training
    # (0 = deterministic sender). Noise can be sampled once, stored, and
    # replayed via ``send(..., noise=...)`` so PPO recomputes the same
    # stochastic message it acted on.
    noise_dim = 0

    def sample_noise(self, rows: int, key):
        return None

    def send(self, x, *, train, key, eng_bits, noise=None):  # pragma: no cover
        raise NotImplementedError

    def receive(self, msg, h):  # pragma: no cover - interface
        raise NotImplementedError


class _BitSender(Comm):
    """MLP sender: Linear 64->16, hard > 0, ST through sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Linear(MSG_DIM, N_BITS)

    def send(self, x, *, train, key, eng_bits, noise=None):
        m = self.encode(x)
        hard = (m > 0).astype(mx.float32)
        if not train:
            return hard, hard
        soft = mx.sigmoid(m)
        return hard + (soft - mx.stop_gradient(soft)), hard  # exact: value == hard


class MLPComm(_BitSender):
    def __init__(self) -> None:
        super().__init__()
        self.decode = nn.Linear(N_BITS, MSG_DIM)

    def receive(self, msg, h):
        return self.decode(msg)


class ATOCComm(_BitSender):
    """Receiver cross-attends over the 16 received bit tokens only."""

    def __init__(self) -> None:
        super().__init__()
        self.bit_embed = nn.Embedding(2, 32)
        self.pos_embed = mx.random.normal((N_BITS, 32)) * 0.02
        self.query = nn.Linear(HID_DIM, 32)
        self.out = nn.Linear(32, MSG_DIM)

    def receive(self, msg, h):
        e = self.bit_embed.weight  # [2, 32]
        # differentiable token lookup: works for hard or ST bits
        tok = msg[..., None] * e[1] + (1.0 - msg[..., None]) * e[0]  # [B,16,32]
        tok = tok + self.pos_embed
        q = self.query(h)[:, None, :]  # [B,1,32]
        att = mx.softmax((q @ tok.transpose(0, 2, 1)) / math.sqrt(32.0), axis=-1)
        return self.out((att @ tok)[:, 0, :])


class DIALComm(Comm):
    """DIAL DRU: train sigma(m + 2*eps), eval hard m > 0."""

    SIGMA_N = 2.0
    noise_dim = N_BITS

    def sample_noise(self, rows, key):
        return mx.random.normal((rows, N_BITS), key=key)

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Linear(MSG_DIM, N_BITS)
        self.decode = nn.Linear(N_BITS, MSG_DIM)

    def send(self, x, *, train, key, eng_bits, noise=None):
        m = self.encode(x)
        hard = (m > 0).astype(mx.float32)
        if not train:
            return hard, hard
        eps = mx.random.normal(m.shape, key=key) if noise is None else noise
        return mx.sigmoid(m + self.SIGMA_N * eps), hard

    def receive(self, msg, h):
        return self.decode(msg)


class EngComm(Comm):
    def __init__(self) -> None:
        super().__init__()
        self.decode = nn.Linear(N_BITS, MSG_DIM)

    def send(self, x, *, train, key, eng_bits, noise=None):
        if eng_bits is None:
            raise ValueError("EngComm requires eng_bits")
        bits = eng_bits.astype(mx.float32)
        return bits, bits

    def receive(self, msg, h):
        return self.decode(msg)


class NoComm(Comm):
    def __init__(self) -> None:
        super().__init__()
        self.decode = nn.Linear(N_BITS, MSG_DIM)

    def send(self, x, *, train, key, eng_bits, noise=None):
        z = mx.zeros((x.shape[0], N_BITS), dtype=mx.float32)
        return z, z

    def receive(self, msg, h):
        return self.decode(msg)


class GTMComm(Comm):
    """K=4 symbols from A=16 over the 16-bit bus (rulings R2)."""

    noise_dim = K_SYMBOLS * ALPHABET

    def sample_noise(self, rows, key):
        return mx.random.gumbel((rows, K_SYMBOLS * ALPHABET), key=key)

    def __init__(self, constellation_noise_key: mx.array) -> None:
        super().__init__()
        self.encode_logits = nn.Linear(MSG_DIM, K_SYMBOLS * ALPHABET)
        ang = 2.0 * math.pi * mx.arange(ALPHABET, dtype=mx.float32) / ALPHABET
        psk = mx.stack([mx.cos(ang), mx.sin(ang)], axis=-1)
        self.constellation = psk + 0.01 * mx.random.normal(
            (ALPHABET, 2), key=constellation_noise_key
        )
        self._cfg = GTMConfigMLX()
        n_t = (round(self._cfg.sample_rate_hz / self._cfg.gamma_hz)) * self._cfg.symbols_per_theta
        self._pool = _adaptive_pool_matrix(n_t, MSG_DIM)
        self.decode = nn.Linear(MSG_DIM + 2 * K_SYMBOLS, MSG_DIM)

    def send(self, x, *, train, key, eng_bits, noise=None):
        logits = self.encode_logits(x).reshape(x.shape[0], K_SYMBOLS, ALPHABET)
        hard_oh = mx.eye(ALPHABET, dtype=mx.float32)[mx.argmax(logits, axis=-1)]
        hard_bits = (hard_oh @ CODE_TABLE).reshape(x.shape[0], N_BITS)
        if not train:
            return hard_bits, hard_bits
        g = (mx.random.gumbel(logits.shape, key=key) if noise is None
             else noise.reshape(logits.shape))
        soft = mx.softmax(logits + g, axis=-1)  # tau = 1.0
        # hard one-hot of the noisy sample, ST gradient through the softmax
        hard_oh = mx.eye(ALPHABET, dtype=mx.float32)[mx.argmax(logits + g, axis=-1)]
        st = hard_oh + (soft - mx.stop_gradient(soft))  # value == hard_oh exactly
        msg = (st @ CODE_TABLE).reshape(x.shape[0], N_BITS)
        return msg, (hard_oh @ CODE_TABLE).reshape(x.shape[0], N_BITS)

    def receive(self, msg, h):
        oh = onehot_from_bits(msg)  # [B, 4, 16]
        mux = GammaThetaMultiplexerMLX(self._cfg, self.constellation)
        carrier = mux.carrier_from_onehot(oh)  # [B, T]
        pooled = carrier @ self._pool  # adaptive mean-pool -> [B, 64]
        iq = (oh @ self.constellation).reshape(msg.shape[0], 2 * K_SYMBOLS)
        return self.decode(mx.concatenate([pooled, iq], axis=-1))


def make_comm(arm: str, key: mx.array) -> Comm:
    """Build an arm. `key` (mx.random.key) fixes init: it seeds the global MLX
    RNG used by layer initialisers and, for GTM, the constellation noise."""
    k_init, k_noise = mx.random.split(key)
    mx.random.seed(int(mx.random.randint(0, 2**31 - 1, (1,), key=k_init).item()))
    if arm == "gtm":
        return GTMComm(k_noise)
    table = {
        "mlp": MLPComm,
        "atoc": ATOCComm,
        "dial": DIALComm,
        "eng": EngComm,
        "nocomm": NoComm,
    }
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}")
    return table[arm]()
