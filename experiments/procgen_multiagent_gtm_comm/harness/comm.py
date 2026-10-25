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
# VQ arm (N10-C-bis addendum 2026-10-25, Arm 3; PI decisions): d_e = 8,
# beta = 0.25 (van den Oord et al. 2017), lambda_VQ = 1.0, plain VQ (no
# information-bottleneck term, no gate).
VQ_D_E = 8
VQ_BETA = 0.25
VQ_LAMBDA = 1.0

# Arm families (N10-C-bis addendum, Arm 2a implementation note): arms whose
# sender is the engineered reference (``eng_bits`` required) and arms whose
# GTM constellation gradient follows the cosine plasticity schedule.
ENG_SENDER_ARMS: tuple[str, ...] = ("eng", "eng_gtmrx")
PLASTIC_ARMS: tuple[str, ...] = ("gtm", "eng_gtmrx", "gtm_warm", "gtm_mis")
GTM_SENDER_ARMS: tuple[str, ...] = ("gtm", "gtm_mis", "gtm_warm")

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
        # GTMBridge straight-through path: the receiver gets the ST one-hot
        # [K, A] itself (flattened to [B, K*A]); the bus still validates the
        # 16 hard bits returned alongside.
        return (st.reshape(x.shape[0], K_SYMBOLS * ALPHABET),
                (hard_oh @ CODE_TABLE).reshape(x.shape[0], N_BITS))

    def receive(self, msg, h):
        if msg.shape[-1] == K_SYMBOLS * ALPHABET:  # training: ST one-hot
            oh = msg.reshape(msg.shape[0], K_SYMBOLS, ALPHABET)
        else:  # evaluation / probes: rebuild the one-hot from the 16 bits
            oh = onehot_from_bits(msg)  # [B, 4, 16], exact for hard bits
        mux = GammaThetaMultiplexerMLX(self._cfg, self.constellation)
        carrier = mux.carrier_from_onehot(oh)  # [B, T]
        pooled = carrier @ self._pool  # adaptive mean-pool -> [B, 64]
        iq = (oh @ self.constellation).reshape(msg.shape[0], 2 * K_SYMBOLS)
        return self.decode(mx.concatenate([pooled, iq], axis=-1))


class EngGTMRxComm(GTMComm):
    """N10-C-bis Arm 2a ``eng_gtmrx``: engineered sender, GTM receive path.

    The sender returns the N10-C engineered 16-bit message bit for bit (as
    ``EngComm.send``); the receiver is ``GTMComm.receive`` (one-hot rebuilt
    from the 16 bits, gamma/theta carrier, pooled samples + IQ coordinates,
    ``decode``), with the trainable constellation. The engineered field
    boundaries (1 + 5 + 10 bits) do not align with the four 4-bit symbols
    (disclosed in the addendum). ``encode_logits`` is inherited and stays
    untrained (no sender gradient): its checkpoint entry is what
    ``gtm_warm`` excludes when warm-starting.
    """

    noise_dim = 0  # deterministic sender: no Gumbel noise to store

    def sample_noise(self, rows, key):
        return None

    def send(self, x, *, train, key, eng_bits, noise=None):
        if eng_bits is None:
            raise ValueError("EngGTMRxComm requires eng_bits")
        bits = eng_bits.astype(mx.float32)
        return bits, bits


class VQComm(Comm):
    """N10-C-bis Arm 3 ``vq``: plain VQ baseline with GTM's symbol structure.

    Sender: Linear(64 -> K * d_e) reshaped to [K=4, d_e=8]; each slot is
    quantised to the nearest of A = 16 codebook vectors (one codebook shared
    by the 4 slots); the 4 indices go on the bus as 4-bit nibbles
    (``CODE_TABLE``, ruling R2): exactly 16 hard bits, in training too.
    Straight-through estimator: the receiver gets ``sg(q) + (z - sg(z))``
    (value exactly the codebook rows, gradient to the encoder only, as in
    van den Oord et al. 2017). Receiver: codebook lookup of the 4 received
    indices -> [4, d_e] -> Linear(K * d_e -> 64); no gamma/theta carrier.
    ``aux_loss(x)`` = lambda_VQ * (codebook loss + beta * commitment loss),
    recomputed from the sender input ``x`` (no cached state).
    """

    noise_dim = 0  # deterministic sender

    def __init__(self) -> None:
        super().__init__()
        self.encode = nn.Linear(MSG_DIM, K_SYMBOLS * VQ_D_E)
        # uniform(-1/A, 1/A) codebook initialisation (VQ-VAE reference
        # implementations); drawn from the global RNG seeded by make_comm
        self.codebook = mx.random.uniform(
            -1.0 / ALPHABET, 1.0 / ALPHABET, (ALPHABET, VQ_D_E))
        self.decode = nn.Linear(K_SYMBOLS * VQ_D_E, MSG_DIM)

    def _quantize(self, x):
        """x [B, 64] -> (z [B, K, d_e], indices [B, K] int32, q [B, K, d_e])."""
        z = self.encode(x).reshape(x.shape[0], K_SYMBOLS, VQ_D_E)
        d = ((z[:, :, None, :] - self.codebook[None, None, :, :]) ** 2).sum(-1)
        idx = mx.argmin(d, axis=-1).astype(mx.int32)  # [B, K]
        return z, idx, self.codebook[idx]

    def send(self, x, *, train, key, eng_bits, noise=None):
        z, idx, q = self._quantize(x)
        hard_bits = (mx.eye(ALPHABET, dtype=mx.float32)[idx] @ CODE_TABLE)
        hard_bits = hard_bits.reshape(x.shape[0], N_BITS)
        if not train:
            return hard_bits, hard_bits
        st = mx.stop_gradient(q) + (z - mx.stop_gradient(z))  # value == q
        return st.reshape(x.shape[0], K_SYMBOLS * VQ_D_E), hard_bits

    def aux_loss(self, x):
        """VQ loss on the sender rows: codebook + beta * commitment, * lambda."""
        z, _, q = self._quantize(x)
        codebook = ((mx.stop_gradient(z) - q) ** 2).sum(-1).mean()
        commitment = ((z - mx.stop_gradient(q)) ** 2).sum(-1).mean()
        return VQ_LAMBDA * (codebook + VQ_BETA * commitment)

    def receive(self, msg, h):
        if msg.shape[-1] == K_SYMBOLS * VQ_D_E:  # training: ST codebook rows
            q = msg.reshape(msg.shape[0], K_SYMBOLS, VQ_D_E)
        else:  # evaluation / probes: look the 4 received indices up
            q = self.codebook[bits_to_indices(msg)]
        return self.decode(q.reshape(msg.shape[0], K_SYMBOLS * VQ_D_E))


def codebook_usage(bits) -> np.ndarray:
    """Per-codeword counts [16] of the symbol indices carried by 16-bit
    messages ``bits`` [..., 16] (descriptive VQ codebook usage)."""
    b = (np.asarray(bits, dtype=np.float32) > 0.5).reshape(-1, K_SYMBOLS, 4)
    idx = (b * np.array([8, 4, 2, 1], dtype=np.float32)).sum(-1).astype(np.int64)
    return np.bincount(idx.reshape(-1), minlength=ALPHABET)


def make_comm(arm: str, key: mx.array) -> Comm:
    """Build an arm. `key` (mx.random.key) fixes init: it seeds the global MLX
    RNG used by layer initialisers and, for GTM, the constellation noise.

    The seeding path is identical for every arm (N10-C arms unchanged by
    the N10-C-bis additions; see ``tests/unit/n10c/test_bis_invariance.py``).
    """
    k_init, k_noise = mx.random.split(key)
    mx.random.seed(int(mx.random.randint(0, 2**31 - 1, (1,), key=k_init).item()))
    if arm in GTM_SENDER_ARMS:  # gtm, gtm_mis, gtm_warm: the GTMComm module
        return GTMComm(k_noise)
    if arm == "eng_gtmrx":
        return EngGTMRxComm(k_noise)
    table = {
        "mlp": MLPComm,
        "atoc": ATOCComm,
        "dial": DIALComm,
        "eng": EngComm,
        "nocomm": NoComm,
        "vq": VQComm,
    }
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}")
    return table[arm]()
