"""16-bit message bus with registered enforcement (pre-registered protocol).

Asserts shape (n, 2, 16) and values in {0, 1} at evaluation for all arms and
in training for all arms except DIAL (whose training messages are soft DRU
outputs, disclosed). A violation is an implementation error (infrastructure
failure), never an outcome-based exclusion. The bus never inspects any
`finished` flag: a finished agent's frozen-observation message is delivered.
"""
from __future__ import annotations

import numpy as np

from .codes import ARM_CODES

N_BITS = 16


class MessageBusViolation(RuntimeError):  # noqa: N818 (public name)
    pass


class MessageBus:
    def __init__(self, arm: str) -> None:
        if arm not in ARM_CODES:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.bits_logged = 0

    def deliver(self, bits: np.ndarray, *, phase: str,
                pairs: np.ndarray | None = None) -> np.ndarray:
        """bits [n, 2, 16] -> partner-swapped [n, 2, 16] (out[:,0]=bits[:,1]).

        ``pairs`` (N10-C-bis ``gtm_mis`` arm): a derangement sigma of the n
        pairs; agent i of pair e then receives the message emitted by role
        1 - i of pair sigma(e) (``out[e, i] = bits[sigma(e), 1 - i]``). The
        shape and {0, 1} assertions and ``bits_logged`` are the same as for
        the default partner delivery. A ``pairs`` that is not a derangement
        (not a permutation, or with a fixed point) is a violation.
        """
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        bits = np.asarray(bits)
        if bits.ndim != 3 or bits.shape[1:] != (2, N_BITS):
            raise MessageBusViolation(f"expected shape (n, 2, 16), got {bits.shape}")
        check_binary = phase == "eval" or self.arm != "dial"
        if check_binary and not np.all((bits == 0) | (bits == 1)):
            raise MessageBusViolation(
                f"non-binary message for arm {self.arm!r} in phase {phase!r}"
            )
        if pairs is not None:
            sigma = np.asarray(pairs)
            if not is_derangement(sigma, bits.shape[0]):
                raise MessageBusViolation(
                    f"pairs is not a derangement of the {bits.shape[0]} pairs")
        self.bits_logged += N_BITS * bits.shape[0] * bits.shape[1]
        out = np.empty_like(bits)
        out[:, 0] = bits[:, 1]
        out[:, 1] = bits[:, 0]
        return out if pairs is None else out[sigma]


def is_derangement(sigma: np.ndarray, n: int) -> bool:
    """True if ``sigma`` is a permutation of range(n) without fixed points."""
    sigma = np.asarray(sigma)
    if sigma.shape != (n,) or n < 2:
        return False
    ident = np.arange(n)
    return bool(np.array_equal(np.sort(sigma), ident) and not (sigma == ident).any())
