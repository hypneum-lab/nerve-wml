"""Manifest integer codes and per-run seed derivation (pre-registered)."""
from __future__ import annotations

import numpy as np

ARM_CODES: dict[str, int] = {
    "gtm": 0,
    "mlp": 1,
    "atoc": 2,
    "dial": 3,
    "eng": 4,
    "nocomm": 5,
}
CONDITION_CODES: dict[str, int] = {"registered": 0, "closed": 1}


def run_seed_sequence(seed: int, arm: str, condition: str) -> np.random.SeedSequence:
    """SeedSequence([seed, arm_code, cond_code]) for one (seed, arm, condition) run."""
    return np.random.SeedSequence([seed, ARM_CODES[arm], CONDITION_CODES[condition]])


def init_key_int(seed: int, arm: str, condition: str) -> int:
    """Integer seed of the run's policy-initialisation key (stream 1 of 4)."""
    init_ss = run_seed_sequence(seed, arm, condition).spawn(4)[1]
    return int(init_ss.generate_state(1)[0] % (2**31 - 1))
