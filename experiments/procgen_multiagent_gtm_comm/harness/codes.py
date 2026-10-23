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



# Seeds and hosts (amendment 2026-10-23, second round). Six seeds: the five
# registered seeds plus 131 (smallest prime > 101 ending in digit 1, fixed
# before any run). The host is a block factor fixed per seed: all arms and
# both conditions of a seed train on that seed's host.
SEEDS: tuple[int, ...] = (0, 17, 42, 73, 101, 131)
HOSTS: tuple[str, ...] = ("macm1", "kxkm")
SEED_HOST: dict[int, str] = {0: "macm1", 17: "macm1", 42: "macm1",
                             73: "kxkm", 101: "kxkm", 131: "kxkm"}
