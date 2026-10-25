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
    # N10-C-bis control arms (addendum 2026-10-25): appended, never
    # renumbered, so every N10-C seed stream is unchanged.
    "gtm_mis": 6,
    "eng_gtmrx": 7,
    "gtm_warm": 8,
    "vq": 9,
}
CONDITION_CODES: dict[str, int] = {"registered": 0, "closed": 1}
# The six N10-C arms (frozen, tag n10c-freeze-20261024) and the four
# N10-C-bis control arms (addendum 2026-10-25), in code order.
N10C_ARMS: tuple[str, ...] = ("gtm", "mlp", "atoc", "dial", "eng", "nocomm")
BIS_ARMS: tuple[str, ...] = ("gtm_mis", "eng_gtmrx", "gtm_warm", "vq")


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
# Registered per-host implementation settings (amendment 2026-10-23): the
# single source of truth for run_campaign (every run of that host), the
# stage-2 manifest and run.json. They are implementation settings, not
# registered quantities of the estimand; registered runs cannot override them.
HOST_SETTINGS: dict[str, dict] = {
    "kxkm": {"micro_batch_pairs": 512, "clear_cache": True, "n_workers": 8},
    "macm1": {"micro_batch_pairs": None, "clear_cache": False, "n_workers": 1},
}
SEED_HOST: dict[int, str] = {0: "macm1", 17: "macm1", 42: "macm1",
                             73: "kxkm", 101: "kxkm", 131: "kxkm"}
# Declared per-run-pretraining control (amendment 2026-10-25, PI decision
# 19:12): every run of this seed (a KXKM-AI seed) is repeated with its OWN
# pre-trained trunk, seeded SeedSequence([seed, 99, arm_code,
# condition_code]); the registered runs share SeedSequence([seed, 99]).
PER_RUN_TRUNK_SEED = 101
# N10-C-bis: every run trains and evaluates on KXKM-AI (addendum, Seeds,
# host and known deviations, item 1); the seed labels are the N10-C ones.
SEED_HOST_BIS: dict[int, str] = {s: "kxkm" for s in SEEDS}
# Fixed cross-seed cross-play pairing (role-0 seed, role-1 seed), wrapping.
CROSS_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (s, SEEDS[(k + 1) % len(SEEDS)]) for k, s in enumerate(SEEDS))

# Registered x86-64 Procgen build used for ALL held-out evaluations (KXKM-AI).
PROCGEN_VERSION = "0.10.7"
PROCGEN_X86_WHEEL = (
    "procgen-0.10.7-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
PROCGEN_X86_WHEEL_SHA256 = (
    "4b594d14e42f0f2166e59a9e294477b906eb99c90c3343b570b7124cbc865f53")
