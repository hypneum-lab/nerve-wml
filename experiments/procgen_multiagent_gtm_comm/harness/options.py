"""Registered training-option presets of the N10-C harness (no mlx import).

Amendment 2026-10-25 (trunk collapse). Shared by ``harness.train`` and the
campaign launcher (``campaign.run_campaign``), which must not import mlx.
"""
from __future__ import annotations

from dataclasses import dataclass

FROZEN_PRESET = "frozen-20261024"


@dataclass(frozen=True)
class TrainOptions:
    """Registered training options (amendment 2026-10-25).

    The default is the configuration frozen by tag ``n10c-freeze-20261024``
    (kept selectable for reproducibility of that tag). Only the named
    ``PRESETS`` are valid; a registered run uses the preset fixed by the
    freeze (``campaign.guards.REGISTERED_TRAIN_OPTIONS``).

    * ``coin_reward``: per-agent reward for reaching its OWN coin, added to
      the team reward (+1 when both coins fall within delta_t = 20 steps).
      0 = team reward only (frozen).
    * ``adam_eps``: Adam epsilon (1e-8 = MLX default, frozen; 1e-5 = the
      usual PPO value).
    * ``mask_finished``: exclude agents that were already finished (frozen,
      action ignored) from the policy loss, the entropy bonus and the
      advantage normalisation statistics. Their forward pass (hence the
      message they send to the live partner) and their value loss are
      kept; the value target is the agent's own GAE return (team reward
      plus its own-coin reward, which is 0 once it is finished).
    * ``liveness``: trunk-liveness guard: ``off``, ``flag`` (record only)
      or ``abort`` (stop the run at the first flagged probe). The probe
      runs at step 0 and after the first update ending at or after every
      ``liveness_every`` pair-steps; metrics are logged in every non-off
      mode.
    * ``adam_bias_correction``: Adam bias correction (off = MLX default,
      frozen and gate-validated; on = the usual PPO implementations).
      Only the ``-bc`` comparison preset turns it on (PI 2026-10-25).
    * ``pretrained_trunk``: the IMPALA trunk is initialised from a
      single-agent CoinRun pre-training checkpoint (``harness/pretrain.py``)
      instead of at random; the rest of the network is initialised as before.
    """

    preset: str = FROZEN_PRESET
    coin_reward: float = 0.0
    adam_eps: float = 1e-8
    mask_finished: bool = False
    liveness: str = "off"
    liveness_every: int = 100_000
    pretrained_trunk: bool = False
    adam_bias_correction: bool = False


_GUARDS = dict(adam_eps=1e-5, mask_finished=True, liveness="abort")
PRESETS: dict[str, TrainOptions] = {
    FROZEN_PRESET: TrainOptions(),
    # (1) individual coin reward + guards
    "fix-a-coin": TrainOptions(preset="fix-a-coin", coin_reward=0.5, **_GUARDS),
    # (2) single-agent trunk pre-training, team reward only, + guards
    "fix-b-pretrain": TrainOptions(preset="fix-b-pretrain",
                                   pretrained_trunk=True, **_GUARDS),
    # A + B (PI choice 2026-10-25)
    "fix-c-coin-pretrain": TrainOptions(preset="fix-c-coin-pretrain",
                                        coin_reward=0.5,
                                        pretrained_trunk=True, **_GUARDS),
    # C with Adam bias correction: gate comparison only (PI 2026-10-25 12:10)
    "fix-c-coin-pretrain-bc": TrainOptions(preset="fix-c-coin-pretrain-bc",
                                           coin_reward=0.5,
                                           pretrained_trunk=True,
                                           adam_bias_correction=True,
                                           **_GUARDS),
}


def options_for(preset: str) -> TrainOptions:
    if preset not in PRESETS:
        raise ValueError(f"unknown training preset {preset!r} "
                         f"(known: {sorted(PRESETS)})")
    return PRESETS[preset]
