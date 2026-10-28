"""Pinned SHA-256 of the registered pre-trained trunks (written by
``campaign.pin_trunks``, committed by the freeze tag; code, so it cannot
change after the tag without failing the launch guard).

SHARED: registered seed -> SHA-256 of its shared trunk
(``SeedSequence([seed, 99])``). CONTROL: "<condition>/<arm>" -> SHA-256 of
the own trunk of that run of the per-run-pretraining control (seed 101,
``SeedSequence([101, 99, arm_code, condition_code])``). Empty until the
pre-freeze registered pre-trainings (freeze plan, phase T).
"""
SHARED: dict[int, str] = {}
CONTROL: dict[str, str] = {}
