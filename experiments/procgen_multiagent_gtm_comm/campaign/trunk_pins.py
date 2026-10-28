"""Pinned SHA-256 of the registered pre-trained trunks (written by
``campaign.pin_trunks``, committed by the freeze tag; code, so it cannot
change after the tag without failing the launch guard).

SHARED: registered seed -> SHA-256 of its shared trunk
(``SeedSequence([seed, 99])``). CONTROL: "<condition>/<arm>" -> SHA-256 of
the own trunk of that run of the per-run-pretraining control (seed 101,
``SeedSequence([101, 99, arm_code, condition_code])``). Empty until the
pre-freeze registered pre-trainings (freeze plan, phase T).
"""
SHARED: dict[int, str] = {
    0: "9d2a2acf83c065bdb10e003eb284429b75587d18aa0a27b149ff02eb8f4b4389",
    17: "807e9fff9c1f62854a8ab2663eda01b58a188b437c4b6271b37abd911d8769b6",
    42: "41700e7c67c87c74f7f05b9e54d3ac84e424a5e4d9f8e7d863d5e878f8069594",
    73: "2f6daec72c44c46b9a985515f8d53626542517bbf29f34a69731b93956edb013",
    101: "a7ed1a66ed823bfa66a0d0f3a5f292d71f80e548a1438d67a42e37c7dd6e014f",
    131: "6f6998da8bff141ab27c3c69dfefcbcb294b6289147391335150456ec9a9b77d",
}
CONTROL: dict[str, str] = {
    "closed/dial": "3e6cd13d2edd0bfa3f8f96abd82a6f421cff02af82565349af4a505b777bacc4",
    "closed/eng": "8589fbbb507612fabaf48be2eb227398b7655bc91ccdc918a6a4059ea799b2c7",
    "closed/gtm": "ea4c865765eee8710222022da97a1db62094336cb8f6cc1f158689ec9f3f828e",
    "closed/nocomm": "1eb97f49ab20b717a4e2b7b214d09b4eef83edee4ec9d113af0a747612aec6d5",
    "registered/atoc": "0a08208a7ee104f53e9f9e92858a07e7ea6e4bef0192bc3677541d7e6e106125",
    "registered/dial": "de86cbea9126f4acf5edb0916e70663c07cdaed260cc443fbece0b5ee959ea81",
    "registered/eng": "a45919d08413c107305e3f3a09923de24fa8fa128f1d40f02230b9d03a64af8b",
    "registered/gtm": "60959f0784a89609fe2ab40836cac5bca03d2265f09a07d0a57737ad39bf925c",
    "registered/mlp": "c52a0936d97180a250d7c4d28cafaec4a3895bf6c0e58b842196b808a79bd72c",
    "registered/nocomm": "cee1e3b8b438adc5f67b9b7f0509ae6be0496ac7dc410c6d0ff26e915a7fe7f7",
}
