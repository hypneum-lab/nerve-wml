# Golden test fixtures

Large fixtures (>100 MB) are not committed to git — GitHub rejects them.

## Sleep-EDF epochs (`sleep_edf_epochs.npz`, 312 MB)

Pre-extracted Sleep-EDF Database epochs used by the reproducibility
test suite. Local-only; regenerate via:

```bash
uv run python scripts/eeg_preprocess_sleep_edf.py \
    --subjects 0 1 2 3 4 5 6 7 8 9 \
    --out tests/golden/sleep_edf_epochs.npz
```

Or pull from the project Zenodo archive (DOI 10.5281/zenodo.19656342)
when the fixture is published as a versioned release artefact.

## Committed NPZ fixtures

Seven NPZ files are committed to `tests/golden/`. Each row gives the
file, the script that produced it, the test(s) that consume it, and
the SHA-256 digest (as printed by `shasum -a 256` on 2026-11-20,
commit `357da7c`).

| File | Producer script | Consumer test(s) | SHA-256 |
|------|-----------------|------------------|---------|
| `codes_dvnc.npz` | `scripts/baseline_dvnc.py` | _(analysis only; no automated test loads it directly)_ | `120c37956801b6eca8e3dc432c6679ea001e00cbe58c61d6c7648dddd85d818d` |
| `codes_mlp_lif.npz` | `scripts/save_codes_for_checks.py` | _(analysis scripts; no automated test loads it directly)_ | `71eda0bbba6fbfd8e7f2459ec5ed4e1b217736a3fef824dfa55deaf4597ab9b5` |
| `codes_mlp_lif_eeg_d128_spectro.npz` | `scripts/save_codes_eeg.py` (options not recorded; see commit `02fe868`) | _(analysis only)_ | `19dfa6e0f578e613d4ec15d2989624e98dd3ef80dd2bb522470a0d3fc3c07001` |
| `codes_mlp_lif_eeg_n10.npz` | `scripts/save_codes_eeg.py` (options not recorded; see commit `2a5d982`) | _(analysis only)_ | `8c9a7a8ccf57b12382ed4d3933f9e26645ff94959d50f0afe404034213840354` |
| `cycle_trace_4wmls_seed0.npz` | `scripts/freeze_golden.py` | `tests/golden/test_golden_regressions.py` | `141b1f4da65bf99e6195521ac9dcee455f47aa5dad81eabbe6e22c3ab75d679d` |
| `n10c_gtm_k4_a16.npz` | `scripts/n10c_make_gtm_golden.py` | `tests/unit/n10c/test_gtm_mlx.py` | `c81fc6596993e5b577708826e5be7061b7792cf8d4466f6c3db4bedbef21a523` |
| `transducers_merged.npz` | `scripts/freeze_golden.py` | `tests/golden/test_golden_regressions.py` | `bed50ca227d94c11a971a07f96056582b0890acab97d560dab1f8551d340dd8a` |

## Tolerance policy

From the docstring of `tests/golden/test_golden_regressions.py`:

- `test_cycle_trace_bit_stable` — integer code arrays must match the
  frozen NPZ **exactly** (bit-exact, no tolerance).
- `test_transducer_logits_bit_stable` — float logits use
  `rtol=atol=1e-6` to tolerate documented cross-platform RNG drift in
  the last few mantissa bits (tightened from the original 1e-14 in N3
  Task 9, 2026-05-10); any genuine algorithmic regression is still
  caught.
- Regenerating an NPZ to make a failing test pass is **not** allowed
  without a spec-sanctioned change; investigate first.
