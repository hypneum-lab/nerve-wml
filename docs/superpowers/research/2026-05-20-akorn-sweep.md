# AKOrN parametrisation sweep

_Generated: 2026-05-20, wall-clock 937.4s._

Sweep: n_oscillators ∈ {32, 64, 128} × n_steps ∈ {8, 16, 32} × lr ∈ {0.01, 0.02, 0.05} = 27 cells × 5 seeds, train_steps=200.

## Headline

**Verdict: WEAKENED** — best cell reaches synchrony 0.4542, crossing the 0.15 threshold toward GTM's ~0.20. AKOrN's earlier clustering with simple_gating was a sub-parametrisation artifact.

## Top 3 cells (synchrony_index, mean ± std)

| cell_id | synchrony | accuracy | mi_bits |
|---------|-----------|----------|---------|
| `n64_s32_lr0.05` | 0.4542 ± 0.1624 | 0.1527 ± 0.0544 | 1.3866 ± 0.0324 |
| `n128_s32_lr0.05` | 0.3900 ± 0.2776 | 0.5342 ± 0.2337 | 1.7040 ± 0.2564 |
| `n128_s32_lr0.02` | 0.3824 ± 0.1695 | 0.5121 ± 0.3920 | 1.7257 ± 0.3924 |

Baselines for reference (Plan A.4, 2000 train_steps, 5 seeds):

| arm | synchrony_index |
|-----|-----------------|
| GTM | 0.2004 |
| simple_gating | 0.0762 |
| AKOrN (32 osc, 8 steps, lr=0.02) | 0.0727 |

## Top 10 cells by synchrony_index

| rank | cell_id | n_osc | n_steps | lr | synchrony (mean ± std) | 95% CI | accuracy | mi_bits |
|------|---------|-------|---------|----|------------------------|--------|----------|---------|
| 1 | `n64_s32_lr0.05` | 64 | 32 | 0.05 | 0.4542 ± 0.1624 | [0.3114, 0.5878] | 0.1527 ± 0.0544 | 1.3866 ± 0.0324 |
| 2 | `n128_s32_lr0.05` | 128 | 32 | 0.05 | 0.3900 ± 0.2776 | [0.1732, 0.6201] | 0.5342 ± 0.2337 | 1.7040 ± 0.2564 |
| 3 | `n128_s32_lr0.02` | 128 | 32 | 0.02 | 0.3824 ± 0.1695 | [0.2372, 0.5227] | 0.5121 ± 0.3920 | 1.7257 ± 0.3924 |
| 4 | `n64_s16_lr0.05` | 64 | 16 | 0.05 | 0.3100 ± 0.1723 | [0.2083, 0.4828] | 0.6605 ± 0.3469 | 1.8620 ± 0.3526 |
| 5 | `n32_s32_lr0.05` | 32 | 32 | 0.05 | 0.2837 ± 0.0634 | [0.2391, 0.3448] | 0.8627 ± 0.2746 | 2.0741 ± 0.2863 |
| 6 | `n128_s16_lr0.05` | 128 | 16 | 0.05 | 0.2643 ± 0.0809 | [0.1936, 0.3317] | 0.8415 ± 0.2271 | 2.0369 ± 0.2525 |
| 7 | `n32_s32_lr0.02` | 32 | 32 | 0.02 | 0.2338 ± 0.0157 | [0.2224, 0.2484] | 1.0000 ± 0.0000 | 2.2179 ± 0.0017 |
| 8 | `n64_s32_lr0.02` | 64 | 32 | 0.02 | 0.1850 ± 0.0319 | [0.1582, 0.2132] | 0.9181 ± 0.1070 | 2.1198 ± 0.1262 |
| 9 | `n128_s32_lr0.01` | 128 | 32 | 0.01 | 0.1680 ± 0.0279 | [0.1448, 0.1911] | 0.9875 ± 0.0188 | 2.2018 ± 0.0234 |
| 10 | `n32_s32_lr0.01` | 32 | 32 | 0.01 | 0.1639 ± 0.0134 | [0.1536, 0.1769] | 1.0000 ± 0.0000 | 2.2179 ± 0.0017 |

## Bottom 5 cells by synchrony_index

| cell_id | n_osc | n_steps | lr | synchrony (mean ± std) | 95% CI |
|---------|-------|---------|----|------------------------|--------|
| `n64_s8_lr0.02` | 64 | 8 | 0.02 | 0.0647 ± 0.0017 | [0.0633, 0.0661] |
| `n128_s16_lr0.01` | 128 | 16 | 0.01 | 0.0640 ± 0.0022 | [0.0622, 0.0659] |
| `n128_s8_lr0.02` | 128 | 8 | 0.02 | 0.0635 ± 0.0026 | [0.0615, 0.0662] |
| `n64_s8_lr0.01` | 64 | 8 | 0.01 | 0.0633 ± 0.0016 | [0.0620, 0.0647] |
| `n128_s8_lr0.01` | 128 | 8 | 0.01 | 0.0587 ± 0.0014 | [0.0574, 0.0599] |

## Interpretation

At least one AKOrN config drives the synchrony index above 0.15, suggesting that the earlier ~0.07 reading reflected an under-parametrised oscillator head rather than an intrinsic property of coupled-oscillator multiplexers. The Plan A.4 robust-distinctness conclusion should be revisited with the best-cell config or a hyperparameter-fair comparison protocol.

## Caveat

This sweep uses the minimalist `KuramotoMultiplexer` in `track_p.transducer_baselines` — learned natural frequencies and coupling matrix plus a linear readout. It is **not** Miyato et al.'s full AKOrN with phase-aware readout, block-structured coupling, or stimulus-driven driving forces. A full AKOrN port could still behave differently; this result only rules out a sub-parametrisation explanation for the minimalist flavour benchmarked in Plan A.4.
