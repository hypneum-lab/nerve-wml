# Extended multi-seed evaluation v3 (50 seeds)

Renf 4: massive-seed multiprocessing run to confirm v1/v2 findings at higher statistical power. Wilcoxon at n=50 reaches arbitrarily small p when effect sizes are large.

- transducer wall-clock: 67.8s
- gtm wall-clock: 359.6s
- scale wall-clock: 2.4s

## Transducer (MI in bits)

| method | mean ± std | 95% CI | n |
|---|---|---|---|
| vec2vec | 1.8875 ± 0.0314 | [1.8785, 1.8964] | 50 |
| procrustes | 1.8408 ± 0.0246 | [1.8340, 1.8475] | 50 |
| relative_rep | 1.9630 ± 0.0164 | [1.9583, 1.9674] | 50 |
| learned | 2.1202 ± 0.0011 | [2.1200, 2.1206] | 50 |
| null | 1.1967 ± 0.0063 | [1.1950, 1.1986] | 50 |

### Paired tests vs learned (Wilcoxon)

| method | p_value | cohens_dz | median_diff |
|---|---|---|---|
| vec2vec | 1.776e-15 | +7.40 | +0.2312 |
| procrustes | 1.776e-15 | +11.21 | +0.2776 |
| relative_rep | 1.776e-15 | +9.36 | +0.1584 |
| null | 1.776e-15 | +132.84 | +0.9240 |

## GTM ablation (synchrony_index)

| method | mean ± std | 95% CI | n |
|---|---|---|---|
| gtm | 0.2026 ± 0.0099 | [0.1999, 0.2054] | 50 |
| null | 0.3189 ± 0.0202 | [0.3136, 0.3246] | 50 |
| akorn | 0.0714 ± 0.0026 | [0.0706, 0.0721] | 50 |
| simple_gating | 0.0758 ± 0.0040 | [0.0747, 0.0769] | 50 |

### Paired tests vs gtm on synchrony (Wilcoxon)

| method | p_value | cohens_dz | median_diff |
|---|---|---|---|
| null | 1.776e-15 | -4.98 | -0.1180 |
| akorn | 1.776e-15 | +13.61 | +0.1321 |
| simple_gating | 1.776e-15 | +10.98 | +0.1264 |

## Scale-robustness (CKNNA)

| method | mean ± std | 95% CI | n |
|---|---|---|---|
| null_N128.0 | 0.0795 ± 0.0098 | [0.0768, 0.0823] | 50 |
| real_N256.0 | 0.4180 ± 0.0175 | [0.4132, 0.4230] | 50 |
| null_N512.0 | 0.0201 ± 0.0028 | [0.0193, 0.0208] | 50 |
| null_N256.0 | 0.0387 ± 0.0044 | [0.0375, 0.0400] | 50 |
| null_N1024.0 | 0.0098 ± 0.0011 | [0.0095, 0.0101] | 50 |
| real_N1024.0 | 0.6577 ± 0.0106 | [0.6547, 0.6606] | 50 |
| null_N64.0 | 0.1556 ± 0.0185 | [0.1506, 0.1607] | 50 |
| real_N128.0 | 0.2557 ± 0.0252 | [0.2489, 0.2628] | 50 |
| real_N64.0 | 0.2316 ± 0.0307 | [0.2227, 0.2401] | 50 |
| real_N512.0 | 0.7291 ± 0.0137 | [0.7254, 0.7331] | 50 |

## Configuration

- 50 seeds (0..49), multiprocessing pool.
- transducer: steps=2000 (default). gtm: steps=2000. scale: sizes=(64,128,256,512,1024).
- Bootstrap CI: 2000 resamples per metric.
