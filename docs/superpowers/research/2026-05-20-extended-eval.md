# Extended multi-seed evaluation (2026-05-20)

## Headline findings

- Transducer: learned MI = 2.120 bits beats the best baseline (relative_rep, 1.966) by Δ=+0.154 bits (Wilcoxon p=0.00195, d_z=+11.93; significant).
- Transducer null check: learned vs null MI Δ_median=+0.924 bits, p=0.00195, d_z=+221.50 — the learned arm is clearly above chance.
- GTM ablation: gtm MI=2.218 vs simple_gating MI=2.218 (Δ=+0.000, p=1, d_z=+0.00).
- GTM synchrony: gtm=0.199 vs simple_gating=0.078 (Δ=+0.120, p=0.00195, d_z=+13.84).
- Scale-robustness: real vs null CKNNA paired tests significant at 4/4 sizes; at N=512, real=0.727 vs null=0.019 (p=0.00195, d_z=+49.91).

## Configuration

- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)
- Runners: `run_transducer_benchmark`, `run_gtm_ablation`, `run_scale_robustness`
- Scale-robustness sizes: `[64, 128, 256, 512]`
- Bootstrap CI: 1000 resamples, seed=0; Wilcoxon two-sided (scipy.stats), Cohen's d_z = mean(diff)/std(diff, ddof=1).

## Transducer baselines

### Per-method (10 seeds)

| method | metric | mean | std | 95% CI |
|---|---|---|---|---|
| learned | entropy_bits | 5.9532 | 0.0088 | [5.9482, 5.9583] |
| learned | mi_bits | 2.1201 | 0.0012 | [2.1194, 2.1208] |
| procrustes | entropy_bits | 4.3487 | 0.2505 | [4.1901, 4.4853] |
| procrustes | mi_bits | 1.8412 | 0.0192 | [1.8290, 1.8523] |
| relative_rep | entropy_bits | 5.2872 | 0.0629 | [5.2465, 5.3225] |
| relative_rep | mi_bits | 1.9657 | 0.0121 | [1.9585, 1.9737] |
| vec2vec | entropy_bits | 4.7770 | 0.2076 | [4.6395, 4.8995] |
| vec2vec | mi_bits | 1.8866 | 0.0282 | [1.8659, 1.9017] |
| null | entropy_bits | 5.9532 | 0.0089 | [5.9482, 5.9584] |
| null | mi_bits | 1.1955 | 0.0034 | [1.1933, 1.1977] |

### Paired contrasts vs `learned`

| contrast | metric | n | p-value | Cohen's d_z | median diff |
|---|---|---|---|---|---|
| learned vs procrustes | mi_bits | 10 | 0.001953 | 13.3786 | 0.2767 |
| learned vs relative_rep | mi_bits | 10 | 0.001953 | 11.9308 | 0.1539 |
| learned vs vec2vec | mi_bits | 10 | 0.001953 | 7.8800 | 0.2260 |
| learned vs null | mi_bits | 10 | 0.001953 | 221.5034 | 0.9239 |

## GTM ablation

### Per-arm (10 seeds)

| method | metric | mean | std | 95% CI |
|---|---|---|---|---|
| gtm | accuracy | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| gtm | mi_bits | 2.2179 | 0.0015 | [2.2170, 2.2187] |
| gtm | synchrony_index | 0.1986 | 0.0074 | [0.1940, 0.2029] |
| simple_gating | accuracy | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| simple_gating | mi_bits | 2.2179 | 0.0015 | [2.2170, 2.2187] |
| simple_gating | synchrony_index | 0.0781 | 0.0037 | [0.0760, 0.0804] |
| null | accuracy | 0.0238 | 0.0076 | [0.0195, 0.0282] |
| null | mi_bits | 1.3316 | 0.0048 | [1.3285, 1.3345] |
| null | synchrony_index | 0.3250 | 0.0111 | [0.3185, 0.3320] |

### Paired contrasts

| contrast | metric | n | p-value | Cohen's d_z | median diff |
|---|---|---|---|---|---|
| gtm vs simple_gating | mi_bits | 10 | 1 | 0.0000 | 0.0000 |
| gtm vs null | mi_bits | 10 | 0.001953 | 156.7444 | 0.8852 |
| gtm vs simple_gating | synchrony_index | 10 | 0.001953 | 13.8419 | 0.1189 |
| gtm vs null | synchrony_index | 10 | 0.001953 | -9.4184 | -0.1269 |
| gtm vs simple_gating | accuracy | 10 | 1 | 0.0000 | 0.0000 |
| gtm vs null | accuracy | 10 | 0.001953 | 121.8077 | 0.9777 |

## Scale-robustness

| N | real CKNNA mean [CI] | null CKNNA mean [CI] | paired p | d_z | real HSIC mean [CI] | null HSIC mean [CI] | HSIC paired p |
|---|---|---|---|---|---|---|---|
| 64 | 0.2355 [0.2178, 0.2528] | 0.1623 [0.1522, 0.1705] | 0.001953 | 2.1043 | 0.3897 [0.3466, 0.4414] | 0.0017 [-0.0064, 0.0108] | 0.001953 |
| 128 | 0.2497 [0.2405, 0.2588] | 0.0826 [0.0786, 0.0870] | 0.001953 | 14.3242 | 0.4181 [0.4051, 0.4302] | 0.0007 [-0.0025, 0.0040] | 0.001953 |
| 256 | 0.4154 [0.4064, 0.4237] | 0.0423 [0.0391, 0.0456] | 0.001953 | 20.2718 | 0.4106 [0.4002, 0.4204] | 0.0022 [0.0003, 0.0040] | 0.001953 |
| 512 | 0.7269 [0.7186, 0.7346] | 0.0193 [0.0182, 0.0205] | 0.001953 | 49.9080 | 0.4072 [0.4017, 0.4126] | 0.0003 [-0.0007, 0.0012] | 0.001953 |

## Limitations

- Seeds=10 is the minimum recommended for paired Wilcoxon; the smallest two-sided p-value achievable with n=10 is ~0.002, so marginally non-significant results may simply be power-limited.
- The transducer pilot uses synthetic codebook pairs with default `steps=2000`; absolute MI numbers are not directly comparable across runners or task configurations.
- Bootstrap CIs are non-parametric over the 10 seed values; with n=10 the CI is wide and edge effects (ties at endpoints) are expected.
- Scale-robustness only reports CKNNA + HSIC; other geometry indices are not evaluated.
