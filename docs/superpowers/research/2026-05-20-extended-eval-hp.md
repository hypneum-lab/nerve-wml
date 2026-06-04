# Extended multi-seed × HP-grid evaluation (2026-05-20)

Companion to `2026-05-20-extended-eval.md`. Where the original ran each transducer at default HPs, this sweeps the real knobs wired through by Plan A.3 and reports best-of-grid performance + paired-Wilcoxon contrasts at each method's best HP.

## Headline findings

- Learned best HP = lr=0.03 → MI = 2.120 bits [2.119, 2.121] over 5 seeds.
- vs Procrustes (best HP n/a, MI=1.840): Δ=+0.280 bits, Wilcoxon p=0.0625, d_z=+19.82 (n.s.).
- vs vec2vec (best HP lambda_cycle=10.0, MI=1.897): Δ=+0.223 bits, Wilcoxon p=0.0625, d_z=+12.70 (n.s.).
- vs relative_rep (best HP n_anchors=64, MI=2.120): Δ=-0.000 bits, Wilcoxon p=0.625, d_z=-0.71 (n.s.).
- HP sensitivity floor: learned worst-HP MI=1.366 (lr=0.001); vec2vec worst-HP MI=1.673 (lambda_cycle=1.0); relative_rep worst-HP MI=1.915 (n_anchors=8).

## Per-method HP sweeps

### Learned transducer (`lr`)

| lr | mean | std | 95% CI | best |
|---|---|---|---|---|
| 0.001 | 1.3655 | 0.0127 | [1.3527, 1.3755] | |
| 0.003 | 2.0764 | 0.0069 | [2.0700, 2.0825] | |
| 0.01 | 2.1190 | 0.0020 | [2.1170, 2.1207] | |
| 0.03 | 2.1199 | 0.0009 | [2.1191, 2.1208] | ★ |

### vec2vec (`lambda_cycle`)

| lambda_cycle | mean | std | 95% CI | best |
|---|---|---|---|---|
| 1.0 | 1.6728 | 0.0391 | [1.6422, 1.7108] | |
| 10.0 | 1.8967 | 0.0161 | [1.8836, 1.9095] | ★ |
| 100.0 | 1.8828 | 0.0496 | [1.8323, 1.9169] | |

### relative_rep (`n_anchors`)

| n_anchors | mean | std | 95% CI | best |
|---|---|---|---|---|
| 8 | 1.9148 | 0.0096 | [1.9065, 1.9231] | |
| 16 | 1.9447 | 0.0161 | [1.9284, 1.9555] | |
| 32 | 1.9605 | 0.0065 | [1.9551, 1.9659] | |
| 64 | 2.1204 | 0.0012 | [2.1192, 2.1215] | ★ |

### Procrustes (closed-form, no HP)

- mean MI = 1.8399 ± 0.0122 bits (95% CI [1.8281, 1.8495])

## Paired Wilcoxon at best HPs

| contrast | best HPs | n | p-value | Cohen's d_z | median diff | Δ mean |
|---|---|---|---|---|---|---|
| learned vs procrustes | lr=0.03 vs n/a (closed-form) | 5 | 0.0625 | 19.8190 | 0.2790 | 0.2800 |
| learned vs vec2vec | lr=0.03 vs lambda_cycle=10.0 | 5 | 0.0625 | 12.7041 | 0.2163 | 0.2232 |
| learned vs relative_rep | lr=0.03 vs n_anchors=64 | 5 | 0.625 | -0.7141 | 0.0000 | -0.0005 |

## Configuration

- Seeds: `[0, 1, 2, 3, 4]` (n=5)
- `lr` grid: `[0.001, 0.003, 0.01, 0.03]`
- `lambda_cycle` grid: `[1.0, 10.0, 100.0]`
- `n_anchors` grid: `[8, 16, 32, 64]`
- Training steps: learned=2000, vec2vec=2000, relative_rep=500, procrustes=500.
- Bootstrap CI: 500 resamples, seed=0; Wilcoxon two-sided (scipy.stats), Cohen's d_z = mean(diff)/std(diff, ddof=1).

## Notes & limitations

- Procrustes is non-tunable (closed-form orthogonal alignment), so it appears as a single multi-seeded row rather than a grid.
- The `lambda_cycle` grid is intentionally small (3 values) to keep total wall-clock under ~5 minutes; vec2vec is the slowest runner because it trains a full encoder–decoder pair per seed.
- With n=5 paired seeds, the smallest achievable two-sided Wilcoxon p-value is 0.0625 — borderline p-values may reflect power limits rather than no effect.
- Best-of-grid selection is mildly optimistic (a hold-out HP would be stricter); the worst-of-grid row provides a sensitivity floor.
