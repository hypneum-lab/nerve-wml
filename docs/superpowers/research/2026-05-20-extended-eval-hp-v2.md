# Extended multi-seed × HP-grid evaluation v2 (2026-05-20)

Expanded version of `2026-05-20-extended-eval-hp.md`. Addresses the v1 limitation (4 HP cells × 5 seeds) by using larger grids (8 cells per axis) and stronger statistical power (10 seeds). Confirms whether relative_rep's tie with learned persists or whether wider search reveals a decisive winner.

## Headline findings

- Learned best HP = lr=0.3 → MI = 2.120 bits [2.120, 2.121] over 10 seeds.
- vs Procrustes (best HP n/a, MI=1.841): Δ=+0.279 bits, Wilcoxon p=0.00195, d_z=+13.52 (significant).
- vs vec2vec (best HP lambda_cycle=30.0, MI=1.943): Δ=+0.177 bits, Wilcoxon p=0.00195, d_z=+5.00 (significant).
- vs relative_rep (best HP n_anchors=64, MI=2.120): Δ=+0.000 bits, Wilcoxon p=1, d_z=+0.00 (n.s.).
- HP sensitivity floor: learned worst-HP MI=1.195 (lr=0.0001); vec2vec worst-HP MI=1.312 (lambda_cycle=0.1); relative_rep worst-HP MI=1.926 (n_anchors=4).

## Per-method HP sweeps

### Learned transducer (`lr`)

| lr | mean | std | 95% CI | best |
|---|---|---|---|---|
| 0.0001 | 1.1948 | 0.0032 | [1.1929, 1.1969] | |
| 0.0003 | 1.1980 | 0.0035 | [1.1957, 1.2000] | |
| 0.001 | 1.3667 | 0.0150 | [1.3581, 1.3773] | |
| 0.003 | 2.0730 | 0.0080 | [2.0677, 2.0781] | |
| 0.01 | 2.1181 | 0.0020 | [2.1169, 2.1193] | |
| 0.03 | 2.1199 | 0.0010 | [2.1193, 2.1205] | |
| 0.1 | 2.1201 | 0.0012 | [2.1194, 2.1208] | |
| 0.3 | 2.1203 | 0.0010 | [2.1197, 2.1208] | ★ |

### vec2vec (`lambda_cycle`)

| lambda_cycle | mean | std | 95% CI | best |
|---|---|---|---|---|
| 0.1 | 1.3123 | 0.6593 | [0.8361, 1.6519] | |
| 1.0 | 1.6677 | 0.0422 | [1.6424, 1.6917] | |
| 3.0 | 1.6968 | 0.0444 | [1.6705, 1.7240] | |
| 10.0 | 1.8866 | 0.0282 | [1.8663, 1.9019] | |
| 30.0 | 1.9428 | 0.0336 | [1.9204, 1.9612] | ★ |
| 100.0 | 1.8727 | 0.0517 | [1.8385, 1.9021] | |
| 300.0 | 1.8489 | 0.0397 | [1.8272, 1.8720] | |
| 1000.0 | 1.8512 | 0.0547 | [1.8137, 1.8792] | |

### relative_rep (`n_anchors`)

| n_anchors | mean | std | 95% CI | best |
|---|---|---|---|---|
| 4 | 1.9260 | 0.0284 | [1.9107, 1.9484] | |
| 8 | 1.9263 | 0.0157 | [1.9169, 1.9358] | |
| 12 | 1.9432 | 0.0183 | [1.9332, 1.9537] | |
| 16 | 1.9372 | 0.0144 | [1.9288, 1.9454] | |
| 24 | 1.9520 | 0.0159 | [1.9424, 1.9616] | |
| 32 | 1.9657 | 0.0121 | [1.9586, 1.9731] | |
| 48 | 2.0279 | 0.0084 | [2.0228, 2.0329] | |
| 64 | 2.1203 | 0.0010 | [2.1197, 2.1208] | ★ |

### Procrustes (closed-form, no HP)

- mean MI = 1.8412 ± 0.0192 bits (95% CI [1.8291, 1.8522])

## Paired Wilcoxon at best HPs

| contrast | best HPs | n | p-value | Cohen's d_z | median diff | Δ mean |
|---|---|---|---|---|---|---|
| learned vs procrustes | lr=0.3 vs n/a (closed-form) | 10 | 0.001953 | 13.5167 | 0.2767 | 0.2791 |
| learned vs vec2vec | lr=0.3 vs lambda_cycle=30.0 | 10 | 0.001953 | 4.9959 | 0.1785 | 0.1775 |
| learned vs relative_rep | lr=0.3 vs n_anchors=64 | 10 | 1 | 0.0000 | 0.0000 | 0.0000 |

## Comparison with v1 (4-cell × 5-seed)

v2 expanded the grids to improve statistical power and explore beyond v1's range:

- `lr`: [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3] (8 cells, was (1e-3, 3e-3, 1e-2, 3e-2))
- `lambda_cycle`: [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0] (8 cells, was (1.0, 10.0, 100.0))
- `n_anchors`: [4, 8, 12, 16, 24, 32, 48, 64] (8 cells, was (8, 16, 32, 64) with clamping)
- Seeds: 10 (was 5)

With n=10 paired seeds, two-sided Wilcoxon gains power: smallest achievable p ≈ 0.002 vs. 0.0625 in v1. Any p-value ties at the reported significance threshold now reflect true absence of effect rather than power limits.

## Configuration

- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` (n=10)
- `lr` grid: `[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]`
- `lambda_cycle` grid: `[0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]`
- `n_anchors` grid: `[4, 8, 12, 16, 24, 32, 48, 64]` (clamped to ≤ alphabet_size=64)
- Training steps: learned=2000, vec2vec=2000, relative_rep=500, procrustes=500.
- Bootstrap CI: 500 resamples, seed=0; Wilcoxon two-sided (scipy.stats), Cohen's d_z = mean(diff)/std(diff, ddof=1).

## Notes & limitations

- Procrustes is non-tunable (closed-form orthogonal alignment), so it appears as a single multi-seeded row rather than a grid.
- The `lambda_cycle` grid (8 values) and `n_anchors` grid (8 values) expand the search space significantly; total wall-clock ~10 minutes expected with 10 seeds (8 + 8 + 8 + 1 = 25 HP cells × 10 seeds × ~2s/cell).
- Best-of-grid selection is mildly optimistic (a hold-out HP would be stricter); the worst-of-grid row provides a sensitivity floor.
- `n_anchors` values ≥ 65 are clamped to 64 (alphabet_size), creating duplicate runs; grid uses 4–64 within the legal range.
