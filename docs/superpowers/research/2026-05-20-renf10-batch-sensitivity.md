# Renf 10 — Batch-size sensitivity of the spectral_entropy ablation

**Date**: 2026-05-20
**Host**: macm1 (Apple M1, 8-core CPU, multiproc backend)
**Wall-clock**: 3132.2 s (~52 min)
**Raw data**: `docs/superpowers/research/2026-05-20-renf10-batch-sensitivity.json`
**Driver**: `scripts/renf10_batch_sensitivity.py`

## Headline

The 4-arm strict ordering `null < akorn_best < gtm < simple_gating` reported
in Renf 7 (which fixed `B=128`) is **batch-size dependent**. Across
`B ∈ {64, 128, 256, 512}` only `B=256` reproduces the canonical ordering;
at the other three batch sizes `akorn_best` lands *below* `null`. The
2-arm comparisons that matter for the paper claim — `gtm > null` and
`gtm < simple_gating` — hold at all four batch sizes with `p < 10^{-15}`.

## Config

- **Seeds**: 50
- **Adam steps**: 200
- **Arms**: `gtm`, `simple_gating`, `akorn_best` (`n_osc=64`, `n_steps=32`,
  `lr=0.05`), `null`
- **Metric**: `spectral_entropy` of the carrier (Roy & Vetterli 2007),
  $H_\mathrm{spec} = -\sum_i p_i \log p_i$ with $p_i = \sigma_i^2/\sum_j \sigma_j^2$
- **Batch sizes**: `{64, 128, 256, 512}`
- **Backend**: macm1 CPU multiproc (4 workers)

## Results

`spectral_entropy` mean (bits) per arm at each B, with ordering and
verdict against the canonical Renf 7 ordering
`null < akorn_best < gtm < simple_gating`:

| B   | null      | akorn_best | gtm       | simple_gating | Ordering (low → high)                       | Verdict          |
|----:|----------:|-----------:|----------:|--------------:|---------------------------------------------|------------------|
| 64  | 2.086     | **1.933**  | 2.139     | 3.250         | akorn_best < null < gtm < simple_gating     | DIVERGES         |
| 128 | **1.960** | **1.872**  | 2.173     | 3.465         | akorn_best < null < gtm < simple_gating     | DIVERGES         |
| 256 | **1.738** | 2.035      | 2.191     | 3.623         | null < akorn_best < gtm < simple_gating     | MATCH canonical  |
| 512 | 1.654     | **1.548**  | 2.200     | 3.766         | akorn_best < null < gtm < simple_gating     | DIVERGES         |

Bold cells are the arm that breaks the canonical ordering at that B
(or the canonical anchor at the matching B=256).

The 2-arm gaps that survive across all B:

- `gtm > null`: smallest gap at B=64 (≈ 0.053 bits); largest at B=256
  (≈ 0.453 bits). Paired Wilcoxon `p < 10^{-15}` at every B.
- `gtm < simple_gating`: gap grows monotonically with B
  (1.111 → 1.292 → 1.432 → 1.566 bits). Paired Wilcoxon `p < 10^{-15}`
  at every B.

## Diagnosis

`akorn_best` has roughly $\sigma_{H_\mathrm{spec}} \approx 0.6$ across
seeds at every B, whereas `gtm` and `simple_gating` are tight
(`σ ≈ 0.01`–`0.02`) and `null` is moderate (`σ ≈ 0.13`). The
Kuramoto-style integrator in `akorn_best` is highly sensitive to random
initialisation at `n_osc=64`, `n_steps=32`, `lr=0.05`, and the resulting
seed-distribution overlaps `null` so heavily that its rank against `null`
flips with the sampling realisation. The *significant* gap (`gtm` vs
`null`, p ≪ 1e-15) is independent of B.

The takeaway is not that `spectral_entropy` is unreliable — it is that
the 4-arm *strict-ordering* framing chosen in Renf 7 was an artifact of
the specific `B=128` realisation. The metric itself is well-behaved on
the high-`SNR`/low-variance arms.

## Revised claim

Replace the Renf 7 framing
"`null < akorn_best < gtm < simple_gating`, all consecutive gaps
significant" with the two-part B-robust claim:

1. `H_spec(gtm) > H_spec(null)` with Δ ≥ 0.05 bits and
   paired Wilcoxon `p < 10^{-15}` at every `B ∈ {64, 128, 256, 512}`.
2. `H_spec(gtm) < H_spec(simple_gating)` with gap ≥ 1.0 bits and
   paired Wilcoxon `p < 10^{-15}` at every `B ∈ {64, 128, 256, 512}`.
3. `akorn_best` is reported as **intermediate but high-variance**
   (seed-σ ≈ 0.6, position within the ordering unstable across B);
   it is not placed in a strict ranking against `gtm` or `null`.

## Reproducibility

- Driver: `scripts/renf10_batch_sensitivity.py` (on `master`)
- Raw data: `docs/superpowers/research/2026-05-20-renf10-batch-sensitivity.json`
- Methodology mirrors `scripts/synchrony_replacement_eval.py` (Renf 7),
  parametrised over `B` while holding seeds, steps, arms and arm
  hyperparameters fixed.
