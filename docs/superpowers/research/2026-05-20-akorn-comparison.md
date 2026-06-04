# AKOrN 4-Arm GTM Ablation — Result Note

**Date.** 2026-05-20
**Plan.** `docs/superpowers/plans/2026-05-20-A4-akorn-comparison.md`
**Status.** Smoke complete; 5 seeds × 4 arms × 3 metrics + paired Wilcoxon.

## Headline

- AKOrN's synchrony index (mean 0.0727) lands **essentially on top of
  simple_gating** (0.0762), **NOT** between GTM (0.2004) and the
  gating control — `akorn vs simple_gating` median diff = -0.0036,
  d_z = -0.99, p = 0.1250 (n.s.).
- GTM remains distinctly higher in synchrony than **both** the
  non-oscillator gating control AND the minimal Kuramoto-based
  multiplexer: `gtm vs akorn` d_z = +7.46, median diff = +0.1232.
- Accuracy and MI are saturated (1.0 / 2.22 bits) for all three
  trained arms; the null arm collapses as expected (acc ≈ 0.025).
- **Interpretation:** GTM's elevated synchrony is **NOT** a generic
  feature of "any oscillator-style head" — a coupled-oscillator
  multiplexer with learned natural frequencies and coupling matrix
  collapses to the same low-synchrony regime as plain learned
  gating. GTM's signature appears genuinely tied to its band
  structure rather than to oscillator dynamics per se.
- Caveat: this is a **minimal** Kuramoto unit (32 oscillators,
  8 Euler steps, linear read-out). It is not a full AKOrN
  re-implementation and the negative result must be re-tested with
  a deeper/wider configuration before it can be promoted to a
  claim against Miyato et al.

## Method

- Runner: `scripts/gtm_ablation_pilot.run_gtm_ablation`, 4 arms
  (`gtm`, `simple_gating`, `akorn`, `null`).
- 5 seeds (0..4) via `scripts.multi_seed.run_multi_seed`.
- `steps = 200` per arm; matches the slow-test budget.
- AKOrN config: `n_oscillators = 32`, `n_steps = 8`, `dt = 0.1`,
  trained with the same Adam(lr=0.02) + cross-entropy loop as the
  simple-gating control to keep budget fair.
- Synchrony index = fraction of carrier variance on the top
  principal mode (`_synchrony_index` in `gtm_ablation_pilot.py`).

## Result Table (5 seeds, mean ± std)

| arm           | accuracy        | mi_bits         | synchrony_index |
|---------------|-----------------|-----------------|-----------------|
| gtm           | 1.0000 ± 0.0000 | 2.2179 ± 0.0017 | 0.2004 ± 0.0133 |
| simple_gating | 1.0000 ± 0.0000 | 2.2179 ± 0.0017 | 0.0762 ± 0.0018 |
| akorn         | 1.0000 ± 0.0000 | 2.2179 ± 0.0017 | 0.0727 ± 0.0026 |
| null          | 0.0248 ± 0.0074 | 1.3301 ± 0.0045 | 0.2952 ± 0.0284 |

## Paired Wilcoxon (synchrony_index, n = 5)

| pair                       | p      | Cohen's d_z | median diff |
|----------------------------|--------|-------------|-------------|
| gtm vs akorn               | 0.0625 | +7.46       | +0.1232     |
| akorn vs simple_gating     | 0.1250 | -0.99       | -0.0036     |
| gtm vs simple_gating       | 0.0625 | +8.90       | +0.1216     |

n = 5 is the minimum sample size at which two-sided Wilcoxon can
reach p = 0.0625, so the two `gtm vs *` comparisons hit the
floor of the test even though the effect sizes (d_z ≈ +8) are
extreme. The `akorn vs simple_gating` row has both small d_z and
non-significant p — these two arms are statistically
indistinguishable on synchrony at this scale.

## Interpretation

The minimal Kuramoto multiplexer behaves like the simple-gating
control on the synchrony axis, not like GTM. This is the
**robust-distinctness** outcome enumerated in the plan: GTM's
elevated synchrony is preserved when a third, oscillator-based
contender is added, which strengthens the claim that GTM's
band-multiplexing is the actual driver rather than a generic
oscillator side-effect.

## Open Questions

- **Scaling.** Does a wider AKOrN (e.g. `n_oscillators = 128`)
  with longer integration (`n_steps = 64`) move closer to GTM,
  or does it stay locked at the gating-control floor? If the
  former, the present result is an under-parametrisation
  artefact; if the latter, the GTM-vs-AKOrN gap is real and
  band-multiplexing is the operative mechanism.
- **Longer Euler integration.** `n_steps = 8` may be too short
  to let coupling dynamics fully express; a sweep over
  `n_steps ∈ {1, 2, 4, 8, 16, 32, 64}` would resolve whether
  AKOrN's read-out is dominated by the initial bias injection
  vs. the coupled trajectory.
- **Full ODE-integrator AKOrN.** The Miyato et al. (ICLR 2025)
  implementation uses a more sophisticated integrator and
  learned per-stimulus oscillator pools. The minimal Kuramoto
  unit here trades fidelity for budget — replicating the
  paper's full architecture is the canonical next step before
  any claim against the AKOrN line is published.
- **Larger n.** With only 5 seeds the paired Wilcoxon p hits
  its floor at 0.0625; replicating at n = 10 or n = 20 would
  give the `gtm vs *` rows headroom to reach p < 0.05 even
  under FDR correction.
