# Equal-tuning protocol

## Motivation

Cross-method comparisons (learned Transducer vs. vec2vec vs. relative
representations vs. Procrustes) are only credible when every tunable
arm has had a comparable opportunity to find a good hyperparameter
configuration. Otherwise the result reflects the experimenter's prior
beliefs about which method "deserves" more tuning effort.

## Protocol

1. **Equal trial budget.** Every tunable method gets the same number
   of trials (`budget`, default 8) drawn deterministically from a
   fixed, declared grid. The grid is published in
   `scripts/equal_tuning_pilot.py` (`LEARNED_GRID`, `VEC2VEC_GRID`,
   `RELREP_GRID`).
2. **Equal seed budget per trial.** Each trial is run under
   `seeds_per_trial` independent seeds (default `(0, 1, 2)`). MI is
   averaged across seeds; a non-parametric bootstrap 95% CI is
   computed from the per-seed values.
3. **Best-of-budget selection.** For each method we report the trial
   with the highest mean MI (best-of-budget). This matches the
   convention used in HP-search papers (Lucic et al. 2018,
   Henderson et al. 2018) and avoids the "first-trial-wins" bias.
4. **Procrustes parity.** Procrustes is closed-form and has no
   hyperparameters. It is reported once under the same
   multi-seed budget (`seeds_per_trial`), so its CI is computed on
   the same number of seeds as a single trial of a tunable method.
   This is documented in the output as the `procrustes.note` field.
5. **No method-specific budget inflation.** If reviewers request a
   larger budget for one method, the same larger budget MUST be
   applied to every tunable method.

## Output shape

```python
{
  "learned":   {"trials": [...8 trials...], "best": {...}},
  "vec2vec":   {"trials": [...8 trials...], "best": {...}},
  "relrep":    {"trials": [...8 trials...], "best": {...}},
  "procrustes":{"trials": [1 trial],        "best": {...}, "note": "..."},
}
```

Each trial dict has `params`, `mi_values` (one per seed), `mi_mean`,
`mi_ci95_low`, `mi_ci95_high`.

## Reporting checklist

When citing equal-tuning results in a paper or report:

- [ ] State the budget and the seed-count per trial.
- [ ] Publish the full grid (not just the winner).
- [ ] Report mean MI ± bootstrap 95% CI for the best trial of each
      method.
- [ ] Disclose that Procrustes is non-tunable and reported at parity.
- [ ] Disclose the limitations below (in this file, this pilot's HPs
      are not all wired through the underlying runners).

## Limitations (as of Plan A.3, 2026-05-20)

All three tunable axes (`lr`, `lambda_cycle`, `n_anchors`) are now
threaded through the underlying training calls. Trial variation
therefore reflects real hyperparameter effect plus seed noise.

Open knobs still hard-coded inside the runner (left as future work):
- Vec2Vec internal generator/discriminator widths (`hidden`).
- Learned transducer Gumbel softmax `tau`.
- GTM PSK/PAM modulation choice.
