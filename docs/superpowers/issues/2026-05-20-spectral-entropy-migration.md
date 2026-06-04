# Migrate `_synchrony_index` (top-PC variance) to `spectral_entropy` in production code

## Problem

`_synchrony_index` in `track_p/multiplexer.py:_synchrony_index()` returns the
fraction of carrier variance carried by the top principal component of the
centered batch. The intended reading is "high score = collapse onto a single
mode".

Empirical evidence (Renf 7, 50 seeds × 4 arms × 3 hosts; see
`docs/superpowers/research/2026-05-20-synchrony-replacement.md` and the raw
data in `docs/superpowers/research/2026-05-20-synchrony-replacement.json`)
shows this metric is **anti-monotone** with respect to the intended semantics:

```
null = 0.32  >  gtm = 0.20  >  simple_gating = 0.08
                                 (paired Wilcoxon p < 1e-15)
```

A trained-but-broken model (null arm: GTM trained on shuffled supervision)
concentrates variance on a single mode and therefore scores **higher** than a
model that successfully decomposes the task. The semantics are inverted. The
permutation-shuffled null arm used to detect this inversion is a special case
of the formal null-calibration framework of Gröger, Wen & Brbić (2026,
arXiv:2602.14486), which we adopt as our cross-condition baseline methodology.

## Replacement

`spectral_entropy` from `scripts/synchrony_alternatives.py` (already committed
on `master` at SHA `076f770`). It is defined as
`H = -Σ p_i log p_i` over the normalised eigenvalues of the centered carrier
batch's Gram matrix.

*Note: `spectral_entropy = log(effective_rank)` is the standard
collapse-analysis quantity introduced by Roy & Vetterli (2007, EUSIPCO) and
reapplied to LLM representations by Wei et al. (2024, Diff-eRank,
arXiv:2410.10672). This is alignment with established practice, not a
methodological proposal.*

On the same 4-arm × 50-seed × 3-host ablation:

```
null (1.96 ± 0.13) < akorn_best (1.97 ± 0.61) < gtm (2.17 ± 0.01) < simple_gating (3.46 ± 0.02)
```

All consecutive gaps are significant at paired Wilcoxon `p = 1.78e-15`. The
ordering matches the intended "high = many effective modes, low = collapse"
reading.

## Scope of migration

- Replace `_synchrony_index` calls in `scripts/gtm_ablation_pilot.py` with
  `spectral_entropy`.
- Update the existing `_synchrony_index` function in
  `track_p/multiplexer.py` to either be removed or marked as `@deprecated`
  with a clear pointer to `spectral_entropy`.
- Update result dicts: `synchrony_index` → `spectral_entropy` (key rename,
  no compat shim — see Non-goals).
- Update tests that assert on `synchrony_index`
  (`grep -rn synchrony_index tests/`).
- Re-generate any research notes that quote the old metric.

## Acceptance criteria

- `_synchrony_index` removed or marked `@deprecated` with a docstring pointer
  to `spectral_entropy`.
- `scripts/gtm_ablation_pilot.py:run_gtm_ablation()` returns
  `spectral_entropy` per arm.
- All tests in `tests/integration/test_gtm_ablation_pilot.py` pass with the
  new metric.
- The synchrony figure in the paper (`papers/paper1`) uses
  `spectral_entropy` end-to-end (production code, not the side script).

## Non-goals

- Backward compatibility for the old `synchrony_index` field. The metric is
  inverted with respect to its documented semantics; preserving it under the
  same name would perpetuate the bug.

## Effort estimate

~1–2 hours. One PR, ≈50 LOC delta plus test updates.

## References

- Paper integration: `papers/paper1/main.tex` §Limitations and Future Work,
  paragraph "Carrier-spectrum metric — from `synchrony_index` to spectral
  entropy".
- Research note: `docs/superpowers/research/2026-05-20-synchrony-replacement.md`
- Raw data: `docs/superpowers/research/2026-05-20-synchrony-replacement.json`
- Replacement implementation: `scripts/synchrony_alternatives.py` (master
  SHA `076f770`).

## B-sensitivity caveat (added 2026-05-20 post-Renf 10)

The 4-arm strict ordering "null < akorn_best < gtm < simple_gating"
reported above (and in `2026-05-20-synchrony-replacement.md`) is
**B-dependent**. Renf 10 (50 seeds × 4 B values × 4 arms; see
`docs/superpowers/research/2026-05-20-renf10-batch-sensitivity.md` and
the raw data in
`docs/superpowers/research/2026-05-20-renf10-batch-sensitivity.json`)
finds the canonical ordering only at `B = 256`; at `B ∈ {64, 128, 512}`
the `akorn_best` arm falls **below** `null` due to its high
seed-variance (σ ≈ 0.6 vs `gtm`'s 0.01). The robust 2-arm claims
(`gtm > null`, `gtm < simple_gating`) hold at all four `B` values with
`p < 1e-15`. The migration of `_synchrony_index` → `spectral_entropy`
is unaffected; only the **framing of the 4-arm comparison** needs to be
revised to "`gtm > null` and `gtm < simple_gating`, `akorn_best`
intermediate but high-variance".
