# External reviewer critique — nerve-wml (Paper v0.9)

> Status: external review (deep-research harness), 2026-06-03.
> CAVEAT: the harness adversarial-verification step FAILED for this corpus (verdicts not
> returned); a gateway re-verification with a small local model was unreliable (it
> "refuted" textbook-true results). The claims below are from CANONICAL sources and are
> judged credible on source quality, but are NOT independently harness-verified.
> Confirm with opus or a human reviewer.

## Verdict
The harshest of the four reviews: the central claim — substrate-agnosticism evidenced by
**MI(c_MLP;c_LIF)/H = 0.91–0.96** and a small polymorphism gap — is attacked on two
independent fronts: it is partly **prior art**, and the headline number is plausibly an
**estimator artefact**.

## ① Novelty — direct prior art
- **Relative Representations / zero-shot model stitching** (Moschella et al. 2022,
  arXiv 2209.15430): independently-trained nets converge to latent spaces differing only
  by an angle-preserving transform; cross-architecture/modality stitching works WITHOUT
  training. Cross-substrate representational convergence was established by 2023.
- Platonic-vs-Aristotelian debate (arXiv 2602.14486): if alignment is training-dependent,
  a low gap is not evidence of a universal protocol.

## ② Soundness — the strongest attack
- **MI/entropy estimator bias** (Paninski 2003): plug-in estimators are biased; the bias
  does not vanish in the undersampled regime (N/m bounded); CIs **drastically
  underestimate** error → the 0.91–0.96 seed-to-seed stability is a **false sense of
  precision**. Inconsistency theorem: a stable MI is not a correct MI.
- **Plug-in MI positive bias** (Panzeri 2007): a true-MI=0 neuron yields a plug-in
  estimate of 0.202 bit at N=20 → **an inflated MI is the NULL expectation**, not evidence
  of alignment. Impugns any uncorrected MI(c_MLP;c_LIF).
- CKA/similarity metrics are biased toward false-positive alignment, inflated by high
  dimensionality, detectable only via **permutation testing** (arXiv 2602.14486).
- Representations align mainly under same data + objective → MLP/LIF/Transformer share
  task + data + Protocol objective → alignment **expected by construction**.

## ③ Threats to validity
- Is "substrate-agnosticism" falsifiable, or is high-MI/low-gap expected by construction?
  Needs a **permutation null-model + a bias-corrected MI estimator (Panzeri/NSB)** to be
  falsifiable.
- The decoder-asymmetry bug (12.1% → fixed) shows the metric is sensitive to decoder
  details — what else is artefactual? Task-loss stitching is known to manufacture apparent
  equivalence (arXiv 2412.11299).

## Recommendation
Before §7.4 of the flagship cites this as independent corroboration, **re-measure MI with
bias correction + a permutation null-model**, or the headline number may be the null
expectation.
