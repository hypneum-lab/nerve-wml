# ADR 0002 — Novelty framing vs prior art (relative representations / stitching)

> Status: accepted, 2026-06-04. Answers REVIEW.md item ① ("Novelty — direct prior art":
> Relative Representations / zero-shot model stitching, Moschella et al. 2022
> (arXiv:2209.15430); Platonic-vs-Aristotelian debate, arXiv:2602.14486).
> Companion to ADR-0001 (which answers item ②, the MI/entropy estimator-bias attack).
> Paper edits: `papers/paper1/main.tex` §Related Work (§74, §80 `sec:contribution-delim`)
> and §Threats Test (10).

## Context

The external review attacks the novelty of the substrate-agnosticism claim on prior-art
grounds:

1. **Relative Representations / zero-shot stitching** (Moschella et al. 2022,
   arXiv:2209.15430): independently trained nets converge to latent spaces differing only
   by an angle-preserving transform; cross-architecture/modality stitching works *without*
   training. Cross-substrate representational convergence was therefore established by
   2022--2023 — *before* this work.
2. **Platonic-vs-Aristotelian debate** (arXiv:2602.14486): if representational alignment is
   training-dependent (shared task + data + objective), a low polymorphism gap and a high
   MI/H are *not* evidence of a *universal* protocol. Our MLP/LIF/Transformer substrates
   share task, data, and the Protocol objective by construction, so a degree of alignment
   is expected.

## What was already in the paper (pre-existing acknowledgements)

- Moschella relative representations cited with a quantified stitching example, and their
  VQ-anchor "future work" note (§74).
- Kornblith CKA, Morcos PWCCA, Saxe RDMs, Huh PRH all cited (§74); `aristotelianprh2026`
  cited against rhetorical overextension of PRH (§74).
- §76 already frames the discrete-protocol-output measurement as "the slot this family
  leaves open."
- Test (10) frozen-encoder control (§Threats) already quantifies that a non-trivial
  fraction of Test (1) alignment is explained by shared task + capacity, not the VQ
  protocol alone (shared-encoder floor 0.949; distinct-encoder floor 0.762).

## Decision

We do **not** re-run any experiment — the empirical controls (Test 10, ADR-0001) already
exist. We **requalify the framing** to be honest about prior art, in lockstep across three
sites in `main.tex`:

1. **§74 (Related Work):** explicit concession that cross-architecture representational
   convergence is *not novel to this work* and was established by Moschella et al. /
   stitching in 2022--2023; our contribution is its measurement at the discrete
   protocol-output level, not the observation of convergence.
2. **§80 → `sec:contribution-delim` (new "Delimiting the contribution" paragraph):**
   explicit *established (not claimed)* vs *new here* split. New = (a) discrete
   protocol-output measurement, (b) structurally heterogeneous biologically-inspired
   substrate pair (MLP vs spiking LIF vs Transformer), (c) integration into one protocol.
3. **§Threats, Test (10):** "Platonic-vs-Aristotelian caveat" linking the frozen-encoder
   floors to the Aristotelian critique — a low gap / high MI is bounded by shared
   task+data+objective and does *not* demonstrate a task-independent universal code.

## Consequences

- The headline number is unchanged; only the *interpretive scope* of Claim B is narrowed
  and the prior-art lineage is made explicit.
- No fabricated results: all cited references (`moschella2022relative`,
  `huh2024platonic`, `aristotelianprh2026`) were already present in `refs.bib`.
- Paper integration tests under `tests/` (`-k paper`) must still pass after the edits.
