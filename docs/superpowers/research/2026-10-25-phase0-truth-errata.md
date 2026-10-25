# Errata: phase 0 "truth" corrections (issues #64, #71)

Date: 2026-10-25. Scope: documentation only (milestone M0 "Vérité", PI
decision of 2026-10-25). No experiment was rerun, no JSON artefact was
modified, no tag was touched. Source: Hypneum Lab hypothesis review of
2026-10-24, revised 2026-10-25 (section 2.2, section 7.1, annex 12
point M2).

## Erratum 1 — N-3 ablations do not establish "type-checker, not differential mechanism" (issue #71)

### Facts

- `docs/research-notes/n3-gate-role.md` (2026-04-21) and Paper 1
  (`papers/paper1/main.tex`, sections Method and Threats to Validity)
  concluded from three ablations with Δ = 0 that N-3 is "a formal
  correctness contract — a type-checker for the protocol — rather than a
  differential mechanism".
- `papers/paper1/figures/ablation_n3_guard.json`: `n_received: 0` in all
  30 cells (`listen()` returned nothing); the guard itself is exercised:
  `n_caught: 70` at v = 0.05, seed 0 (`n_caught_mean` 57.0 / 99.7 /
  241.0 / 492.3 for v = 0.05 / 0.10 / 0.25 / 0.50, expected 50 / 100 /
  250 / 500), and `scripts/ablation_n3_guard.py:80-87` raises on
  violations at `nerve.send()`.
- `papers/paper1/figures/ablation_n3_predictive.json`:
  `apply_consolidation: false` in all six cells, with
  `apply_err: "'MockNerve' object has no attribute '_transducers'"`. The
  200 ε-letters per run were collected, but the consolidation step never
  ran.
- `scripts/ablation_n3.py` ran the canonical pipeline with the θ phase
  inactive: no ε-letter was ever emitted.
- Consequence: in every ablation the mechanism that `strict_n3` could
  have influenced was inactive. Δ = 0 is uninformative; it is neither
  evidence for nor against a differential role.

### Corrected status

- Established: the typing function of N-3 (the guard is exercised and
  catches malformed letters at send time).
- Not established by these ablations: any causal utility of N-3 for
  communication or learning.
- Not "untested" tout court: the typing result is real and is kept.
- A differential test requires an ablation with `n_received > 0` and
  active consolidation, pre-registered (L1) with a matched null.

### Places corrected

- `papers/paper1/main.tex` section Method (N-3 paragraph) and section
  Threats to Validity ("γ/θ empirical stress", dated erratum sentence).
- `docs/research-notes/n3-gate-role.md` (title, status, results of
  ablations 2 and 3, conclusion, erratum block).
- `README.md` (dated erratum section) and `CHANGELOG.md` (Corrections).

### Left unchanged

- The three JSON files and the three scripts (historical records; the
  print verdict in `scripts/ablation_n3_predictive.py:210` still says
  "formal type-checker only" and should be reworded when the script is
  next touched).
- `docs/changelog/v1.7.0.md` and `CHANGELOG.md` v1.7.0 entries (history).
- The programme-level master table (annex 01 of the review, repository
  `hypneum-lab/.github`) is outside this repository.

## Erratum 2 — stale facts in Paper 1 and metadata (issue #64)

### 2a. Direction stability "15/15" → 19/20

- The abstract already reports 19/20 (v1.7.0 recount: 4/5 at N=2, 5/5
  at N=16/32/64), but two passages of `main.tex` still said 15/15: the
  caption of the scaling figure and the Sleep-EDF paragraph ("$15/15$-seed
  HardFlowProxyTask finding"). Both aligned on 19/20.
- `scripts/render_scaling_figure.py` titled the figure "LIF ≥ MLP in
  15/15"; string corrected. The committed `papers/paper1/figures/
  w2_hard_scaling.pdf` and `.png` still carry the old title and must be
  re-rendered (`uv run python scripts/render_scaling_figure.py`) — not
  done here (documentation-only pass).
- `tests/integration/track_w/test_bigger_arch.py` docstring aligned.
- Historical mentions in `CHANGELOG.md` (v1.1.0) and
  `scripts/track_w_pilot.py` (review narrative) are left as history.

### 2b. Scaling table N=2 row

- The table in section Experiments listed `N=2 (1 seed) 10.71 % /
  10.71 % / 1/1`. The 5-seed rerun exists
  (`papers/paper1/figures/w2_hard_n2_multiseed.json`): median gap
  0.10714, max gap 0.16912, LIF ≥ MLP in 4/5 seeds (seed 4: MLP 0.660 >
  LIF 0.617). Row updated to `N=2 (5 seeds) 10.71 % / 16.91 % / 4/5`.

### 2c. "All experiments use N=4"

- Section Experiments ("Scale." paragraph) and section Limitations said
  all experiments use N=4 WMLs and that N=16/32 was an open question.
  The polymorphism law was measured at N ∈ {16, 32, 64} (5 seeds each)
  and the triple-substrate pool at N ∈ {15, 30, 60}. Both passages now
  say so; N=4 remains the Gate-W pilot size.

### 2d. Transformer "not measured"

- Section Information Transmission, Test (5), said "Pool-scale multi-seed
  for TRF is future work" while the abstract and README report the
  triple-substrate pool (v1.1.4, 5 seeds per scale, 8.2 % → 4.3 %). The
  sentence now states the measurement.
- Caveat recorded in the paper: unlike the pairwise law, no per-seed JSON
  of the triple-substrate pool run is archived under
  `papers/paper1/figures/`; the pinned regression is
  `tests/integration/track_w/test_triple_pool_hard_multiseed_distribution`.
  Archiving that JSON (or re-running the 5 seeds) is left to the PI.

### 2e. Loihi 2 / Lava

- `neuromorphic/loihi_stub.py`, `docs/neuromorphic/deployment-guide.md`,
  `README.md` and Paper 1 (Gate Neuro, Limitations) presented Intel
  Loihi 2 via the `lava-nc` SDK as a deployment path.
- GitHub API, checked 2026-10-25: `lava-nc/lava` and `lava-nc/lava-dl`
  both `archived: true`, last push 2026-05-13 (the archive date itself
  is not exposed by the API; 2026-05-13 is a last-push date). nerve-wml
  never compiled anything for Loihi 2 and has no Loihi 2 access.
- Requalified everywhere as a historical interface: the stub still
  raises `NotImplementedError` (message now states the archival; new
  unit test `test_loihi_stub_states_lava_archived`), the guide keeps the
  old recipe in a collapsed "historical, unsupported" block, and no
  future target is chosen here (SpiNNaker2 / NIR are PI decisions).

### 2f. Bylines (single-PI policy)

- `.zenodo.json`: creator "nerve-wml project contributors" removed.
- `CITATION.cff`: `preferred-citation.authors` entry "dreamOfkiki project
  contributors" removed.
- `CONTRIBUTORS.md`: byline policy rewritten to the single byline
  "Clément Saillant".
- `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`: "and
  nerve-wml contributors" removed from the author line.
- `papers/paper1/main.tex` already had a single author; unchanged.

## Handling

- Zenodo: no record exists for v1.9.x (see the 2026-10-24 L1a erratum);
  these corrections go into the description of the next published
  release together with the L1a one.
- Historical JSON files, tags (`v1.9.*`, `n10c-freeze-20261024`) and the
  N10-C manifests are not modified.
- Root cause (both errata): documentation written ahead of, or left
  behind, the committed results. Mitigation already in place for
  numbers (`scripts/factcheck_audit.py --ci`); qualitative conclusions
  such as the N-3 reading are not covered by that audit.
