# ADR-0003: reading-gate exception for P1bis and P8c (2026-11-06)

> Status: **Accepted — PI decision 2026-11-06** 17:12 CET (QCM, option
> "ADR d'exception daté"). Drafted by an agent; signed by the PI
> (section 8). Studies: `hopfield-p1bis-ext`
> (`docs/prereg/hopfield-p1bis-ext/prereg.md`) and `hopfield-p8c-nocap`
> (`docs/prereg/hopfield-p8c-nocap/prereg.md`). Gate source: hypneum-lab/.github
> `docs/ROADMAP.md` principle 6 and
> `docs/superpowers/specs/2026-11-01-verrou-lecture-plan-trimestre-design.md`
> §4 (cited below at .github commit `c8967a2`).
>
> This ADR was drafted from process records only (tags, PRs, logs, file
> listings). No score, verdict file or results report of either study was
> opened to write it.

## 1. Context

### 1.1 What the gate requires

- `.github` `docs/ROADMAP.md:14` (principle 6, "Lire avant de geler"): no
  freeze tag without a reading note, a contradictor and three passed tests
  (saturation, cross-checking, coverage; spec §4.2). The gate applies to any
  heavy campaign (GPU, or > 6 h CPU, or > 1 calendar week) and to any L3
  prereg. "Les campagnes déjà gelées (E1-small, N10-C run 2, Me6, GB300 g7)
  ne sont pas re-soumises."
- Spec §4.1 (lines 40-44): the three parts are (1) a reading note next to
  the prereg, (2) a blind contradictor whose report is archived with the
  note, (3) several dated watch rounds on different sources. Spec §4.2
  (lines 46-56): freeze is allowed only when the three tests pass and the
  contradictor has failed. Spec §4.3 (line 60): same scope and the same
  closed exemption list as the roadmap; a short CPU pilot is exempt but
  cannot be cited as a result.
- `.github` `docs/ROADMAP.md:198` (KPI table): "Tags de gel posés sans
  verrou | 0".

### 1.2 Chronology (CET unless marked Z)

| When | Event | Record |
|---|---|---|
| 2026-11-01 14:47-21:36Z | GB300 pilots P1bis (`cpu-p1bis-hopfield-fracbudget`) and P8c (`gb300-p8c-dream-kernel`), exploratory | P1bis prereg.md:246, P8c prereg.md:265 |
| 2026-11-01 21:03 | Gate design spec committed | .github `111f3b7` |
| 2026-11-01 21:32 | Roadmap v2 "under the reading gate": principle 6 and KPI = 0 in force | .github `8014889` (ROADMAP.md:14, :172 at that commit) |
| 2026-11-04 18:16-18:20:58 | PR #106 "prereg: Hopfield P1bis-ext and P8c-nocap (L0)" opened and merged | merge `4c7ab1b` |
| 2026-11-04 18:22:10 | Annotated tag `prereg-hopfield-p1bis-p8c-freeze-20261104` (object `2528028`, target `257e469`) | freeze-manifest.md:66-72 |
| 2026-11-04 17:22:57Z | Campaign START on kxkm-ai, 47 s after the tag; lanes started 17:25:10Z | `kxkm-ai:/ai/saisail/p1bis-p8c-20261104/logs/campaign.log` |
| 2026-11-04 18:25:18Z | P1bis lane done (rc=0) | campaign.log |
| 2026-11-04 21:37:31Z | P8c N=4096 lane done (rc=0) | campaign.log; amendment-2026-11-05.md:37-38 |
| 2026-11-05 05:18:34Z | P1bis frozen analysis run | `analysis-p1bis/analyse.log` |
| 2026-11-05 19:05:21Z | PI stops the N=16384 arm (18 h HU ceiling too short) | amendment-2026-11-05.md:12-16 |
| 2026-11-05 19:12:45Z | P8c N=4096 frozen analysis run (scores read) | `analysis-p8c/analyse.log`; amendment-2026-11-05b.md:71-72 |
| 2026-11-05 19:19:09Z | PR #111 "docs(p8c): amend prereg, stop N=16384 arm" merged | merge `f0383bd`, branch commit `084b840` |
| 2026-11-05 19:33:59Z | Annotated tag `prereg-hopfield-p8c-relaunch-20261105` (object `b59a141`, target `db4eb5b`, not yet merged) | amendment-2026-11-05b.md:79-86 |
| 2026-11-05 19:34:32Z | Relaunch of N=16384, alpha=0.5, 20 seeds, 8 workers, HU ceiling 36 h, wall cap 120 h | campaign.log; `RELAUNCH.md`; `results/hopfield_p8c_nocap/n16384/run_info.json` |
| 2026-11-06 06:06:20Z | PR #112 "docs(prereg): amend P8c N=16384 relaunch" merged | merge `0aeae44` |
| 2026-11-06 17:12 | PI decision: dated exception ADR | this ADR |

Both freeze tags were therefore posed **after** the gate came into force
(2026-11-01 21:32) and neither study is in the closed exemption list
(E1-small, N10-C run 2, Me6, GB300 g7). Neither study appears anywhere in
the `.github` roadmap or spec (no match for "P1bis", "P8c" or "Hopfield"
in `docs/ROADMAP.md` or the spec at `c8967a2`).

### 1.3 Scope of the gate, per study

- **P8c-nocap** is a heavy campaign on any reading of the threshold: the
  frozen plan budgets ≈ 250 core-hours and 36-42 h wall for N=16384 alone
  (P8c prereg.md:303-306, wall cap 72 h at :310); the relaunch sets a
  36 h HU ceiling per cell and a 120 h wall cap on the kxkm-ai CPU
  (amendment-2026-11-05b.md:30-39).
- **P1bis-ext** on its own is below "> 6 h CPU": ≈ 7 800 core-s, 15-25 min
  wall, 6 h cap (P1bis prereg.md:277-282; actual lane 1 h 00 min,
  campaign.log). It is covered here because principle 6 opens with "no
  freeze tag without..." and P1bis was frozen by the same tag and manifest
  as P8c (freeze-manifest.md:3-11). This ADR does not decide whether P1bis
  alone needed the gate; it records that its tag was posed without one.

### 1.4 What was and was not done

- Reading note (spec §4.1-1): **none** for either study.
- Blind contradictor (spec §4.1-2): **none**.
- Dated watch rounds (spec §4.1-3) and the three tests of spec §4.2:
  **none**.
- What exists instead, and is not a gate piece: a literature paragraph in
  each prereg, written by the authors (P1bis prereg.md:30-41 cites Benedetti
  et al. 2022, Benedetti et al. 2023, Marinari-Rossi-Zamponi and declares
  the effect "not presented as new"; scoop risk acknowledged at :264-265);
  an independent prereg review "ACCEPT, four minor documentation points"
  (freeze-manifest.md:19-20; PR #106 body). The review checked the protocol,
  not the literature, and saw the drafts and their framing, so it is not
  blind in the sense of spec §4.1-2.

## 2. Decision

1. The runs continue. The N=16384 relaunch of P8c-nocap is not stopped and
   no frozen parameter changes because of this ADR.
2. Both preregs are labelled **"frozen prereg, gate not passed"**. This
   label goes with every result, table and verdict of either study.
3. This ADR is the record of the exception. It is dated, it names the two
   studies and the two tags, and it is not a precedent.
4. The KPI "Tags de gel posés sans verrou" counts **2 for S2** (2-8 Nov
   2026), not 0. Both readings give 2: two studies, and two tags
   (`prereg-hopfield-p1bis-p8c-freeze-20261104`,
   `prereg-hopfield-p8c-relaunch-20261105`).
5. No new freeze follows this precedent. Any later study, including a
   follow-up of P1bis or P8c, passes the gate before its tag.

## 3. Scope: what the results cannot claim

- No confirmatory language. A verdict reached under the frozen analysis
  plan is reported as "frozen prereg, gate not passed", never as
  "confirmed", "established" or "replicated".
- No "first", "novel", "necessary" or "unique to dreaming" claim without a
  **post-hoc** reading note and a blind contradictor that fails.
- Any use in a paper, preprint, talk or README requires the gate to be
  passed retroactively (reading note + contradictor + the three tests of
  spec §4.2), and the result keeps the explicit label "post-hoc gate".
  A retroactive gate cannot turn the study into a gated one.
- The frozen analysis rules, gates and verdict branches of both preregs
  are unchanged; this ADR adds a reporting constraint, not a new test.
- The stop and relaunch amendments stand as signed: PR #111
  (`amendment-2026-11-05.md`) and PR #112 (`amendment-2026-11-05b.md`),
  including the declared deviation of 2026-11-05b §8 (run started from a
  tagged, unmerged commit) and its blindness statement §7 (not blind
  w.r.t. N=4096, blind w.r.t. N=16384).

## 4. Consequences

### 4.1 Line to add to each prereg (by signed amendment only)

The frozen `prereg.md` files are hashed in the manifest
(freeze-manifest.md:34-35) and are not edited in place; the line below is
carried by a signed amendment for each study, in the same way
`amendment-2026-11-05.md` §7 recorded the stale "NOT frozen" header
without touching the file. This ADR does not write those amendments.

- `docs/prereg/hopfield-p1bis-ext/prereg.md` (status block, after line 8):

  > **Reading gate:** not passed. Frozen on 2026-11-04 after the reading
  > gate came into force (.github ROADMAP principle 6), without reading
  > note, contradictor or watch round. Results are reported as "frozen
  > prereg, gate not passed". See `docs/adr/0003-reading-gate-exception-p1bis-p8c.md`.

- `docs/prereg/hopfield-p8c-nocap/prereg.md` (status block, after line 7):

  > **Reading gate:** not passed. Frozen on 2026-11-04 and relaunched on
  > 2026-11-05 after the reading gate came into force (.github ROADMAP
  > principle 6), without reading note, contradictor or watch round.
  > Results are reported as "frozen prereg, gate not passed". See
  > `docs/adr/0003-reading-gate-exception-p1bis-p8c.md`.

### 4.2 Elsewhere

- Roadmap KPI "Tags de gel posés sans verrou" to be updated to 2 (S2) at
  the next resume-point session, in hypneum-lab/.github, not in this
  repository.
- Results notes `docs/results/<date>-hopfield-p1bis-ext.md` and
  `docs/results/<date>-hopfield-p8c-nocap.md` carry the label of section 2
  in their first paragraph.
- Cost of a post-hoc gate, if the PI wants either study in a paper: the
  plan budgets 25 h for the N10-D gate (spec §5.2, line 75) and 19 h in
  the roadmap v2.1 carry-over (ROADMAP.md:97). Order of magnitude: 19-25 h
  of PI time per study, less if both share one reading note on Hopfield
  unlearning. This is not budgeted in the quarter.

## 5. Alternatives considered

- **Stop P8c with a signed deviation.** Not retained (PI choice): it would
  discard about 21 h of a relaunch that is blind w.r.t. N=16384, and the
  gate's missing pieces do not bear on the frozen protocol itself.
- **Leave the runs going with a mention only.** Not retained (PI choice):
  a mention leaves no dated record, no claim limits and no KPI correction.

## 6. State of the run at drafting (process fields only)

Read on kxkm-ai on 2026-11-06 17:20 CET: the relaunch processes (wrapper
2153161, wall guard 2153166, pool parent 2153167, 8 workers) have been up
20 h 45 min; `results/hopfield_p8c_nocap/n16384/cells.jsonl` has 0 lines
(wave 1 of 8 cells not finished). Per `RELAUNCH.md`, expected end between
about 2026-11-08 and 2026-11-10, wall cap 2026-11-10T19:34Z. No score was
read.

## 7. References

- hypneum-lab/.github `docs/ROADMAP.md` v2.1 (commit `c8967a2`), lines 14,
  97, 198; v2 at `8014889`.
- hypneum-lab/.github
  `docs/superpowers/specs/2026-11-01-verrou-lecture-plan-trimestre-design.md`
  §4.1-4.3, §5.2 (commit `111f3b7`).
- nerve-wml PR #106 (merge `4c7ab1b`), PR #111 (merge `f0383bd`), PR #112
  (merge `0aeae44`); tags `prereg-hopfield-p1bis-p8c-freeze-20261104`
  (target `257e4698cadd6786d2381cf61d2eef4cdbeb4475`) and
  `prereg-hopfield-p8c-relaunch-20261105` (target
  `db4eb5ba5644df8dc5622f53f249c19d52a9d82b`).

## 8. PI signature

- [x] PI signature (Clément Saillant)
- Date and time (CET): 2026-11-06 21:13 CET

Signed: Clément Saillant (PI), 2026-11-06 21:13 CET
