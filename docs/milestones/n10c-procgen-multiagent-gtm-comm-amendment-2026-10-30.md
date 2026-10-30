# N10-C — Deviation declaration 2026-10-30: second blinding breach, freeze record and omitted covariance

**Status:** DRAFT — declaration, not frozen. It changes no registered provision of N10-C.
**Declares against:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (FROZEN by tag `n10c-freeze-20261029`), which amends `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (frozen by tag `n10c-freeze-20261024`) and the locked `docs/milestones/n10c-procgen-multiagent-gtm-comm-2026-05-11.md`; and the N10-C-bis addendum `docs/milestones/2026-10-25-n10c-bis-control-arms.md` (DRAFT, not frozen).
**Date drafted:** 2026-10-30 (dated by the GitHub push time of its first commit; git dates are not evidence of time).
**Investigator:** Clément Saillant (Hypneum Lab, sole investigator).

The frozen amendments and the locked pre-registration are not modified. This document is append-only. It only touches `docs/`, so the N10-C launch guard (HEAD a descendant of the tag commit with only `docs/` and manifest-folder changes) still holds on master.

## 1. Deviation: second blinding breach (the « 0.96 » reading)

**This breach is declared before any N10-C-bis tag.**

- **Rule broken.** The N10-C-bis addendum, section « Blinding during N10-C » (binding): until the N10-C-bis freeze tag is pushed, no held-out evaluation result and **no training-curve field (training success rate, time-to-threshold)** of any N10-C run is read by the PI or by any tool that summarises them. The rule applies to the runs of the new N10-C campaign (addendum, section A7.2).
- **What was read.** On 2026-10-30, a training-curve field of a registered run of the campaign of tag `n10c-freeze-20261029` was read and quoted during the PI's review of the N10-C-bis design: the **final training success 0.96 of `registered-gtm-s0`**. The exact time and channel of the reading are to be recorded by the PI next to this line.
- **First breach.** The first breach is the one of 2026-10-25 (held-out team success of `registered-eng-s73` and `registered-gtm-s0` of the stopped campaign of tag `n10c-freeze-20261024`; 2026-10-25 amendment, section 1; addendum, sections A7.2 and B2). This is the second.
- **Consequences.**
  1. The PI decision of 2026-10-30 at about 15:10 CET (necessity gate before the N10-C-bis freeze; on failure, an asymmetric N10-D; addendum, section « Necessity gate ») was taken **after** this reading. It is reported as a **non-blind design decision**. It is not, and may not be, justified by the figure read; its stated grounds are development records only (addendum, section G1).
  2. The hypotheses H-bis-1 to H-bis-3, arms and contrasts of the addendum are those of its first commit (`c910298`) and are not revised because of this reading.
  3. The N10-C registered analysis is unchanged: the run is not excluded, re-run or re-analysed because of the reading, and no reading of N10-C (reading rule R1–R5 of the addendum) is based on it.
  4. From now on, campaign monitoring is again restricted to run status, attempt counts, throughput and failure reasons.

## 2. Freeze record of tag `n10c-freeze-20261029`

Not recorded in the repository until now (the freeze plan, `docs/milestones/n10c-freeze-plan-2026-10-28.md`, step 8, asked for it). Checked on 2026-10-30 against git and the Software Heritage API.

| Item | Value |
|---|---|
| Tag | `n10c-freeze-20261029`, annotated, taggerdate 2026-10-29T06:25:08+01:00 (git date, not evidence of time) |
| Tag object | `b14717b7efbbee111b577debce981e9b58795d19` |
| Tagged commit (revision) | `99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` (merge of PR #89) |
| Manifest (tag message) | `experiments/procgen_multiagent_gtm_comm/manifest/stage2-manifest-fixc.md`, SHA-256 `68539232fb1d729703ecd5370aa2bcf8a024c408ef6decce5159ec80268d3640` |
| Software Heritage visit | origin `https://github.com/hypneum-lab/nerve-wml`, visit 2, 2026-10-29T10:55:51Z, status `full` |
| SWHID, snapshot | `swh:1:snp:01242c8b5873518a503b84a80ce3a302bb69b8a8` |
| SWHID, release (the tag) | `swh:1:rel:b14717b7efbbee111b577debce981e9b58795d19` |
| SWHID, revision (the tagged commit) | `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` |

**Correction, disclosed.** `b14717b…` is the **tag object**, not a commit. `swh:1:rev:b14717b7efbbee111b577debce981e9b58795d19` does **not** resolve (Software Heritage API: `Revision … not found`); the archive holds it as a **release** that targets revision `99a7cf4…`. The campaign guard accepts only `snp`, `rev` or `dir` SWHIDs (`campaign/guards.py`), so the identifier for `--swhid` is the revision SWHID `swh:1:rev:99a7cf4bb01d31e8e7f6c99c0e6676f41e6e17bb` (or the snapshot SWHID). Which SWHID the running campaign's `campaign.json` actually records is to be checked on both hosts (read-only: `jq .swhid`) and reported here; a well-formed but non-resolving SWHID would be a recording error of the launch, not a change of the frozen configuration.

## 3. Omitted covariance in the registered analysis (to be corrected)

- **Fact.** Since fix C, all arms and both conditions of one seed start from the **same pre-trained trunk** (2026-10-25 amendment, section 3.2: common random numbers across arms). Arm values of the same seed are therefore not independent.
- **Registered analysis.** The analysis is still « Welch / independent groups over the 6 seed values of each arm » (2026-10-23 amendment, section « Seeds and hosts », bullet « Analysis »), with arm streams described as independent (same amendment, « Evaluation protocol and randomness », « Independent arm streams »). It ignores the within-seed covariance induced by the shared trunk (and by the shared host of each seed). If that covariance is positive, the independent-groups variance of a difference is too large (intervals conservative); in any case it is not the model of the data as now generated.
- **Correction to adopt.** For within-campaign contrasts, an analysis **paired by seed** (per-seed difference of the two arms, one-sample t interval on the 6 differences, df 5, same adjustment levels and pilot categories).
- **Disposition.** The frozen pre-registration is **not rewritten**. Whether the paired analysis is reported next to the registered one, and which is primary, is an **open decision for the PI**, to be taken and recorded here **before any held-out N10-C result is read**. The N10-C-bis addendum has the same issue (its arms start from the N10-C seed trunks, section A2, while its estimation plan states that no pairing by seed is used); it is not frozen and can be corrected before its tag.
- Source of the observation: an independent read-only review (Codex), checked against the files cited above.

## 4. Open decisions for the PI

1. Time and channel of the reading of section 1.
2. Disposition of the covariance correction (section 3), before any held-out N10-C result is read.
3. Seeds of the necessity gate, if more than the default 9101 (addendum, G2), before its first run.
