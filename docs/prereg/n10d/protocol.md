# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. Open "TBD by PI" items block the freeze (hypneum-prereg).
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. Open PI decisions (each blocks the freeze)

| id | decision | proposal in this draft |
|---|---|---|
| TBD-1 | Backend: JaxMARL (JAX, `mlx` not used) or EGG (PyTorch), or both | EGG Lewis game first (CPU, cheap), JaxMARL second |
| TBD-2 | Task family and sizes K (targets), C (candidates), A (alphabet) | K=16, C=4, A=16 (placeholders) |
| TBD-3 | Learner: reuse the N10-C PPO trunk, or the backend's native learner | native learner (no coupling to Procgen) |
| TBD-4 | Seed namespaces: pilot, confirmatory; seed count k | pilot 92xx, confirmatory 93xx, k=8 |
| TBD-5 | Primary margin: prediction D_asym >= 0.30 and D_sym <= 0.05 (NH-96) | as NH-96, see section 4 for the problem with 0.05 |
| TBD-6 | Meaning of the architecture escape (section 5, case C) | stop N10-D, no verdict on necessity |
| TBD-7 | Host and budget (GB300 pilot vs KXKM-AI) | decided after the gate result; no GPU needed for the L0 |
| TBD-8 | Dependency pins (see `requirements-to-hash.md`), HITL diff review | candidates only |
| TBD-9 | Pilot budget used to measure `sigma_seed` of the null (section 4) | development seeds only |

## 1. Question

Does a message become necessary and used when, by construction, the listener cannot
succeed without information that only the speaker holds? Equivalently: is the
`eng - nocomm` contrast large on an asymmetric task and null on its symmetric twin?
This is a test of the task and of the channel plumbing, not of emergent-language
theory: NH-96 is labelled VARIANT (expected result, Lowe 2019).

## 2. Link with the gate

- Gate passes (`eng - nocomm >= 0.20` on N10-C): N10-D is not needed for necessity;
  it stays an optional follow-up. Nothing here is run.
- Gate fails: N10-C-bis is not frozen (G3); N10-D is the next study. The gate failure
  is read as "the Procgen task does not need the message", which N10-D tests on a task
  where it must. A gate failure is a design-decision input, not an N10-D result;
  gate runs are never pooled with N10-D runs.

## 3. Design

Conditions: `asym` (only the speaker sees the target; the listener sees C candidates
that contain it) and `sym` (both see the target). Arms (code: `experiments/n10d/spec.py`):

| arm | channel | role |
|---|---|---|
| `eng` | engineered sender, ideal code, fixed | upper reference, as in the gate |
| `nocomm` | zeroed | blind reference |
| `learned` | trained jointly, alphabet A | emergent-protocol arm (secondary) |
| `msg_shuffle` | message from another episode (fixed permutation) | negative control |

Metric (primary): team success = mean of the last 5 logged values of the training
success curve (same definition as gate G2); D = mean over seeds of per-seed
`S_eng - S_nocomm`, paired by seed. Secondary: `learned - nocomm`, `eng - msg_shuffle`.
Blind ceiling (exact, `blind_ceiling()`): asym 1/C, sym 1.0.

Controls (reported next to every result, same pipeline):
1. Positive: a scripted listener that reads the true target through the message must
   reach >= 0.95 on `asym`. If not, the pipeline cannot measure and no D is read.
2. Negative: `msg_shuffle` must fall to the blind ceiling within the null P99.
3. Learning gate: `eng` above `blind ceiling + P99 of the null` before D is interpreted
   (floor effects are "not measurable", never "null effect").
4. Leak audit: `nocomm` on `asym` above ceiling + P99 means the task is not
   asymmetric (side channel); the run set is invalid.

## 4. Analysis plan

Confirmatory (after freeze only): sign test and paired bootstrap (10 000 resamples,
percentile) of D on k seeds; with k=8 the smallest two-sided sign-test p is 2/256 = 0.0078, so
k=8 is the floor; paired-t power is to be computed once
`sigma_seed` is measured. Decision rules are in `nh96-null.md`. Non-rejection is
"non concluant", never "refuted". Pilots (listed, seeds 92xx) inform N and
`sigma_seed` only and are never pooled with confirmatory runs.

## 5. Outcomes and consequences (TBD-6 closes case C)

A. D_asym >= 0.30 and D_sym <= 0.05, controls pass: necessity confirmed on this task.
B. D_asym between 0.10 and 0.30: weaker than predicted, report with CI.
C. D_asym < 0.10: the channel is not used even where it is needed. This does not
   "confirm that the architecture blocks": it is a failure of NH-96's prediction
   and the architecture explanation is a new hypothesis, tested by a separate arm
   (channel wired into the policy input) in an amendment before any new run.
D. D_sym > 0.05 beyond the null P99: the symmetric control is not symmetric; the
   contrast is uninterpretable.

## 6. Not done here

No run, no install, no compute. Missing before a pilot: PI decisions above, a
backend choice and pinned hashed environment, the real env implementation
(`backends.py` raises), the learner/trainer, `sigma_seed` from development seeds,
the engineered sender and scripted listener, ledger/allowance code, freeze manifest.
