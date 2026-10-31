# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. Open "TBD by PI" items (TBD-10, TBD-11) block the freeze (hypneum-prereg).
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. PI decisions 2026-10-31 (multiple-choice, 22:30-22:35 CET)

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived from sigma_seed by the formula of section 4 (margin 0.30, alpha 0.05, power 0.9, minimum 5) |
| TBD-5 | CLOSED. D_sym must not exceed the P99 of its calibrated null, with sigma_seed measured on development seeds (no fixed 0.05) |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot |
| TBD-10 | OPEN. Backend-divergence rule (section 7): proposal "verdict per backend, global conclusion only if both agree" |
| TBD-11 | OPEN. Reduced pilot step budget and n_eval; exact trunk adaptation per backend |

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

Confirmatory (after freeze only), seeds 600+: one-sided paired sign test and paired
bootstrap (10 000 resamples, percentile) of D on k seeds. k is fixed in the manifest by
a formula written in advance (`required_seeds`, `experiments/n10d/sigma_seed.py`):
k = max(5, ceil(((z_0.95 + z_0.90) * sigma_D / 0.30)^2)), sigma_D^2 = 2 (sigma_seed^2 +
0.25 / n_eval) (worst-case binomial term), normal approximation, margin 0.30, alpha 0.05,
power 0.9. sigma_seed comes from the pilot (8 development seeds 9600-9607, asym,
`nocomm` arm only). Pilot: nothing else is read, no NH-96 score; the pilot seeds are
never pooled with confirmatory runs. Non-rejection is "non concluant", never "refuted".

## 5. Outcomes and consequences (case C closed by TBD-6)

A. D_asym >= 0.30 and D_sym <= P99 of its calibrated null, controls pass: necessity confirmed on this task.
B. D_asym between 0.10 and 0.30: weaker than predicted, report with CI.
C. D_asym < 0.10: failed prediction. "The architecture blocks" is a new hypothesis,
   reserved to a separate amendment (e.g. a channel wired into the policy input)
   filed before any new run.
D. D_sym > P99 of its null: the symmetric control is not symmetric; the
   contrast is uninterpretable.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, ledger code. Before the freeze: TBD-10/11,
sigma_seed, k, freeze manifest.

## 7. Two backends (TBD-1, TBD-2)

Same task (K=16, C=4, A=16, conditions asym/sym, arms of section 3), same learner (the
N10-C PPO trunk, adapted per backend; same hyperparameters wherever the interface
allows, each difference listed in the manifest), same metric and same seeds list.
JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6, reviewed) in
PyTorch; neither is pooled with the other. Each backend gets its own null calibration
and its own sigma_seed. Each (backend, arm, seed) cell stays on one host.
Divergence rule (TBD-10, proposal): verdict per backend; a global conclusion is drawn
only if both agree. If they disagree, the global status is "non concluant", the cause
is a separate hypothesis for an amendment, and no result is selected after the fact.
