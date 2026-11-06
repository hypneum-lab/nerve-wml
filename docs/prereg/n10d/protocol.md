# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Revision:** 2026-11-06 — revised after the blind contradictor pass (PR #113, VERIFICATION.md):
(R1) learning gate and `msg_shuffle` control restricted to `asym`, own gate for `sym` (section 3);
(R2) `sym` out of H1, case D becomes a leak report (sections 1, 5; PI decision 2026-11-01);
(R3) `msg_shuffle` is a derangement of episodes inside the batch, not a fixed permutation (section 3);
(R4) primary metric = frozen-policy evaluation on n_eval independent episodes, last-5-logs
secondary (section 3, TBD-11); (R5) a-priori audit of the task generator (section 3, control 4);
(R6) each threshold on one arm gets its own null quantile, not the P99 of D (section 3);
(R7) seed-count rule aligned on the test actually used, variance source of D stated, no
factor 2 for a direct sigma_D (section 4); (R8) case A reworded "plumbing and trunk learn the
engineered code", requires rejection of the confirmatory test and one decision rule (section 5);
(R9) case D reads "possible leak or optimisation effect; audit required" (section 5);
(R10) the reading "the Procgen task does not need the message" is conditioned (section 2);
(R11) per-backend statuses and combination table (section 7, TBD-10); (R12) JaxMARL game
specified, `simple_reference` excluded (section 7, TBD-11); (R13) new open items TBD-12 to TBD-16.

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. Open "TBD by PI" items (TBD-10 to TBD-16) block the freeze (hypneum-prereg).
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. PI decisions 2026-10-31 (multiple-choice, 22:30-22:35 CET), updated 2026-11-06

**Status of this table (2026-11-06):** nothing in this document is frozen. CLOSED means
decided by the PI for this draft, not registered; OPEN items TBD-10 to TBD-16 block the freeze.

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived by the rule of section 4 (alpha 0.05, power 0.9). Margin and floor under revision: TBD-12, TBD-13 |
| TBD-5 | SUPERSEDED 2026-11-06 by PI-1. D_sym is no longer in H1; it is reported as a leak check against its own paired null (section 3, control 5; section 5, case D) |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. Possible amendment: TBD-14 |
| PI-1 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 1). `sym` leaves H1; it stays a leak check reported next to the results |
| PI-2 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 2). The null is paired by seed (seed effect common to both arms), `nh96-null.md` section 2 |
| TBD-10 | OPEN. Backend-divergence rule (section 7): statuses and combination table proposed, to be confirmed |
| TBD-11 | OPEN. Reduced pilot step budget; n_eval of the frozen-policy evaluation and the log interval; exact trunk adaptation per backend; custom JaxMARL game and message length L (section 7) |
| TBD-12 | OPEN. Margin: absolute 0.30, or relative to the oracle contrast 1 - 1/C = 0.75 (e.g. 0.9 x 0.75 = 0.675)? With 0.30, a listener at 55 % passes (D = 0.55 - 0.25) |
| TBD-13 | OPEN. Seed-count rule (section 4): which test sizes k (sign test, paired mean, or k = max of both); assumed q = P(D_i > 0) for the sign test; floor MIN_K (now 5: at k = 5 only 5/5 rejects, power q^5 = 0.59 at q = 0.9) |
| TBD-14 | OPEN. Source of sigma_D for power. Amends TBD-9: may the pilot also extract the between-seed sd of S_eng on asym (eng records alone, D never computed, `sigma_seed.py <json> eng`)? If not, sigma_seed(nocomm/asym) is about 0 by construction (estimate 0 in 57 % of 8-seed pilots, P(chi2_7 < 7) = 0.571) and k rests on TBD-13. Same question for the sigma of the sym leak-check null (not transported from asym) |
| TBD-15 | OPEN. epsilon of the sym gate: `eng_sym >= 1 - epsilon` and `nocomm_sym >= 1 - epsilon` (section 3, control 3) |
| TBD-16 | OPEN. One decision rule for cases A/B (section 5) and the NH-96 rejection criteria (`nh96-null.md` section 4): (i) CI rule, A iff the lower bound of the 95 % paired-bootstrap CI of D_asym >= margin, rejection iff its upper bound < margin, in between "non concluant"; or (ii) point rule, A iff D_asym >= margin with the CI excluding 0, and the status of the zone "D_asym >= margin, lower bound < margin" stated. In both, A also requires the sign test to reject (section 5) |

## 1. Question

Does a message become necessary and used when, by construction, the listener cannot
succeed without information that only the speaker holds? Operationally: is the
`eng - nocomm` contrast large on the asymmetric task? The symmetric twin is not part of
the question (PI-1); it is a leak check. This is a test of the task and of the channel
plumbing, not of emergent-language theory: NH-96 is labelled VARIANT (expected result,
Lowe 2019). For optimal policies the contrast is fixed by construction (0.75 on `asym`,
0 on `sym`, `blind_ceiling()`); what a run measures is whether the plumbing and the
trained trunk reach it.

## 2. Link with the gate

- Gate passes (`eng - nocomm >= 0.20` on N10-C): N10-D is not needed for necessity;
  it stays an optional follow-up. Nothing here is run.
- Gate fails: N10-C-bis is not frozen (G3); N10-D is the next study. A gate failure is
  a design-decision input, not an N10-D result; gate runs are never pooled with N10-D runs.
  N10-D runs in another harness (JaxMARL and EGG, section 7). A gate failure is therefore
  not read as "the Procgen task does not need the message", and an N10-D result says
  nothing about the N10-C plumbing, unless an N10-D arm is also run inside the N10-C
  harness with the same channel code (not planned in this draft).

## 3. Design

Conditions: `asym` (only the speaker sees the target; the listener sees C candidates
that contain it) and `sym` (both see the target). Arms (code: `experiments/n10d/spec.py`):

| arm | channel | role |
|---|---|---|
| `eng` | engineered sender, ideal code, fixed | upper reference, as in the gate |
| `nocomm` | zeroed | blind reference |
| `learned` | trained jointly, alphabet A | emergent-protocol arm (secondary) |
| `msg_shuffle` | message of another episode of the same batch (derangement), at training and evaluation | negative control |

`msg_shuffle` permutation (`interfaces.episode_derangement`): a seeded fixed-point-free
permutation of the episodes of the current batch, so each listener receives the message
emitted for another episode; it keeps the marginal of messages and their length; identical
target labels across episodes are not excluded (a 1/K coincidence is not a leak). A fixed
permutation of target identities is not allowed: the message would stay a deterministic
function of the target and a listener could learn it up to 1.0.

Metric (primary): team success S of the policy frozen at the end of training, evaluated
on n_eval independent episodes (fresh targets and candidate orders, no learning; n_eval
in TBD-11), so that S x n_eval is binomial under the null. D = mean over seeds of
per-seed `S_eng - S_nocomm`, paired by seed. The training-curve summary of gate G2 (mean
of the last 5 logged values, each over the last 100 episodes) is reported as secondary
only: its windows overlap and are not stationary, so it is not a binomial of n_eval.
Secondary contrasts: `learned - nocomm`, `eng - msg_shuffle`.
Blind ceiling (exact, `blind_ceiling()`): asym 1/C, sym 1.0.

Controls (same pipeline; each applies to the condition stated). A threshold on ONE arm
uses the P99 of that arm's own null (mean over k seeds of S_arm under H0,
`null.simulate_arm_null`), never the P99 of the null of D:
1. Positive (`asym`): a scripted listener that reads the true target through the message
   must reach >= 0.95. If not, the pipeline cannot measure and no D is read.
2. Negative (`asym` only): `msg_shuffle` must not exceed 1/C by more than the P99 of its
   single-arm null.
3. Learning gate (`asym` only): `eng` above 1/C + the P99 of its single-arm null before D
   is interpreted (floor effects are "not measurable", never "null effect").
   Gate for `sym`: `eng_sym >= 1 - epsilon` and `nocomm_sym >= 1 - epsilon` (epsilon:
   TBD-15); if it fails, the sym leak check is "not measurable".
4. Leak audit (`asym`). (a) A priori, before the pilot, for each backend: the target
   position among the C candidates is uniform and independent of everything the listener
   observes (EGG reads the target position from the data file, so the file generator is
   audited); a centralised critic that sees the target during training is not an
   execution-time leak. (b) A posteriori: `nocomm` above 1/C + the P99 of its single-arm
   null means a side channel; the run set is invalid.
5. Sym leak check (reported next to the results, no role in H1): D_sym against its own
   paired null (B0 = 1.0; P99 = 0 at sigma = 0 by construction), see section 5, case D.

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+: one-sided paired sign test at alpha 0.05
and paired bootstrap (10 000 resamples, percentile) of D_asym on k seeds. k is fixed in
the manifest by a rule written in advance that gives power 0.9 for the test actually
used (TBD-13): k = max(MIN_K, k_mean, k_sign), with (`experiments/n10d/sigma_seed.py`)
- k_mean = ceil(((z_0.95 + z_0.90) * sigma_D / margin)^2) (`required_seeds_from_sigma_d`),
  sigma_D the per-seed sd of D. Var(D) = Var(S_eng) + Var(S_nocomm) - 2 Cov, so the
  sigma_seed of `nocomm` alone does not determine it (source: TBD-14). With sigma_D taken
  directly there is no factor 2 (sigma_D = 0.30 gives k = 9 at margin 0.30). The former
  bound sigma_D^2 = 2 (sigma_seed^2 + 0.25 / n_eval) (`required_seeds`) assumes two
  independent arms and is kept for the report only.
- k_sign = smallest k such that the sign test rejects with power >= 0.9 when
  P(D_i > 0) = q (`sign_test_seeds`); at k = 5 only 5/5 rejects (p = 1/32) and the power
  is q^5 = 0.59 at q = 0.9. For reference (exact binomial): k_sign = 23, 11, 8 at
  q = 0.80, 0.90, 0.95.
Margin (TBD-12), q and MIN_K (TBD-13), and the source of sigma_D (TBD-14) are open.
Pilot: 8 development seeds 9600-9607, asym; `sigma_seed.py` reads one arm per call
(`nocomm`; `eng` only if TBD-14 allows it), never computes D, reads no NH-96 score; the
pilot seeds are never pooled with confirmatory runs. Non-rejection is "non concluant",
never "refuted".

## 5. Outcomes and consequences (case C closed by TBD-6)

A. The `asym` controls of section 3 pass, D_asym meets the confirmation rule (TBD-16,
   margin TBD-12) and the one-sided paired sign test rejects at alpha 0.05: the channel
   plumbing and the trunk learn the engineered code on this task. This is not read as
   "necessity confirmed": necessity is fixed by construction (section 1).
B. D_asym >= 0.10 without meeting case A: weaker than predicted, or "non concluant" if the
   test does not reject (TBD-16 fixes the boundary); report with CI.
C. D_asym < 0.10: failed prediction. "The architecture blocks" is a new hypothesis,
   reserved to a separate amendment (e.g. a channel wired into the policy input)
   filed before any new run.
D. Not an H1 outcome (PI-1). D_sym above the P99 of its paired null: possible leak or
   optimisation effect (trained policies at finite budget can sit below 1.0 without any
   leak: `nocomm_sym` at 0.625 against `eng_sym` at 1.0 gives D_sym = 0.375);
   audit required. The reading of A-C is suspended until the audit is reported.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a-priori generator audit (control 4a),
ledger code. Before the freeze: TBD-10 to TBD-16, sigma_seed, k, freeze manifest.

## 7. Two backends (TBD-1, TBD-2)

Same task (K=16, C=4, A=16, conditions asym/sym, arms of section 3), same learner (the
N10-C PPO trunk, adapted per backend; same hyperparameters wherever the interface
allows, each difference listed in the manifest), same metric and same seeds list.
JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6, reviewed) in
PyTorch; neither is pooled with the other. Each backend gets its own null calibration
and its own sigma_seed. Each (backend, arm, seed) cell stays on one host.

JaxMARL game (TBD-11): a custom JAX Lewis game with exactly the task above: K = 16
targets, C = 4 candidates in uniform random order, alphabet A = 16, message length L
(TBD-11), one step (the speaker emits, then the listener acts), one identical team reward
(1 if the listener picks the target, else 0). MPE `simple_reference` does not fit
(3 landmarks asserted, `dim_c` = 10, reward mixed by `local_ratio` = 0.5; JaxMARL
@31b0a23, `simple_reference.py:25,26,31,37,161`) and is not used.

Divergence rule (TBD-10, proposal). Status per backend: `effect` (case A),
`non concluant` (case B, or the test does not reject), `failed prediction` (case C),
`not measurable` (control 1 or the learning gate fails), `invalid` (leak audit fails).
Combination: both `effect` -> global effect; both `failed prediction` -> global failed
prediction; any other pair, including one `invalid` or one `not measurable` ->
global "non concluant". Two non-rejections do not establish a common null, and the
conjunction is not a Bonferroni correction. If the backends disagree, the cause is a
separate hypothesis for an amendment, and no result is selected after the fact.
