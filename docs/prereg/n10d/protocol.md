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
Update 2026-11-06 (PI decision, multiple-choice in session, about 17:05 CET): TBD-10 and
TBD-12 to TBD-16 closed (section 0); TBD-11 is the only open item. Sym gate on `eng_sym`
only and case D through a CI bound (TBD-15, origin: the BLOQUANT "sym gate impossible" of
the contradictor pass, PR #113, `docs/n10d/contradictor-2026-11-06/VERIFICATION.md`); CI
decision rule (TBD-16); k by simulation of that rule (TBD-13, section 4).

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. The open "TBD by PI" item TBD-11 blocks the freeze (hypneum-prereg).
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
decided by the PI for this draft, not registered; the OPEN item TBD-11 blocks the freeze.

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived by the rule of section 4 (alpha 0.05; power: TBD-13). Margin: TBD-12; k: TBD-13 |
| TBD-5 | SUPERSEDED 2026-11-06 by PI-1. D_sym is no longer in H1; it is reported as a leak check against its own paired null (section 3, control 5; section 5, case D) |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. Not amended (TBD-14): the pilot reads `nocomm` only |
| PI-1 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 1). `sym` leaves H1; it stays a leak check reported next to the results |
| PI-2 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 2). The null is paired by seed (seed effect common to both arms), `nh96-null.md` section 2 |
| TBD-10 | CLOSED (PI decision, QCM in session, 2026-11-06). Divergence rule of section 7 confirmed as written: per-backend statuses and conjunctive combination table (both backends must conclude). k is sized for the global power (section 4, TBD-13) |
| TBD-11 | OPEN (reworded 2026-11-06). Deferred until a throughput measurement without score (one cell, macM1 or KXKM). Working values, not decisions: L = 1, n_eval = 500, 100 logs per run, trunk MLP 2 x 256, pilot budget B/4. B is not fixed |
| TBD-12 | CLOSED (PI decision, QCM in session, 2026-11-06). Absolute margin m = 0.30, written 0.4 x (1 - 1/C). Under the CI rule (TBD-16) a listener at 55 % no longer passes: the CI lower bound of D_asym must reach 0.30 |
| TBD-13 | CLOSED (PI decision, QCM in session, 2026-11-06). k by simulation of the actual rule (TBD-16: CI rule + sign test) under the planning assumption of section 4, power >= 0.90 for the conjunction of the two backends (TBD-10): K_CONFIRM = 18 per backend (macM1, 20 000 simulations, 2026-11-06) (section 4, `experiments/n10d/required_seeds_mc.py`). Provisional value of the session was 19 (1 000-1 500 simulations, learned D_i centred at 0.74). The former k = max(MIN_K, k_mean, k_sign) is obsolete |
| TBD-14 | CLOSED (PI decision, QCM in session, 2026-11-06). sigma_D comes from the planning assumption (section 4), not from the pilot. TBD-9 is not amended: the pilot reads `nocomm` only, never the sd of `eng`; sigma_seed(nocomm/asym) is a consistency check (control 4b), it does not size k. No sigma is transported to sym |
| TBD-15 | CLOSED (PI decision, QCM in session, 2026-11-06). Sym gate on `eng_sym` only: mean over k seeds >= 1 - epsilon, epsilon = 0.05 (`SYM_EPSILON`); `nocomm_sym` not gated. Case D iff the lower bound of the 95 % bootstrap CI of D_sym >= delta_sym = 0.10 (`SYM_LEAK_DELTA`). Origin: BLOQUANT of the contradictor pass (PR #113, `docs/n10d/contradictor-2026-11-06/VERIFICATION.md`): gating both arms forces abs(D_sym) <= epsilon, so case D was reachable only for negligible gaps |
| TBD-16 | CLOSED (PI decision, QCM in session, 2026-11-06). CI rule (i), section 5: A iff the lower bound of the 95 % paired-bootstrap CI of D_asym >= margin and the sign test rejects; NH-96 rejected iff the upper bound < margin; otherwise "non concluant". Percentile bootstrap for now; BCa or paired t to be fixed at the freeze (type I against the margin about 2.9 % in the session simulation, 2.3 % at k = 18 on macM1, section 4) |

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
   Gate for `sym` (TBD-15): mean of `eng_sym` over k seeds >= 1 - epsilon, epsilon = 0.05;
   `nocomm_sym` is not gated. If the gate fails, the sym leak check is "not measurable".
4. Leak audit (`asym`). (a) A priori, before the pilot, for each backend: the target
   position among the C candidates is uniform and independent of everything the listener
   observes (EGG reads the target position from the data file, so the file generator is
   audited); a centralised critic that sees the target during training is not an
   execution-time leak. (b) A posteriori: `nocomm` above 1/C + the P99 of its single-arm
   null means a side channel; the run set is invalid.
5. Sym leak check (reported next to the results, no role in H1): D_sym against
   delta_sym = 0.10 through the lower bound of its 95 % paired-bootstrap CI (TBD-15,
   `decision.sym_leak_status`); its own paired null (B0 = 1.0; P99 = 0 at sigma = 0 by
   construction) is reported descriptively only. See section 5, case D.

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+: one-sided paired sign test at alpha 0.05
and paired bootstrap (10 000 resamples, percentile) of D_asym on k seeds, combined by the
CI rule of section 5 (TBD-16, `experiments/n10d/decision.py`).

Planning assumption (TBD-13, TBD-14). This is an inference used for planning, not a
measurement. Seeds are bimodal: a fraction pi = 0.8 learns the engineered code
(D_i ~ N(0.75, 0.03)) and the rest stays blind (D_i ~ N(0, 0.027), the binomial sd of a
paired difference at n_eval = 500). Then Delta = 0.60, sigma_D = 0.30 and
q = P(D_i > 0) = 0.90, the reference values used so far. sigma_D is not estimated from
the pilot (TBD-14).

Rule for k (TBD-13). k is the smallest value at which a simulation of the actual
decision rule (CI lower bound >= margin, sign test rejects, D_asym >= 0.10) under the
planning assumption gives a per-backend power p with p^2 >= 0.90 (both backends must
conclude, TBD-10; independence of the backends is assumed, not shown). Script:
`experiments/n10d/required_seeds_mc.py`, numpy, seed 20261106, run on macM1 on 2026-11-06
with `python -m experiments.n10d.required_seeds_mc --sims 20000 --boot 2000`
(20 000 simulations, 2 000 bootstrap resamples per simulation, same quantile index as
`decision.py`). Result (fact, simulation only, no run data): per-backend power
0.812, 0.838, 0.878, 0.875, 0.891, 0.919, 0.947, 0.952 at k = 11 to 18, so
**K_CONFIRM = 18 per backend** (global power 0.952^2 = 0.906); 0.970, 0.966, 0.977 at
k = 19, 20, 21 (separate check, same settings), so every k from 18 to 21 meets the target. One
backend alone would need k = 16. The power is not monotone in k because the sign-test
critical count moves in steps. Type I against the margin (true mean D = 0.30, pi = 0.4)
at k = 18: 0.023; P(A) under H0 (pi = 0): 0. Monte-Carlo sd of the power about 0.0015.
The former rule k = max(MIN_K, k_mean, k_sign) is obsolete: k_mean
(`required_seeds_from_sigma_d`) gives the power of a test of D > 0 at D = margin, which
is neither the CI rule nor the sign test. It, the independent-arms bound `required_seeds`
and MIN_K are kept in `sigma_seed.py` for the report only. For reference (exact
binomial, `sign_test_seeds`): k_sign = 23, 11, 8 at q = 0.80, 0.90, 0.95.
Pilot: 8 development seeds 9600-9607, asym; `sigma_seed.py` reads `nocomm` only
(TBD-14), never computes D, prints no k, reads no NH-96 score; the
pilot seeds are never pooled with confirmatory runs. Non-rejection is "non concluant",
never "refuted".

## 5. Outcomes and consequences (case C closed by TBD-6)

Decision rule (TBD-16, CI rule; `decision.decide_asym`), applied per backend once the
`asym` controls of section 3 have passed, in this order: D_asym < 0.10 -> C; lower bound
of the 95 % paired-bootstrap CI >= margin (0.30, TBD-12) and the sign test rejects -> A;
upper bound < margin -> B; otherwise "non concluant". The cases are disjoint.

A. CI lower bound of D_asym >= 0.30 and the one-sided paired sign test rejects at
   alpha 0.05: the channel plumbing and the trunk learn the engineered code on this task.
   This is not read as "necessity confirmed": necessity is fixed by construction
   (section 1).
B. D_asym >= 0.10 and CI upper bound < 0.30: weaker than predicted; NH-96 rejected
   (`nh96-null.md` section 4, criterion 1); report with CI. Between A and B (the CI
   contains 0.30, or the sign test does not reject): "non concluant".
C. D_asym < 0.10: failed prediction. "The architecture blocks" is a new hypothesis,
   reserved to a separate amendment (e.g. a channel wired into the policy input)
   filed before any new run.
D. Not an H1 outcome (PI-1). Sym gate passed (`eng_sym` mean >= 0.95) and the lower
   bound of the 95 % paired-bootstrap CI of D_sym >= 0.10 (TBD-15): possible leak or
   optimisation effect (trained policies at finite budget can sit below 1.0 without any
   leak: `nocomm_sym` at 0.625 against `eng_sym` at 1.0 gives D_sym = 0.375, which now
   fires case D instead of failing the gate); audit required. The reading of A-C is
   suspended until the audit is reported. A gap below 0.10 does not suspend A-C.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a-priori generator audit (control 4a),
ledger code. Before the freeze: TBD-11 (throughput measurement without score, then B),
sigma_seed, the CI method (percentile, BCa or paired t), freeze manifest.

## 7. Two backends (TBD-1, TBD-2)

Same task (K=16, C=4, A=16, conditions asym/sym, arms of section 3), same learner (the
N10-C PPO trunk, adapted per backend; same hyperparameters wherever the interface
allows, each difference listed in the manifest), same metric and same seeds list.
JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6, reviewed) in
PyTorch; neither is pooled with the other. Each backend gets its own null calibration
and its own sigma_seed. Each (backend, arm, seed) cell stays on one host.

JaxMARL game (TBD-11): a custom JAX Lewis game with exactly the task above: K = 16
targets, C = 4 candidates in uniform random order, alphabet A = 16, message length L
(TBD-11, working value 1), one step (the speaker emits, then the listener acts), one identical team reward
(1 if the listener picks the target, else 0). MPE `simple_reference` does not fit
(3 landmarks asserted, `dim_c` = 10, reward mixed by `local_ratio` = 0.5; JaxMARL
@31b0a23, `simple_reference.py:25,26,31,37,161`) and is not used.

Divergence rule (TBD-10, closed 2026-11-06). Status per backend: `effect` (case A),
`non concluant` (case B, or the test does not reject), `failed prediction` (case C),
`not measurable` (control 1 or the learning gate fails), `invalid` (leak audit fails).
Combination: both `effect` -> global effect; both `failed prediction` -> global failed
prediction; any other pair, including one `invalid` or one `not measurable` ->
global "non concluant". Two non-rejections do not establish a common null, and the
conjunction is not a Bonferroni correction. If the backends disagree, the cause is a
separate hypothesis for an amendment, and no result is selected after the fact.
