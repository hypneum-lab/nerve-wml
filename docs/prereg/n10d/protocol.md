# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. The open "TBD by PI" item TBD-11 blocks the freeze (hypneum-prereg).
**Amended:** 2026-11-06 after a blind contradictor, every finding checked in source
(section 8), then by PI decisions of 2026-11-06 16:55 CET (section 9), then reconciled
with the parallel revision of PR #100 (section 8, "Reconciliation 2026-11-06"; PI
decision 17:12 CET), then reconciled a second time with the head of PR #100 at cb7d66b
(PI decisions of 17:05 CET on TBD-10 and TBD-12 to TBD-16, section 0; section 8,
"Second reconciliation (cb7d66b)"). Nothing was frozen, so the text is edited in place;
the trail is sections 8 and 9.
**Prepared:** 2026-10-31 (night), by decision of the PI (multiple-choice, 2026-10-31 22:11 CET).
**Origin:** `docs/milestones/2026-10-25-n10c-bis-control-arms.md`, section "Necessity gate",
item G3: on failure of the gate (`eng - nocomm >= 0.20`) N10-C-bis is not frozen and the
next study is N10-D, "an asymmetric cooperative task in which each agent holds private
information that its partner needs", pre-registered in its own document before any run.
NH-96 (catalogue of new hypotheses, `.github`, `2026-10-30-etat-des-lieux/synthese/priorites-hypotheses.md`)
is the hypothesis behind it. No equivalent branch, PR or file existed at 2026-10-31.
**Annex G7 (PR #99) is not touched.**

## 0. PI decisions 2026-10-31 (multiple-choice, 22:30-22:35 CET), updated 2026-11-06

Nothing in this table is frozen. CLOSED means decided by the PI for this draft, not
registered; OPEN items block the freeze.

| id | decision (closed unless marked TBD) |
|---|---|
| TBD-1 | CLOSED. Both backends, JaxMARL and EGG. EGG is forked into the `ailiance` org at commit 54eb11c6 with code review. Articulation and divergence rule: section 7 |
| TBD-2 | CLOSED. The N10-C PPO trunk, adapted to each backend (section 7) |
| TBD-3 | CLOSED. K=16, C=4, A=16; blind ceiling 1/C = 0.25 (asym) stays exact |
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived by the rule of section 4 (alpha 0.05). Margin: TBD-12; k: TBD-13. *The 2026-10-31 formula (k from sigma_seed, minimum 5) and its 2026-11-06 A4 amendment (sign test from a pilot sigma_D, minimum 8) are superseded by TBD-13 and TBD-14* |
| TBD-5 | CLOSED. D_sym must not exceed the P99 of its calibrated null. *Superseded 2026-11-06 (A5) by PI-1: `sym` leaves H1; it is a leak check with its own gate (section 3, control 3; section 5, case D)* |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. Not amended (TBD-14): the pilot reads `nocomm` only |
| PI-1 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 1). `sym` leaves H1; it stays a leak check reported next to the results |
| PI-2 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 2). The null is paired by seed (seed effect common to both arms), `nh96-null.md` section 2 |
| TBD-10 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Divergence rule of section 7: per-backend statuses and conjunctive combination table (both backends must conclude). k is sized for the global power (section 4, TBD-13) |
| TBD-11 | OPEN. Deferred until a throughput measurement without score (one cell, macM1 or KXKM). Working values, not decisions: L = 1, 100 logs per run, trunk MLP 2 x 256, pilot budget B/4; observation encoding per backend; evaluation action rule (sampled or argmax). B is not fixed. n_eval = 500 CLOSED (A8, confirmed by the PI 2026-11-06 16:55, section 9) |
| TBD-12 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Absolute margin m = 0.30, written 0.4 x (1 - 1/C). Under the CI rule (TBD-16) a listener at 55 % no longer passes: the CI lower bound of D_asym must reach 0.30 |
| TBD-13 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). k by simulation of the actual rule (TBD-16: CI rule + sign test) under the planning assumption of section 4, power >= 0.90 for the conjunction of the two backends (TBD-10): K_CONFIRM = 18 per backend (macM1, 20 000 simulations, 2026-11-06) (section 4, `experiments/n10d/required_seeds_mc.py`). The former k = max(MIN_K, k_mean, k_sign) is obsolete |
| TBD-14 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). sigma_D comes from the planning assumption (section 4), not from the pilot. TBD-9 is not amended: the pilot reads `nocomm` only; sigma_seed(nocomm/asym) is a consistency check (control 4b), it does not size k. No sigma is transported to sym |
| TBD-15 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). Sym gate on `eng_sym` only: mean over k seeds >= 1 - epsilon, epsilon = 0.05 (`SYM_EPSILON`); `nocomm_sym` not gated. Case D iff the lower bound of the 95 % CI of D_sym >= delta_sym = 0.10 (`SYM_LEAK_DELTA`) |
| TBD-16 | CLOSED (PI decision, QCM in session, 2026-11-06 17:05). CI rule, section 5: A iff the lower bound of the 95 % paired CI of D_asym >= margin and the sign test rejects; NH-96 rejected iff the upper bound < margin; otherwise "non concluant". Percentile bootstrap for now; BCa or paired t to be fixed at the freeze |

## 1. Question

Do the channel wiring and the N10-C PPO trunk, adapted per backend, realise a large
part of the contrast that the task makes available? The information need itself is
analytical, not empirical: the target is uniform among C candidates, so the optimal
`eng - nocomm` contrast is 1 - 1/C = 0.75 on `asym` and 0 on `sym` (`oracle_contrast()`;
K=16, C=4). This is the positive-listening quantity Pl of Eccles et al. 2019 (section
4.1), computable without training. N10-D does **not** test whether the message is
informationally necessary; that holds by construction. It tests the plumbing and the
learner on this task: outcome A reads "the channel plumbing and the trunk learn the
engineered code on this task". NH-96 is labelled VARIANT (expected result: Lowe et al.
2019; Lazaridou et al. 2017 report 99-100 % success on a referential game of this kind).
The symmetric twin is not part of the question (PI-1); it is a leak check.

## 2. Link with the gate

- Gate passes (`eng - nocomm >= 0.20` on N10-C): N10-D is not needed for necessity;
  it stays an optional follow-up. Nothing here is run.
- Gate fails: N10-C-bis is not frozen (G3); N10-D is the next study. A gate failure is
  ambiguous between "the Procgen task does not need the message" and "the N10-C channel
  or trunk does not carry it". N10-D may support the first reading only under a
  condition this protocol does not cover: an N10-D arm executed inside the N10-C harness
  with the same channel code. That arm is an extension, outside this protocol (A10);
  without it, no N10-D outcome is read back onto N10-C. A gate failure is a
  design-decision input, not an N10-D result; gate runs are never pooled with N10-D runs.

## 3. Design

Conditions: `asym` (only the speaker sees the target; the listener sees C candidates
that contain it) and `sym` (both see the target). H1 and the NH-96 null are defined on
`asym` only; `sym` is a leak check: its analytical contrast is 0, it has its own gate
(control 3) and its own read-out (case D, section 5). Arms (code:
`experiments/n10d/spec.py`; environment: section 7):

| arm | channel | role |
|---|---|---|
| `eng` | engineered sender, ideal code, fixed | upper reference, as in the gate |
| `nocomm` | zeroed | blind reference |
| `learned` | trained jointly, alphabet A | emergent-protocol arm (secondary) |
| `msg_shuffle` | evaluation only, on the trained `eng` pair: message of another episode of the same evaluation batch (uniform episode derangement, no label exclusion, `derange_episodes`) | negative control |

`msg_shuffle` permutation (`interfaces.derange_episodes`, applied per evaluation batch by
`Channel.transmit` with a seeded `random.Random(batch_seed)`): a fixed-point-free
permutation of the episodes of the batch, so each listener receives the message emitted
for another episode; it keeps the marginal of messages and their length; identical
target labels across episodes are not excluded (a 1/K coincidence is not a leak). A
fixed permutation of target identities is not allowed: the message would stay a
deterministic function of the target and a listener could learn it up to 1.0.

Metric (primary, A8): S = success of the frozen policy pair after training, evaluated
on n_eval = 500 fresh episodes (confirmed by the PI 2026-11-06), episode seeds disjoint from
the training stream, so S given the seed is Binomial(n_eval, p)/n_eval. The former
metric "mean of the last 5 logged training values" is dropped: its windows are not a
Binomial(n_eval) and n_eval was undefined for it. This departs from gate G2, whose
numbers are not compared with N10-D numbers. D = mean over seeds of per-seed
`S_eng - S_nocomm`, paired by seed. Secondary: `learned - nocomm`, `eng - msg_shuffle`.
Blind ceiling (exact, `blind_ceiling()`): asym 1/C, sym 1.0. Oracle contrast: section 1.

Quantiles used by the controls (`experiments/n10d/null.py`, measured sigma_seed, the
registered k and n_eval): P99_asym = P99 of the paired null of D on asym; L99 = P99 of
the level null of one blind arm mean on asym (`simulate_level_null`). Controls 2 and 4
use L99, their own level quantile, never the P99 of D (A9).

Controls (reported next to every result, same pipeline):
1. Positive: a scripted listener that reads the true target through the message must
   reach >= 0.95 on `asym`. If not, the pipeline cannot measure and no D is read.
2. Negative: `msg_shuffle` mean on `asym` must be <= L99 (blind ceiling within its level null).
3. Learning gates (floor effects are "not measurable", never "null effect"):
   - asym: `eng` above `blind ceiling + P99_asym` before D_asym is interpreted.
   - sym (TBD-15): mean of `eng_sym` over k seeds >= 1 - epsilon, epsilon = 0.05
     (`decision.sym_gate`); `nocomm_sym` is not gated. If the gate fails, the sym leak
     check is "not measurable".
   The gates use the arm means; the training curve is never a gate.
4. Leak audit (A15). (a) A priori, once per backend, before the pilot, committed with
   the environment hash: the asym listener observation holds no field derived from the
   target (no index, no position, candidate order shuffled per episode); the target
   position is uniform over the C slots on 10 000 generated episodes (chi-square,
   alpha 0.01); scripted blind listeners (always slot 0; argmax of each single
   observation feature) stay <= L99 on the same episodes. Result in the manifest.
   EGG reads the target position from its data file, so for EGG the file generator is
   what is audited. A centralised critic that sees the target during training is not
   an execution-time leak.
   (b) At run time: `nocomm` on `asym` above L99 means a side channel; the run set is
   invalid (outcome row 0, section 5). sigma_seed(nocomm/asym) clearly above 0 points
   the same way (consistency check, TBD-14).

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+, asym: one-sided paired sign test of D,
H0 P(D_i > 0) = 1/2, so the count of positive seeds is Binomial(k, 1/2); ties count as
non-positive; reject if the count is >= m_k, the smallest m with P(Bin(k, 1/2) >= m)
<= 0.05 (`sign_test_critical`). Alongside: paired bootstrap 95 % CI of D_asym (10 000
resamples, percentile). Both are combined by the CI rule of section 5 (TBD-16,
`experiments/n10d/decision.py`).

Planning assumption (TBD-13, TBD-14). This is an inference used for planning, not a
measurement. Seeds are bimodal: a fraction pi = 0.8 learns the engineered code
(D_i ~ N(0.75, 0.03)) and the rest stays blind (D_i ~ N(0, 0.027), the binomial sd of a
paired difference at n_eval = 500). Then Delta = 0.60, sigma_D = 0.30 and
q = P(D_i > 0) = 0.90. sigma_D is not estimated from the pilot (TBD-14).

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
k = 19, 20, 21 (separate check, same settings). One backend alone would need k = 16.
The power is not monotone in k because the sign-test critical count moves in steps.
Type I against the margin (true mean D = 0.30, pi = 0.4) at k = 18: 0.023; P(A) under
H0 (pi = 0): 0. Monte-Carlo sd of the power about 0.0015. The former rule
k = max(MIN_K, k_mean, k_sign) is obsolete: k_mean (`required_seeds_from_sigma_d`) gives
the power of a test of D > 0 at D = margin, which is neither the CI rule nor the sign
test; the paired-mean floor and the pilot sigma_D of PR #115 are not kept.

Pilot: 8 development seeds 9600-9607, asym; `sigma_seed.py` reads `nocomm` only
(TBD-14), never computes D, prints no k, reads no NH-96 score. The pilot seeds are
never pooled with confirmatory runs. Non-rejection is "non concluant", never "refuted".

## 5. Outcomes and consequences (one decision table, A7; case C closed by TBD-6)

D-hat = point estimate of D_asym; L, U = lower and upper bounds of its 95 % paired
bootstrap CI. The rule is TBD-16, the CI rule (`decision.decide_asym`), applied per
backend. Rows are checked in this order; the first that holds is the outcome, so every
result has exactly one.

| order | outcome | rule | reading |
|---|---|---|---|
| 0 | invalid | leak audit (control 4, a or b) fails | side channel; the run set is invalid, no D read |
| 1 | non mesurable | any other asym control (1, 2, 3-asym) fails | no D read, never "supported" |
| 2 | C | D-hat < 0.10 | failed prediction. "The architecture blocks" is a new hypothesis, reserved to a separate amendment (e.g. a channel wired into the policy input) filed before any new run |
| 3 | A | L >= 0.30 and the sign test rejects | NH-96 supported: the channel plumbing and the trunk learn the engineered code on this task; never read as "necessity confirmed" (section 1) |
| 4 | B | U < 0.30 | weaker than predicted; NH-96 rejected (`nh96-null.md` section 4); report D-hat with its CI |
| 5 | non concluant | anything else (the CI contains 0.30, or the sign test does not reject) | the prediction is neither met nor excluded; report D-hat and CI |

Case D (sym, not an H1 outcome, PI-1): sym gate passed (`eng_sym` mean >= 0.95) and the
lower bound of the 95 % paired CI of D_sym >= 0.10 (TBD-15, `decision.sym_leak_status`).
Label: "possible leak or optimisation effect" (trained policies at finite budget can sit
below 1.0 without any leak: `nocomm_sym` at 0.625 against `eng_sym` at 1.0 gives
D_sym = 0.375, which fires case D); audit required. It is reported next to the asym
outcome and does not change it, but the reading of that outcome is suspended until the
audit is reported. A gap below 0.10 does not suspend A-C. The degenerate sym null
(`nh96-null.md` section 3) is reported descriptively only.

Margin (A12, TBD-12): 0.30 is 0.40 of the oracle contrast 0.75 (success 0.55, against
0.25 blind and 1.0 optimal). It asks the trained pair to recover a clear share of the
available gain while staying more than ten times above every asym null P99 (<= 0.026,
`nh96-null.md` section 3), so the margin, not noise, binds; near-optimal learning is not
required for a plumbing test. It is a PI choice, not derived from data.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a priori leak-audit script (control 4a),
ledger code. Before the freeze: TBD-11 (throughput measurement without score, then B),
sigma_seed, the CI method (percentile, BCa or paired t), freeze manifest.

## 7. Two backends (TBD-1, TBD-2) and the environment (A11)

Environment, identical on both backends (one step per episode):
1. The target t is uniform over K = 16 identities; C - 1 = 3 distractors are drawn
   uniformly without replacement from the other identities; the C candidates are
   shuffled uniformly per episode.
2. Speaker observation: t (one identity). Listener observation: asym, the C candidates
   in shuffled order; sym, the same plus t.
3. Order: the speaker emits one symbol of alphabet A = 16 (A = K: the engineered code
   maps each identity to one symbol); the arm's channel transforms it; the listener
   receives the channel output with its observation; the listener acts.
4. Listener action: the index in {0, ..., C - 1} of one candidate (always a valid one).
5. Reward: common to both agents, 1 if the chosen candidate is t, else 0; episode ends.
TBD-11: observation encoding per backend (one-hot or embedding), message length of the
`learned` arm if above one symbol, evaluation action rule (sampled or argmax). JaxMARL
runs a custom environment implementing this list, not MPE `simple_reference` (3
landmarks, communication dimension 10, `local_ratio` 0.5: a different game; JaxMARL
@31b0a23, `simple_reference.py:25,26,31,37,161`). EGG runs
it as a Lewis signalling game.

Same learner (the N10-C PPO trunk, adapted per backend; same hyperparameters wherever
the interface allows, each difference listed in the manifest), same metric and same
seeds list. JaxMARL runs the environment in JAX; EGG (forked, commit 54eb11c6,
reviewed) in PyTorch; neither is pooled with the other. Each backend gets its own null
calibration and its own sigma_seed. Each (backend, arm, seed) cell stays on one host.

Divergence rule (TBD-10, closed 2026-11-06 17:05). Each backend gets its own status
(invalid, non mesurable, A, B, C or non concluant, section 5, plus case D), reported
separately. Combination: both A -> global A; both C -> global C; any other pair,
including one invalid or one non mesurable -> global "non concluant". Two non-rejections
do not establish a common null, and the conjunction is not a Bonferroni correction. If
the backends disagree, the cause is a separate hypothesis for an amendment, and no
result is selected after the fact.

## References

- Lowe, Foerster, Boureau, Pineau, Dauphin 2019, "On the Pitfalls of Measuring Emergent
  Communication", AAMAS 2019, arXiv:1903.05168 (section 6.2).
- Eccles, Bachrach, Lever, Lazaridou, Graepel 2019, "Biases for Emergent Communication
  in Multi-agent Reinforcement Learning", NeurIPS 2019, arXiv:1912.05676 (section 4.1,
  positive listening Pl, computed explicitly from optimal policies).
- Lazaridou, Peysakhovich, Baroni 2017, "Multi-Agent Cooperation and the Emergence of
  (Natural) Language", ICLR 2017, arXiv:1612.07182 (section 4, Table 1).

Both contradictor passes of 2026-11-06 read these three texts in full
(`docs/n10d/contradictor-2026-11-06/`, PR #113). References are cited by arXiv id, not
by bracket number (A20: the arXiv version of Lowe 2019 numbers Lazaridou 2017 as [25]).

## 8. Amendments 2026-11-06 (blind contradictor, verified)

Sources, all in `docs/n10d/contradictor-2026-11-06/` (PR #113, branch at 8ca3c20):
`verification-2026-11-06.md` (source verification without the reading note; "n." below
is its finding number) and `VERIFICATION.md` with `verification-{claude,codex}-report.md`
(verification with the reading note and the brainstorm). 22 findings, 0 refuted.

- A1 (n. 2): learning gate split; asym unchanged, sym gets its own gate (section 3, control 3).
- A2 (n. 1): outcome A reworded "plumbing and PPO learning validated"; analytical
  contrast 0.75 / 0 written as reference (`oracle_contrast`, section 1).
- A3 (n. 10): `sigma_seed.py` CLI reads n_eval from the filtered rows, error if they differ; CLI test.
- A4 (n. 3, 4, 14, 16, 17): k sized for the sign test on a sigma_D bound from the eng
  and nocomm spreads; MIN_K 5 -> 8; table in section 4. *The bound was replaced by the
  paired sd of D (PI decision 2026-11-06 16:55, section 9), itself superseded by the
  second reconciliation (TBD-13, TBD-14).*
- A5 (n. 5, 11): `sym` out of H1 and paired null, both per PI decisions of
  `brainstorm-2026-11-01.md` section 0 (items 1-2); `nh96-null.md` sections 1-2.
- A6 (n. 13): latent blind success capped at B0 in `null.py`; `nh96-null.md` section 2.
- A7 (n. 7): one decision table tying sign test, CI and margin to the outcomes (section 5).
- A8 (n. 6): metric = separate evaluation of the frozen policy, n_eval = 500 proposed;
  last-5-logs metric dropped (section 3, `interfaces.Trainer`).
- A9 (n. 8, 18): `msg_shuffle` evaluation only; controls 2 and 4 on the level quantile L99.
- A10 (n. 19): reading of a gate failure back onto N10-C made conditional on an
  N10-D arm in the N10-C harness, declared outside this protocol (section 2).
- A11 (n. 12, 15): explicit environment replaces "simple_reference-like" (section 7, `backends.py`).
- A12 (n. 20): margin 0.30 stated as 0.40 of the oracle, with a justification (section 5).
- A13 (n. 9): Eccles 2019 and Lazaridou 2017 added (References).
- A14 (`VERIFICATION.md`): `msg_shuffle` deranges episodes, never target identities (`derange_episodes`).
- A15 (`VERIFICATION.md`): control 4 gains an a priori audit of the generator.
- A16 (`VERIFICATION.md`): case D labelled "possible leak or optimisation effect".
- A17 (`VERIFICATION.md`): TBD-10 proposal gives one status per backend.
- A18 (n. 11, `VERIFICATION.md`): "0 by construction" on sym holds for optimal policies only (`nh96-null.md` section 5).
- A19 (n. 10, 22): tests for the degenerate sym null and the CLI n_eval path.
- A20 (`VERIFICATION.md`): Lowe 2019 numbering; no bracket number is used here, nothing else to change.

Not applied: none. Freeze lock (verrou de lecture): not passed; no tag.

### Reconciliation 2026-11-06

Two branches revised the 6e682cc draft in parallel on 2026-11-06: PR #100
(`prereg/n10d-skeleton`, 1ccb78b and e5fbe88, 13:48, its revisions R1-R13, from the 19
points of `VERIFICATION.md`) and PR #115 (this branch, A1-A20 above and section 9). The
PI decided at 17:12 CET to bring the head of #100 into #115 by one merge commit.
**PR #115 is the single branch to merge into #100; PI decision 17:12 CET.**
Resolution rule, in order: (a) the PI decisions of 16:55 (section 9) always win;
(b) where both branches fixed the same point differently, the stricter text for the
freeze lock is kept (declares more, requires more, leaves less implicit) and the other
is noted below in one line; (c) what only #100 adds is kept; (d) no duplication: one
amendments section, one decision table, one sigma_D, one function per role; (e) the tests
of both branches are kept and must pass.

Where #100 won or was kept:
- Section 0: rows PI-1 and PI-2.
- Status "invalid" kept apart from "non mesurable" (row 0 of section 5, per-backend
  statuses); explicit backend combination (both A, both C, otherwise "non concluant";
  two non-rejections give no common null; not a Bonferroni correction). Alternative
  (#115): "global status only if both are the same".
- Case D suspends the reading of the asym outcome until the audit is reported; the
  0.625 counterexample is written in section 5.
- `Channel.transmit(messages, *, batch_seed)` works per batch with a seeded derangement;
  the reason a fixed permutation of target identities is forbidden is written in section 3.
- `Trainer.evaluate` returns integer counts (successes, n_eval). Alternative (#115): a
  float success rate. One method, on `Trainer` (#100's separate `Evaluator` dropped).
- Control 4a: for EGG the data-file generator is audited; a centralised critic that sees
  the target during training is not an execution-time leak.
- JaxMARL `simple_reference` source lines (section 7, `backends.py`); in `nh96-null.md`:
  no H0 on sym and its degenerate null stated, the latent effect is not required by the
  game, a sigma measured on asym is never transported to sym.

Where #115 won:
- n_eval = 500 (PI, 16:55) over #100's TBD-11 working value.
- `msg_shuffle` is evaluation only, on the trained eng pair (A9). Alternative (#100): an
  arm trained and evaluated under the derangement; not kept, left to the PI.
- The last-5-logs training summary is dropped (A8, confirmed by the PI); #100 kept it as
  a secondary summary.
- Latent blind success capped at B0 (A6); #100 clipped to [0, 1]. The table of
  `nh96-null.md` section 3 is the capped one; #100's table (uncapped) is superseded.
- One level null, `simulate_level_null` (L99 on the level); #100's `simulate_arm_null`
  (same draws minus B0) is removed and its test rewritten on the level null.
- One decision table (section 5, A7) over #100's prose cases A-D.
- Name `derange_episodes` (not `episode_derangement`).

Points of this first reconciliation that the second one (below) replaced: the pilot
sigma_D with declared breach no. 1, MIN_K 8 and the paired-mean floor, eps_sym =
1/n_eval on both sym arms with failure as flag D, the point-plus-sign-test table with
L > 0, TBD-10/12/16 left open, and `sign_test_threshold` raising.

Tests: all tests of both branches were kept at the first reconciliation; #100's tests
were adapted to the merged code (`episode_derangement` -> `derange_episodes`;
`simulate_arm_null` -> the level null minus B0). The two
`test_sym_null_is_degenerate_at_zero_sigma` are one test holding both sets of assertions.

### Second reconciliation (cb7d66b)

Recorded in the next commits of this branch.

## 9. PI decisions 2026-11-06 (16:55 CET)

Taken by the PI on the choices listed in the draft PR (#115) after section 8.

| item | decision |
|---|---|
| sigma_D (A4) | CHANGED. sigma_D = sd (ddof = 1) of the paired per-seed differences eng - nocomm on the asym pilot, fed directly to `required_seeds`; the Cauchy-Schwarz bound is dropped. The pairing is a declared blinding breach (breach no. 1). MIN_K 8, sign-test table and MAX_K 200 unchanged |
| n_eval (A8) | CONFIRMED without change: n_eval = 500, eps_sym = 1/n_eval |
| A10 | CONFIRMED without change: reading "Procgen does not need the message" stays conditional on an N10-D arm in the N10-C harness, outside this protocol (section 2) |
| Q8 (N10-C-bis vs N10-D) | DEFERRED to the session "point de reprise roadmap v2"; nothing decided here |

No freeze, no tag; the freeze lock (verrou de lecture) is not passed.
