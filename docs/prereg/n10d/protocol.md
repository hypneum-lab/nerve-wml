# N10-D - asymmetric communication-necessity study (L0 skeleton)

**Status:** L0 DRAFT on a branch, in a draft PR. Not merged, not tagged, no run made.
It binds nothing. Open "TBD by PI" items (TBD-10, TBD-11, TBD-12, TBD-16) block the
freeze (hypneum-prereg).
**Amended:** 2026-11-06 after a blind contradictor, every finding checked in source
(section 8), then by PI decisions of 2026-11-06 16:55 CET (section 9), then reconciled
with the parallel revision of PR #100 (section 8, "Reconciliation 2026-11-06"; PI
decision 17:12 CET). Nothing was frozen, so the text is edited in place; the trail is
sections 8 and 9.
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
| TBD-4 | CLOSED. Seeds: pilot 9600+, confirmatory 600+. k derived from sigma_seed by the formula of section 4 (margin 0.30, alpha 0.05, power 0.9, minimum 5). *Amended 2026-11-06 (A4): k sized for the sign test, minimum 8; sigma_D = sd of the paired per-seed differences (PI decision 2026-11-06 16:55, section 9); k never below the paired-mean k (reconciliation, section 8)* |
| TBD-5 | CLOSED. D_sym must not exceed the P99 of its calibrated null, with sigma_seed measured on development seeds (no fixed 0.05). *Superseded 2026-11-06 (A5): `sym` leaves H1 (PI decision, `brainstorm-2026-11-01.md` section 0, item 1); sym is a descriptive control with its own gate (section 3)* |
| TBD-6 | CLOSED. D_asym < 0.10 = failed prediction; "the architecture blocks" is a new hypothesis for a separate amendment (section 5) |
| TBD-7 | CLOSED. Host: KXKM-AI after N10-C-bis; if possible a second DGX GB300 slot, declared as alternative host. Each arm and each seed stays on ONE host (no mixing inside an arm-seed cell) |
| TBD-8 | CLOSED. Grouped HITL review at lock time; flax one version back under the 2-week rule (0.12.10). Hashes still to be produced |
| TBD-9 | CLOSED. Pilot: 8 development seeds 9600-9607, reduced budget. NH-96 scores are not read; only sigma_seed is extracted, by `experiments/n10d/sigma_seed.py`, committed before the pilot. *Amended 2026-11-06 (A4, section 9): the script also reads eng/asym and computes the per-seed differences eng - nocomm to get their sd; it never computes their mean. Declared breach, section 4 "Pilot read-out (declared)"* |
| PI-1 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 1). `sym` leaves H1; it stays a descriptive plumbing and leak control reported next to the results |
| PI-2 | CLOSED (PI, 2026-11-01, brainstorm PR #102 section 0 item 2). The null is paired by seed (seed effect common to both arms), `nh96-null.md` section 2 |
| TBD-10 | OPEN. Backend-divergence rule (section 7): proposal "status per backend, global conclusion only if both agree" (A17) |
| TBD-11 | OPEN. Reduced pilot step budget; exact trunk adaptation per backend; environment details marked TBD-11 in section 7. n_eval = 500 CLOSED (A8, confirmed by the PI 2026-11-06, section 9) |
| TBD-12 | OPEN (raised by PR #100, finding n. 20). Margin: absolute 0.30 (TBD-4 of 2026-10-31, justified in section 5, A12) or relative to the oracle contrast 0.75 (e.g. 0.9 x 0.75 = 0.675)? The text keeps 0.30 until the PI confirms or changes it before the freeze |
| TBD-13 | CLOSED by the PI decision of 2026-11-06 16:55 (section 9): sign test, MIN_K 8, MAX_K 200, q = Phi(margin / sigma_D); the paired-mean k is kept as a floor (section 4) |
| TBD-14 | CLOSED by the PI decision of 2026-11-06 16:55 (section 9): sigma_D = sd of the paired per-seed differences, declared breach no. 1 (section 4). The sym null needs no sigma: sym is descriptive and its gate uses eps_sym (section 3) |
| TBD-15 | CLOSED by the PI decision of 2026-11-06 16:55 (section 9): eps_sym = 1/n_eval |
| TBD-16 | OPEN (raised by PR #100). One decision rule: the table of section 5 (point rule plus sign test plus L > 0) is the proposal; the alternative is the CI rule (A iff L >= margin). PI to confirm before the freeze |

## 1. Question

Do the channel wiring and the N10-C PPO trunk, adapted per backend, realise a large
part of the contrast that the task makes available? The information need itself is
analytical, not empirical: the target is uniform among C candidates, so the optimal
`eng - nocomm` contrast is 1 - 1/C = 0.75 on `asym` and 0 on `sym` (`oracle_contrast()`;
K=16, C=4). This is the positive-listening quantity Pl of Eccles et al. 2019 (section
4.1), computable without training. N10-D does **not** test whether the message is
informationally necessary; that holds by construction. It tests the plumbing and the
learner on this task: outcome A reads "channel plumbing and PPO learning validated on
this task". NH-96 is labelled VARIANT (expected result: Lowe et al. 2019; Lazaridou et
al. 2017 report 99-100 % success on a referential game of this kind). The symmetric
twin is not part of the question (PI-1); it is a control.

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
`asym` only; `sym` is a descriptive control arm: its analytical contrast is 0, D_sym is
reported without a decision, and it has its own plumbing gate (control 3). Arms (code:
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
   - sym (plumbing control, not a hypothesis): `eng_sym` and `nocomm_sym` means both
     >= 1 - eps_sym, eps_sym = max(P99 of the sym null at sigma_seed = 0, 1/n_eval) =
     1/n_eval, because that null is degenerate (every draw is exactly 1.0, P99 = 0):
     "1.0 to within one evaluation episode". Failure gives flag D (section 5).
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
   invalid (outcome row 0, section 5).

## 4. Analysis plan

Confirmatory (after freeze only), seeds 600+, asym: one-sided paired sign test of D,
H0 P(D_i > 0) = 1/2, so the count of positive seeds is Binomial(k, 1/2); ties count as
non-positive; reject if the count is >= m_k, the smallest m with P(Bin(k, 1/2) >= m)
<= 0.05. Alongside: paired bootstrap 95 % CI of D_asym (10 000 resamples, percentile).

k is fixed in the manifest by a formula written in advance (`required_seeds`,
`experiments/n10d/sigma_seed.py`, A4; PI decision 2026-11-06, section 9): sigma_D =
sample sd (ddof = 1) of the paired differences D_s = S_eng,s - S_nocomm,s over the asym
pilot seeds (binomial term included). The two arms must carry the same seed identifiers
and the same n_eval per seed, or the script stops. sigma_D feeds `required_seeds`
directly, with no factor 2 and no sum of bounds (Var(D) = Var(S_eng) + Var(S_nocomm)
- 2 Cov, so neither the nocomm spread alone nor the sum of the arm spreads fixes it) (it replaces the former Cauchy-Schwarz
bound sd(S_eng) + sd(S_nocomm), which overstates sd(D) when seeds are correlated: on
the synthetic correlated set of `tests/test_n10d_sigma.py` the bound gives k = 13 and
the paired sd gives k = 8). Under the alternative D_i ~ N(0.30, sigma_D), q = P(D_i > 0) =
Phi(0.30 / sigma_D); k = the smallest k in [8, 200] with power P(Bin(k, q) >= m_k) >= 0.9;
above 200 the design needs an amendment. Power is not monotone in k (discrete m_k):

| k | m_k | size P(Bin(k, 1/2) >= m_k) | power q = 0.8 | power q = 0.9 |
|---|---|---|---|---|
| 5 (former minimum) | 5 | 0.031 | 0.328 | 0.590 |
| 8 | 7 | 0.035 | 0.503 | 0.813 |
| 10 | 9 | 0.011 | 0.376 | 0.736 |
| 12 | 10 | 0.019 | 0.558 | 0.889 |

`required_seeds` also never returns less than the normal-approximation k of the paired
mean, max(8, ceil(((z_0.95 + z_0.90) sigma_D / 0.30)^2)) (no factor 2: 9 at sigma_D =
0.30; `required_seeds_from_sigma_d`, from PR #100). Under the normal alternative this
floor never binds on sigma_D in [0, 0.50] (sign-test efficiency is below that of the mean;
checked in `tests/test_n10d_sigma.py`); it guards the rule if the alternative is changed.
For reference, the sign-test k from MIN_K 8 is 23, 11 and 8 at q = 0.80, 0.90, 0.95.
Resulting k: 8 for sigma_D <= 0.20, 13 at 0.25, 16 at 0.30 (tests in
`tests/test_n10d_sigma.py`). Pilot: 8 development seeds 9600-9607, asym, arms `nocomm`
and `eng`; the script returns n_eval, sigma_seed (nocomm, binomial-corrected, for the
null), sigma_D and k, and computes no mean of D and no NH-96 score. The pilot seeds are
never pooled with confirmatory runs. Non-rejection is "non concluant", never "refuted".

**Pilot read-out (declared).** The pilot is development, not confirmatory: no NH-96
result is read from it and no outcome of section 5 is assigned to it. Because sigma_D
is the sd of eng - nocomm, the script now pairs the two arms; this is declared as
N10-D blinding breach no. 1 (the repository keeps no N10-D breach register; this
paragraph is the record, and any later breach takes the next number here). What is
read: four numbers only, n_eval, sigma_seed, sigma_D and k, printed by
`experiments/n10d/sigma_seed.py` from the pilot JSON; the PI and the analysis see
these four numbers and nothing else. What is not read: the mean (or median) of the
per-seed differences D_s, any per-seed D_s, any per-seed or per-arm success score,
the eng and nocomm arm means, the sign count, any test or CI; the script computes
the differences in memory only for their sd and never prints them, and no one opens
the pilot JSON or the training logs beyond run status. The pilot seeds 9600-9607 are
excluded from the confirmatory seeds (600+) and from every confirmatory analysis.
Any further look at the pilot data is a new breach, declared here before the freeze tag.

## 5. Outcomes and consequences (one decision table, A7; case C closed by TBD-6)

D-hat = point estimate of D_asym; L, U = lower and upper bounds of its 95 % paired
bootstrap CI. The rule is TBD-16 (proposal; alternative in section 8). Rows are
checked in this order; the first that holds is the outcome, so every result has one.

| order | outcome | rule | reading |
|---|---|---|---|
| 0 | invalid | leak audit (control 4, a or b) fails | side channel; the run set is invalid, no D read |
| 1 | non mesurable | any other asym control (1, 2, 3-asym) fails | no D read, never "supported" |
| 2 | C | U < 0.30 or D-hat < 0.10 | failed prediction, NH-96 rejected. "The architecture blocks" is a new hypothesis, reserved to a separate amendment (e.g. a channel wired into the policy input) filed before any new run |
| 3 | A | sign test rejects, D-hat >= 0.30 and L > 0 | NH-96 supported: channel plumbing and PPO learning validated on this task; never read as "necessity confirmed" (section 1) |
| 4 | B | anything else | non concluant: the prediction is neither met nor excluded; report D-hat and CI |

Example: D-hat = 0.25 with U = 0.40 is not C (U >= 0.30, D-hat >= 0.10) and not A
(D-hat < 0.30), so it is B. D-hat >= 0.30 with a non-significant sign test, or with
L <= 0, is B. D-hat >= 0.30 with 0 < L < 0.30 and a rejecting sign test is A (the
zone asked about in PR #100, TBD-16).

Flag D (sym, A16): the sym gate of control 3 fails. Label: "possible leak or
optimisation effect" (trained policies at finite budget can sit below 1.0 without any
leak: `nocomm_sym` at 0.625 against `eng_sym` at 1.0 gives D_sym = 0.375); audit
required. It is reported next to the asym outcome and does not change it, but the
reading of that outcome is suspended until the audit is reported. D_sym gets no test
(its null is degenerate, `nh96-null.md` section 3).

Margin (A12): 0.30 is 0.40 of the oracle contrast 0.75 (success 0.55, against 0.25 blind
and 1.0 optimal). It asks the trained pair to recover a clear share of the available
gain while staying more than ten times above every asym null P99 (<= 0.026,
`nh96-null.md` section 3), so the margin, not noise, binds; near-optimal learning is not
required for a plumbing test. It is a PI choice (TBD-4), not derived from data.

## 6. Not done here

No run, no install, no compute. Missing before the pilot: hashed environments for both
backends, the real environments (`backends.py` raises), the PPO-trunk adaptation, the
engineered sender and scripted listener, the a priori leak-audit script (control 4a),
ledger code. Before the freeze: TBD-10, TBD-11, TBD-12, TBD-16, sigma_seed, sigma_D, k,
freeze manifest.

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
calibration, its own sigma_seed and sigma_D. Each (backend, arm, seed) cell stays on one
host. Divergence rule (TBD-10, proposal, A17): each backend gets its own status
(invalid, non mesurable, A, B or C, plus flag D), reported separately. Combination:
both A -> global A; both C -> global C; any other pair, including one invalid or one
non mesurable -> global "non concluant". Two non-rejections do not establish a common
null, and the conjunction is not a Bonferroni correction. If the backends disagree, the
cause is a separate hypothesis for an amendment, and no result is selected after the fact.

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
  and nocomm spreads; MIN_K 5 -> 8; table in section 4. *The bound is replaced by the
  paired sd of D (PI decision 2026-11-06, section 9).*
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
- A16 (`VERIFICATION.md`): flag D labelled "possible leak or optimisation effect".
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
- Section 0: rows PI-1 and PI-2; TBD-12 (absolute or relative margin) and TBD-16 (one
  decision rule) kept OPEN; TBD-13, TBD-14 and TBD-15 recorded as closed by section 9.
- Outcome A also requires L > 0 (option (ii) of #100's TBD-16), and the zone
  "D-hat >= 0.30, 0 < L < 0.30" gets a stated status (A). Alternative (#100 option (i),
  CI rule, A iff L >= margin): left to the PI under TBD-16.
- Status "invalid" kept apart from "non mesurable" (row 0 of section 5, per-backend
  statuses); explicit backend combination (both A, both C, otherwise "non concluant";
  two non-rejections give no common null; not a Bonferroni correction). Alternative
  (#115): "global status only if both are the same", so both B or both non mesurable
  would have been global.
- Flag D suspends the reading of the asym outcome until the audit is reported; the
  0.625 counterexample is written in section 5.
- `Channel.transmit(messages, *, batch_seed)` works per batch with a seeded derangement;
  the reason a fixed permutation of target identities is forbidden is written in section 3.
- `Trainer.evaluate` returns integer counts (successes, n_eval). Alternative (#115): a
  float success rate. One method, on `Trainer` (#100's separate `Evaluator` dropped).
- The paired-mean k without factor 2 (`required_seeds_from_sigma_d`) is a floor of
  `required_seeds`, so k = max(sign-test k, paired-mean k), #100's TBD-13 form;
  `sign_test_seeds(q)` kept; the Var(D) identity is written in section 4.
- `READABLE_ARMS` guard: `sigma_seed.py` refuses any arm other than nocomm and eng.
- Control 4a: for EGG the data-file generator is audited; a centralised critic that sees
  the target during training is not an execution-time leak.
- JaxMARL `simple_reference` source lines (section 7, `backends.py`); in `nh96-null.md`:
  no H0 on sym and its degenerate null stated, the latent effect is not required by the
  game, a sigma measured on asym is never transported to sym.

Where #115 won:
- sigma_D = sd of the paired differences and declared breach no. 1 (PI, 16:55) over
  #100's one-arm-per-call read-out (`sigma_seed.py <json> eng`, TBD-14); #100's
  independent-arms `required_seeds(sigma, p0, n_eval)` with factor 2 is removed.
- MIN_K 8 (not 5), MAX_K 200, sign-test k with q = Phi(margin / sigma_D) (#100 left q
  as an assumption, TBD-13).
- n_eval = 500 and eps_sym = 1/n_eval (PI, 16:55) over #100's TBD-11 (n_eval) and TBD-15.
- `msg_shuffle` is evaluation only, on the trained eng pair (A9). Alternative (#100): an
  arm trained and evaluated under the derangement; not kept, left to the PI.
- The last-5-logs training summary is dropped (A8, confirmed by the PI); #100 kept it as
  a secondary summary.
- Latent blind success capped at B0 (A6); #100 clipped to [0, 1]. The table of
  `nh96-null.md` section 3 is the capped one; #100's table (uncapped) is superseded.
- One level null, `simulate_level_null` (L99 on the level); #100's `simulate_arm_null`
  (same draws minus B0) is removed and its test rewritten on the level null.
- `sym` is descriptive, with its own gate and flag D, and no test; #100's sym leak check
  against its paired null P99 (0 at sigma = 0, so it would fire on any positive D_sym)
  is not kept.
- One decision table (section 5, A7) over #100's prose cases A-D.
- Names: `derange_episodes` (not `episode_derangement`), `sign_test_threshold` raising
  (not `sign_test_critical` returning None).

Tests: all tests of both branches are kept. #100's tests were adapted to the merged
code: `episode_derangement` -> `derange_episodes`; `simulate_arm_null` -> the level
null minus B0; `sign_test_critical(4) is None` -> `sign_test_threshold(4)` raises;
one-arm-per-call -> `READABLE_ARMS` guard. Stricter choices: the paired-narrower test
runs at k = 4 (#100) and k = 8 (#115); a stray eng row at n_eval = 1 now makes the CLI
stop (#100 expected k = MIN_K, because it read nocomm only). The two
`test_sym_null_is_degenerate_at_zero_sigma` are one test holding both sets of assertions.
New: sign-test k reference values (23, 11, 8 at q = 0.80, 0.90, 0.95) and the check that
the paired-mean floor never binds under the normal alternative.

Open after reconciliation: TBD-10, the rest of TBD-11, TBD-12, TBD-16. Section 9 is the
PI record and is not edited.

## 9. PI decisions 2026-11-06 (16:55 CET)

Taken by the PI on the choices listed in the draft PR (#115) after section 8.

| item | decision |
|---|---|
| sigma_D (A4) | CHANGED. sigma_D = sd (ddof = 1) of the paired per-seed differences eng - nocomm on the asym pilot, fed directly to `required_seeds`; the Cauchy-Schwarz bound is dropped. The pairing is a declared blinding breach (section 4, "Pilot read-out (declared)", breach no. 1). MIN_K 8, sign-test table and MAX_K 200 unchanged |
| n_eval (A8) | CONFIRMED without change: n_eval = 500, eps_sym = 1/n_eval |
| A10 | CONFIRMED without change: reading "Procgen does not need the message" stays conditional on an N10-D arm in the N10-C harness, outside this protocol (section 2) |
| Q8 (N10-C-bis vs N10-D) | DEFERRED to the session "point de reprise roadmap v2"; nothing decided here |

No freeze, no tag; the freeze lock (verrou de lecture) is not passed. TBD-10 and the
rest of TBD-11 stay open.
