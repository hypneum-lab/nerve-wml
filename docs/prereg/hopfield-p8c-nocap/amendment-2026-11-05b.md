# P8c-nocap — Amendment 2026-11-05b: relaunch of the N=16384 arm

**Study id:** `hopfield-p8c-nocap`. **Original freeze:** tag
`prereg-hopfield-p1bis-p8c-freeze-20261104` (commit 257e469, SWHID
`swh:1:rev:257e4698cadd6786d2381cf61d2eef4cdbeb4475`).
**Amendment signed by:** Clément Saillant (PI), explicit order of
2026-11-05 20:18 CET ("lance les mesures avec un plus grand plafond").
Drafted by an agent. The frozen prereg file is not modified.

## 1. Decision and relation to the earlier amendment

The PI orders the N=16384, alpha=0.5 arm of P8c-nocap to be run again
with a larger Hopfield-unlearning (HU) ceiling. This **partially
supersedes** `amendment-2026-11-05.md` (stop of the N=16384 arm): its
sections 1-3 (facts) stand; its section 4 (arm reported
`not_measured`) and section 5 (discarded options) are replaced by this
decision. Its section 6 (no exploratory analysis at N=16384) still
applies. The N=4096 arm is untouched.

## 2. Cause (process information only)

The 18 h ceiling was too short at N=16384: all 8 cells of wave 1
(seeds 32001-32008) ended in status `ceiling` at r between 0.753 and
0.767, so the HU grid points r=0.8 and r=0.9 were not measured. The
cause is a compute budget that was set too low, not a property of the
hypothesis. No recovery score of an N=16384 cell was read.

## 3. New parameters (the only changes)

| Parameter | Frozen value | New value |
|---|---|---|
| HU ceiling per cell (`--max-stoch-secs`) | 64800 s (18 h) | 129600 s (36 h) |
| Wall cap, lane A | 72 h | 120 h |
| Workers, lane A | 8 | 8 (unchanged) |

Rationale for 36 h: about 700-800 k states are needed, about 22-26 h at
the observed 8.7 states/s; the rest is margin for slower E cores.
Expected duration: 3 waves (8 + 8 + 4 cells), each at most 36 h, so
about 3 to 4.5 days.

Lane B is **not** relaunched (N=4096 and P1bis are complete).

## 4. Unchanged

Seeds (confirmatory 32001-32020), alpha=0.5, N=16384, scientific code
(`p8c_nocap.py`, `analysis.py`, `stats.py`, vendored Hopfield code),
the hash-pinned environment, thresholds, tests and the frozen analysis
plan. The launcher script is not modified: the run is started directly
with the same module and options, lane A only, with the two parameters
above set on the command line and a separate wall-clock guard.

## 5. Cells of the first attempt

The 8 `ceiling` cells of attempt 1 (seeds 32001-32008, `cells.jsonl`)
are archived unread (`results/hopfield_p8c_nocap/n16384-try1-ceiling18h`,
log `logs/p8c_n16384-try1.log`) and **excluded** from the analysis.
They are replaced by the new run of all 20 seeds, in a fresh directory.
Nothing is deleted.

## 6. Reproducibility control

The rows of seeds 32001-32008 for the points with r <= 0.7 and for the
matched kernel point are deterministic and must be identical between
attempt 1 and the relaunch. The control is an equality test only
(byte or value equality of the recorded rows). No score is displayed or
interpreted. A mismatch is reported as a deviation and does not change
the frozen rules.

## 7. Declaration on blindness

The decision to relaunch was taken **after** the N=4096 scores had been
read (frozen analysis, 2026-11-05T19:12:45Z). The relaunch is therefore
**not blind with respect to N=4096**. It **is blind with respect to
N=16384**: no N=16384 score has ever been read. The ceiling and wall cap
were chosen from process quantities (throughput, states needed) only.

## 8. Freeze procedure and declared deviation

The freeze of this amendment is an annotated tag
`prereg-hopfield-p8c-relaunch-20261105` on the head commit of the pushed
branch `docs/p8c-amend-relaunch-n16384`, with its SWHID, created
**before** the relaunch. The pull request is merged afterwards by the
PI. **Declared deviation:** the run starts from a commit that is pushed
and tagged but not yet merged into `master`. The tag commit contains the
same scientific code as the original freeze (only this document is
added).

## 9. Substrate split considered and not adopted

The PI considered running seeds 32017-32020 on macM1 (arm64, Python
3.12.13, same hash-pinned wheels) and seeds 32001-32016 on kxkm-ai
(x86_64, Python 3.12.15), conditional on bit-identical output. Test:
smoke cell seed 99902, N=2048, alpha=0.3 (not a confirmatory seed), same
code commit on both hosts, all rows compared except the fields `secs`,
`stoch_secs`, `host`, `pid`, `ts`. Result: **0 of 26 rows identical**
across platforms (differences at the 1e-7 to 1e-8 relative level in
floating-point fields). On kxkm-ai the rerun is identical to the smoke
reference of 2026-11-04 (26 of 26 rows). The condition is not met, so
**no split is made**: all 20 seeds run on kxkm-ai, one substrate.
Files (sha256 including timing fields): kxkm-ai
`5b1ea8fa2afeeb465b3791346510db15ad36b36bf71152147123631719bd57ed`,
macM1
`daa99b4c31f81f5431d504072ba7b4a87070ff2a44a8382a24fae4d1cd32b746`.

## End record (2026-11-08)

Data record only. No analysis, no verdict and no interpretation is made here.

- End: 2026-11-08T04:43:09Z (`logs/campaign.log`: `STOP laneA(relaunch) rc=0`, then
  `END relaunch`). Host: kxkm-ai.
- Tag `prereg-hopfield-p8c-relaunch-20261105`, commit
  `db4eb5ba5644df8dc5622f53f249c19d52a9d82b`.
- Cells: 520 lines = 20 seeds (32001-32020) x 26 lines, N=16384, alpha=0.5; 20 of 20 seeds
  complete (`D_steps` 750830 to 750884).
- sha256 of the uncompressed `cells.jsonl`:
  `5b43fb46eb2f3c0dae61b56d65011f0487690db229e3c1fd9d20804d80a12a76`. The repository stores it
  gzipped, as for `results/nh01*`: `results/hopfield_p8c_nocap/n16384/cells.jsonl.gz`
  (sha256 `f998cb2765ebbf9ca13dffbc402ebe8b1d7e8c7fc904a0c920938012ac254edc`), with
  `run_info.json` and `SHA256SUMS` (which lists both hashes).
- Reproducibility control, try1 (`n16384-try1-ceiling18h`, ceiling 18 h) against this relaunch,
  as defined in `RELAUNCH.md` (equality only, lines with r <= 0.7 and matched rows): 168 of
  168 lines with r <= 0.7 identical; matched rows 56 of 56 identical.
- Duration note: seeds 32017-32020 ran about 48 000 s each, against 77 000 to 81 000 s for
  the other seeds, because wave 3 had 4 parallel tasks instead of 8.
- Archive on electron-server: `~/hypneum-archives/p1bis-p8c-20261104-20261108/`
  (`results/hopfield_p8c_nocap/n16384/`, `logs/campaign.log`, `logs/p8c_n16384.log`,
  `RELAUNCH.md`).

## Pre-registered read-out (2026-11-08)

Data record and application of the pre-registered analysis plan only. PI
decision of 2026-11-08: the results table and the pre-registered label are
recorded as the pre-registration defines them; no new dose, no relaunch and
no statement on a hypothesis beyond the label assigned by the
pre-registration's own decision rule are made here.

### Analysis run

- Command: `python -m experiments.hopfield_dream_conf.analyse p8c <input_dir>`
  (frozen code, no modification), run from the checkout of commit
  `db4eb5ba5644df8dc5622f53f249c19d52a9d82b` (tag
  `prereg-hopfield-p8c-relaunch-20261105`; abbreviated `db4eb5b`).
- Host macM1, nice -n 10, 2026-11-08T09:38:10Z to 09:38:20Z. Python 3.12.13
  arm64, numpy 2.5.4, scipy 1.18.1. The archive run used Python 3.12.15
  x86_64; floating-point fields may differ at about 1e-7 (see section 9).
- Inputs (sha256): N=16384 `cells.jsonl`
  `5b43fb46eb2f3c0dae61b56d65011f0487690db229e3c1fd9d20804d80a12a76`;
  N=4096 `cells.jsonl`
  `04556ee957bfb9b82a24e4681851711f92569f24f3f04a1a15a8f937003fbfd0`.
- Output: `results/hopfield_p8c_nocap/n16384/verdict.json` (copy of the macM1
  output, listed in `SHA256SUMS`). Every number below is read from it.

### Gate (prereg.md l.177-181)

At N=4096, alpha=0.3, q=0.1: mean kernel recovery on B at t-hat = 1.000
(threshold >= 0.90); mean Hebb recovery on B = 0.000 (threshold <= 0.05).
Gate `pass` = true.

### Primary family (8 tests, n = 20 seeds each)

Delta = kernel recovery on B at t-hat minus HU recovery on B at r-hat
(cross-fitted, prereg.md l.134-144). "kernel B" and "HU B" are the mean
recoveries on set B of the two selected arms. Holm step-down within the
family at one-sided level 0.025; exact sign-flip permutation tests;
bootstrap 10 000 resamples, seed 20261104.

| N | alpha | q | mean Delta | sd | boot 95 % | p_sup (Holm) | p_inf (Holm) | kernel B | HU B | label |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 0.3 | 0.2 | 0.0000 | 0.0065 | [-0.0023, 0.0033] | 1 | 7.63e-06 | 0.9988 | 0.9988 | `below` |
| 4096 | 0.3 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |
| 4096 | 0.4 | 0.2 | 0.2925 | 0.0319 | [0.2790, 0.3058] | 7.63e-06 | 1 | 0.3077 | 0.0153 | `success` |
| 4096 | 0.4 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |
| 4096 | 0.5 | 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |
| 4096 | 0.5 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |
| 16384 | 0.5 | 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |
| 16384 | 0.5 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | 0.0000 | 0.0000 | `floor` |

The four N=4096 rows are produced by the same frozen code on the N=4096
cells already analysed on 2026-11-05; the N=4096 alpha=0.4, q=0.2 row has
mean Delta = 0.2925.

### Labels under the pre-registered rule

Rule as written in `prereg.md` (l.167-172):

> - `success`: mean Δ ≥ 0.10 and Holm-adjusted p_sup < 0.025;
> - `below`: Holm-adjusted p_inf < 0.025 (advantage established < 0.10;
>   includes "HU equals or beats the kernel");
> - `floor`: not `success` and both selected arms have mean recovery on B
>   < 0.05 (no arm recovers; uninformative, replaces `below`);
> - `inconclusive`: otherwise.

Confirmatory N=16384 cells (alpha=0.5):

- q=0.2: mean Delta = 0, kernel B = 0, HU B = 0. Not `success`; both
  selected arms have mean recovery on B < 0.05. Label: `floor`.
- q=0.3: mean Delta = 0, kernel B = 0, HU B = 0. Not `success`; both
  selected arms have mean recovery on B < 0.05. Label: `floor`.

In both cells t-hat = 0.5 and r-hat = 0.2 for all 20 seeds (the smallest
grid values, selected by the tie rule of l.137-138 when every recovery on A is 0).

Decision rule as written in `prereg.md` (l.184-192):

> - **Avantage du noyau soutenu dans les cellules {…}:** gate passes and at
>   least one primary test is `success`; the cells are listed; no
>   generalisation beyond them.
> - **Avantage du noyau non soutenu:** gate passes, no `success`, and every
>   primary test that is not `floor` is `below` (at least one such test).
> - **Non concluant:** otherwise (including all-`floor` or any
>   `not_measured`/`inconclusive` primary test without a `success`).

Code output, field `overall_primary`: `supported_in_cells:N4096_a0.4_q0.2`.
This string is emitted by the frozen code over the 8-test family; the only
`success` row is the N=4096, alpha=0.4, q=0.2 row listed above. The two
N=16384 cells carry the label `floor` and contribute no `success` and no
`below`.

### Secondary contrasts

S1, matched budget without cap (q in {0.2, 0.3}; floor rule not applied
for S1, prereg.md l.174):

| N | alpha | q | mean Delta | sd | boot 95 % | p_sup (Holm) | p_inf (Holm) | label |
|---|---|---|---|---|---|---|---|---|
| 4096 | 0.3 | 0.2 | 0.5762 | 0.0456 | [0.5560, 0.5950] | 7.63e-06 | 1 | `success` |
| 4096 | 0.3 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |
| 4096 | 0.4 | 0.2 | 0.3075 | 0.0344 | [0.2930, 0.3217] | 7.63e-06 | 1 | `success` |
| 4096 | 0.4 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |
| 4096 | 0.5 | 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |
| 4096 | 0.5 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |
| 16384 | 0.5 | 0.2 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |
| 16384 | 0.5 | 0.3 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 7.63e-06 | `below` |

S2, q=0.1 best vs best (4 tests):

| N | alpha | q | mean Delta | sd | boot 95 % | p_sup (Holm) | p_inf (Holm) | label |
|---|---|---|---|---|---|---|---|---|
| 4096 | 0.3 | 0.1 | 0.0003 | 0.0011 | [0.0000, 0.0008] | 1 | 3.81e-06 | `below` |
| 4096 | 0.4 | 0.1 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 3.81e-06 | `below` |
| 4096 | 0.5 | 0.1 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 3.81e-06 | `below` |
| 16384 | 0.5 | 0.1 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 1 | 3.81e-06 | `below` |

At N=16384, alpha=0.5, q=0.1 the selected arms are t-hat = 3 and r-hat = 0.7
for all 20 seeds, with mean recovery on B of 1.000 for both the kernel and HU
(`verdict.json`, `selection_detail`).

### Exploratory dose curves (exploratory, descriptive, not pre-registered)

Source: `n16384_curves_process.txt` (macM1, 2026-11-08), N=16384, alpha=0.5,
20 seeds, mean recovery on set B.
Process check: 20 of 20 seeds `complete`, D_steps 750830 to
750884, `cue_nonconv` total 0, Hebb mean recovery on B 0.0 at q=0.1, 0.2 and 0.3.

Kernel, by t:

| q | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 | 10.0 | inf |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 0 | 0 | 0.7505 | 1.0 | 1.0 | 1.0 | 0.9922 |
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

HU, by r:

| q | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 0 | 0 | 0 | 0.5685 | 1.0 | 1.0 | 0.9788 |
| 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

These curves are descriptive; no test, label or threshold is attached to them.

### Not done

- No new dose, no new threshold and no relaunch is proposed or run.
- No statistic other than those of the frozen `analyse p8c` code and the
  descriptive curves above was computed.
- The equality check of try1 against the relaunch is the one recorded in the
  End record above (168 of 168 and 56 of 56 identical); it was not re-run.
- The archive run (x86_64) of the analysis was not repeated; this read-out
  comes from the macM1 run only.
- No interpretation beyond the pre-registered labels is made in this section.
