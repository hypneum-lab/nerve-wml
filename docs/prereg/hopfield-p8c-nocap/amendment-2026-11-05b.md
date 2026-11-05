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
