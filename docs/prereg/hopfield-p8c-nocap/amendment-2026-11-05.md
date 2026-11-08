# P8c-nocap — Amendment 2026-11-05: stop of the N=16384 arm

**Study id:** `hopfield-p8c-nocap`. **Frozen at:** tag
`prereg-hopfield-p1bis-p8c-freeze-20261104` (commit 257e469, SWHID
`swh:1:rev:257e4698cadd6786d2381cf61d2eef4cdbeb4475`).
**Amendment signed by:** Clément Saillant (PI), decision of 2026-11-05
(about 20:10 CET). Drafted by an agent. The frozen prereg file is not
modified by this amendment.

**L2 deposit (annotation added 2026-11-08):** restricted Zenodo record
[10.5281/zenodo.24053908](https://doi.org/10.5281/zenodo.24053908) (Hypneum Lab lot of
2026-11-08, file `prereg-hopfield-p8c-relaunch-20261105.zip`, SHA-256
`b561ce5b4a8b40b1688113be2e2212f3c5921fea4052dc613cfb1f3937b6fb75`) gives the freeze tag
`prereg-hopfield-p8c-relaunch-20261105` (commit `db4eb5ba5644df8dc5622f53f249c19d52a9d82b`,
`swh:1:rev:db4eb5ba5644df8dc5622f53f249c19d52a9d82b`) a verifiable deposit date, level
L2 of PROJECT-MANAGEMENT section 4. The deposit holds this file as it was at the tag;
anything dated after the tag in this live file is not in the deposit.

## 1. Decision

On 2026-11-05 the PI decided to stop the N=16384, alpha=0.5 arm of
P8c-nocap. The STOP file was touched at 2026-11-05T19:05:21Z (`STOP`,
`STOP.when`, line in `logs/campaign.log`). Wave 2 (seeds about
32009-32016) was interrupted; the four cells of wave 3 that the pool had
already started exit on their own at the first STOP test of the HU loop.

## 2. Cause

The 18 h Hopfield-unlearning ceiling (`--max-stoch-secs`) is too short at
N=16384. Throughput was about 8.7 states/s with 8 workers in parallel,
against 18.5 states/s in an isolated probe. Wave 1 (seeds 32001-32008,
finished 2026-11-05 12:33-12:48Z) ended with all 8 cells in status
`ceiling`, with r reached between 0.753 and 0.767. The HU grid points
r=0.8 and r=0.9 are therefore `not_measured` for these 8 cells; the 8
matched kernel points were measured.

Information source: process fields only (`r_reached`, `status`,
`D_steps`). No recovery score of N=16384 cells was read before this
decision or at the time of writing.

## 3. Affected cells

- Seeds 32001-32008: finished, status `ceiling`, kept as recorded
  (nothing deleted, nothing re-run).
- Seeds 32009-32020: not finished (interrupted or never started).
- N=4096 arm (60 cells, alpha 0.3/0.4/0.5, seeds 32001-32020): complete
  since 2026-11-04T21:37:31Z, not affected.

## 4. Consequences under the frozen rules

By the frozen rule, a seed enters the primary only if its 8 kernel points
and its 8 HU grid points are all measured; the test is analysed only with
at least 16 of 20 seeds, otherwise it is `not_measured`. At N=16384 no
seed satisfies the condition (grid points at r=0.8 and 0.9 are not
measured for the 8 finished seeds, and 12 seeds are not computed). The
primary and every test that requires N=16384 are therefore reported
`not_measured`. This is a statement of non-measurement, not a null result
and not a refutation. Tests at N=4096 are analysed per the frozen
analysis plan.

## 5. Options discarded

- Raise the ceiling to 30 h and re-run everything: about 3.5 days of
  additional compute, discarded by the PI.
- Let the arm finish: the remaining cells would hit the same ceiling at
  the same r, producing no additional measured grid point.

## 6. Commitment

No exploratory analysis at N=16384 (for example a grid limited to
r<=0.7) is performed in the present analysis. If one is done later, it
will be labelled exploratory and declared as such, with the date and the
reason; it cannot change the status of the frozen primary.

## 7. Documentary corrections (no change to the tagged prereg)

- The tagged `docs/prereg/hopfield-p8c-nocap/prereg.md` still carries the
  header "L0 draft — NOT frozen". The freeze is established by the tag
  above and not by that header; the header is a documentary
  inconsistency, signalled here and left untouched.
- `results/` is missing `run.log` and `SHA256SUMS` for P1bis. This is
  signalled as a documentary gap; the P1bis cell files exist and their
  checksums will be recorded at analysis time.
