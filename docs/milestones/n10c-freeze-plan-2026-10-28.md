# N10-C — Freeze plan for the configuration of the 2026-10-25 amendment (fix C)

**Status:** PLAN, decided by the PI (2026-10-28, about 18:00), code implemented; waiting for the PI's go of phase T. Nothing here has been run: no registered pre-training, no GPU job, no tag, no Software Heritage save, no campaign start.
**Governs:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (section 5, « New freeze, archive and manifest ») and the commitment rules of `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (« Filing », items 1–6). Where this plan and the amendments differ, the amendments govern.
**Date:** 2026-10-28. **Base:** master `ddbe517` (PR #72 merged).

## 0. State at writing

- PR #72 merged: presets, pre-training, liveness guard and restart rule, per-run-pretraining control (seed 101, paired), gate, Adam decision (no bias correction), N10-C-bis alignment.
- Tests: all `tests/unit/n10c` tests pass on macM1 (Metal; 539 with this branch). On KXKM-AI (MLX-CUDA) the strict reproducibility tests fail in the same way on the frozen tag and on master (backend non-determinism of the conv backward); the three new strict tests of PR #72 are made CUDA-aware on this branch (`tests/unit/n10c/_backend.py`: two skipped on CUDA with a stated reason, and the ratio-of-one assertion of the third not checked on CUDA; Metal and CPU stay strict).
- `campaign.guards.REGISTERED_TRAIN_OPTIONS = "fix-c-coin-pretrain"` (amendment section 5, item 2). No registered run can start yet: the trunk pins (`campaign/trunk_pins.py`) are empty and the new tag does not exist.

## 1. PI decisions (2026-10-28, about 18:00)

1. **Trunk SHA-256 under the tag: option A, pre-freeze registered pre-training.** `harness.pretrain --pre-freeze` runs the exact registered procedure before the tag: a registered seed (or a control unit of seed 101), exactly `PRETRAIN_BUDGET` = 1,000,000 steps, the seed's fixed host checked at runtime and that host's `HOST_SETTINGS` forced, real Procgen, training levels 0–15, the seeding of sections 3.2 / 3.4, from a clean tree whose commit is on a branch of origin (the commit, host, preset and budget are written in `pretrain.json`, field `pre_freeze`; only the development ledger may differ from the commit). It is ledgered under the kind `pretrain_registered`, not a development allowance. Outcome-blind: pre-training never touches the coop task, a partner, a message or any held-out level. *(Option B — pinning after the tag — would have needed an amendment of the pinning rule; not adopted.)*
2. **An unusable pre-training** (a flagged liveness probe → `usable: false`, or an infrastructure crash) gets **one restart**; after a second failure the freeze stops and the PI decides (the ledger refuses a third attempt and any attempt once a usable trunk exists).
3. **The 16 trunk files (6 seeds + 10 control units, about 2.5 MB each, 40 MB) are committed in the freeze commit** under `experiments/procgen_multiagent_gtm_comm/manifest/trunks/<name>/` (`trunk.safetensors` + `pretrain.json`), so that the tag and its Software Heritage snapshot contain them.
4. **Order:** macM1: T1 pre-trainings, then its campaign right after the tag; KXKM-AI: T2 + T3 pre-trainings, then E1-small (about 3.5 days), then the N10-C campaign; N10-C-bis frozen after the N10-C tag (section 5).
5. **Tag:** `n10c-freeze-<YYYYMMDD>` of the day of the push.

## 2. Code changes (implemented on this branch)

| # | where | change |
|---|---|---|
| C1 | `campaign/guards.py` | `REGISTRATIONS["n10c"]`: `amendment_path` = the 2026-10-25 amendment (`AMENDMENT_PATH`), `manifest_name` = `stage2-manifest-fixc.md` (`MANIFEST_NAME`). The files frozen by `n10c-freeze-20261024` are `FROZEN_PATHS` (the 2026-10-23 amendment, `stage2-manifest.md`, SHA-256 `1bddf9ca…`). |
| C2 | `campaign/freeze_prepare.py` | takes the amendment path and the manifest name from the selected registration (`--registration`, default `n10c`), refuses any path of `FROZEN_PATHS` and a tag of another registration; test: the frozen files keep their bytes. |
| C3 | `harness/manifest.py` | new section 5 (fix C): SHA-256 of both amendments; the registered preset and its fields; Adam decision; liveness guard and restart rule; pre-training procedure; **pinned-trunk table** (16 rows: host, procedure commit, SeedSequence, trunk and `pretrain.json` SHA-256, coin rate; a FROZEN manifest refuses to be written unless all 16 trunk files, records and pins agree); the paired control runs; fix-C throughput and compute; the gate / Adam index and its SHA-256. The header names the 2026-10-25 amendment and the archive paths of the new campaign. |
| C4 | `harness/manifest.py` constants | `FIXC_PAIR_STEPS_PER_S` (KXKM-AI 1,000,000 / 740 s ≈ 1,351, macM1 1,000,000 / 1,291 s ≈ 775, from the gate records), `FIXC_RUNS_PER_HOST` (KXKM-AI 40 = 30 + 10 control, macM1 30), 70 evaluations; the section-4 throughput of the frozen configuration is marked as superseded for planning. |
| C5 | `harness/pretrain.py`, `campaign/guards.py` (`check_prefreeze_pretrain`), `harness/devledger.py` (`pretrain_registered`, 2 attempts) | option A and the restart rule (decisions 1–2). |
| C5b | `campaign/pin_trunks.py`, `campaign/trunk_pins.py` | checks each pre-freeze record (usable, pre-freeze procedure, the seed's host and settings, exact budget, seed / unit, SHA-256 of the file), copies it into `manifest/trunks/<name>/` (decision 3) and rewrites `campaign/trunk_pins.py` (code, read by `guards`, so it cannot change after the tag without failing the launch guard). |
| C6 | `campaign/macm1/day_model_paused.sh` (+ `pretrain.sh`, `start.sh`) | `HEAVY_LOCK=~/scratch/.heavy.lock`: the detached wrapper runs under `/usr/bin/lockf -k "$HEAVY_LOCK"` and pauses the day model only once the lock is held; while another job holds it, the launcher reports `queued`. |
| C7 | `harness/throughput.py` | fix-C measurement path: `--preset`, `--init-trunk`, `--host-settings {kxkm,macm1}`, `--learning`. |
| C8 | `campaign/pull_archive.sh` | defaults to the new campaign (`n10c-run2/campaign` → `archives/n10c-fixc`); the stopped campaign stays in `archives/n10c`. |
| C9 | tests | `tests/unit/n10c/test_freeze_prep.py` (C1, C2, C5, C5b, lanes), `test_manifest.py` (C3, C4, C7), `test_macm1_launcher.py` (C6). |
| — | `campaign/kxkm/gpu_wait.sh`, `pretrain_lane.sh`, `gate_bis_lane.sh` | KXKM wait rules shared by the lanes: busy only while a GPU job of ours is alive (an N10-C harness / campaign process, an E1-small job, one of our tmux lanes) or the card has less than 9.5 GB free. **Foreign GPU jobs are never touched and count only through free memory** (ComfyUI pid 712018; nh81's `measure.py`, pid 1095192, about 386 MiB, which runs from our venv but is not an N10-C process). `pretrain_lane.sh` runs T2 + T3 with the restart rule. |

Note on the registered preset: the training CLI's default `--preset` follows `REGISTERED_TRAIN_OPTIONS`, so a development invocation of `harness.train` without `--preset` now selects fix C and needs `--init-trunk`; direct `train_run()` calls and the smoke / throughput helpers keep the frozen default (`TrainOptions()`).

## 3. Freeze checklist (in order)

**Phase P — preparation (no GPU).**
1. Merge this branch; CI green; all n10c tests on macM1 (under the heavy lock).
2. Fresh clones at the merge commit, pushed to origin: KXKM-AI `/ai/saisail/n10c-freeze/repo`, macM1 `~/scratch/n10c-freeze/repo` (clean; the pre-freeze gate refuses a dirty tree or a commit not on origin). Venvs: the N10-C pins (KXKM-AI `/ai/saisail/n10c-run/.venv`, macM1 the MLX 0.31.1 venv used for the campaign), passed as `PY=`.

**Phase T — pre-freeze registered pre-trainings (needs the PI's go).**

| job | host | command | estimate |
|---|---|---|---|
| T1 seeds 0, 17, 42 | macM1 | for s in 0 17 42: `HEAVY_LOCK=~/scratch/.heavy.lock BASE=~/scratch/n10c-freeze PY=<macM1 venv>/bin/python zsh experiments/procgen_multiagent_gtm_comm/campaign/macm1/pretrain.sh ~/scratch/n10c-freeze/trunks/seed-$s --seed $s --steps 1000000 --pre-freeze` (each waits for the lock, then pauses the day model; launched one after the other) | 3 × about 10.5 min (615 s, rows 4 and 15 of section 4.1) + lock waits |
| T2 + T3 seeds 73, 101, 131 and the 10 control units of seed 101 | KXKM-AI | `REPO=/ai/saisail/n10c-freeze/repo OUT=/ai/saisail/n10c-freeze/trunks nohup bash /ai/saisail/n10c-freeze/repo/experiments/procgen_multiagent_gtm_comm/campaign/kxkm/pretrain_lane.sh </dev/null >/dev/null 2>&1 &` (one at a time, `gpu_wait.sh` before each, one restart of an unusable trunk, then stop) | 13 × about 6.5 min (374 s, row 22) ≈ 85 min of GPU |

Each `pretrain.json` must read `usable: true` and carry the `pre_freeze` record; otherwise decision 2 applies.

**Phase M — freeze commit (macM1).**
3. Merge the `pretrain_registered` ledger entries of both host working copies into the committed `manifest/smoke_ledger.json` by run id (as for the gate runs).
4. Copy the 16 output directories to macM1 and pin them: `python -m experiments.procgen_multiagent_gtm_comm.campaign.pin_trunks <16 dirs>` (copies into `manifest/trunks/`, rewrites `campaign/trunk_pins.py`; exit 0 only when nothing is missing).
5. `python -m experiments.procgen_multiagent_gtm_comm.campaign.freeze_prepare --tag n10c-freeze-<YYYYMMDD>` on macM1 (MLX needed for the toolchain record): rewrites the Status line of the 2026-10-25 amendment to `**Status:** FROZEN by tag n10c-freeze-<YYYYMMDD> (stage 1 + stage 2 manifest); OSF registration to follow`, writes `manifest/stage2-manifest-fixc.md` with `Status: FROZEN (tag …)` (refused unless the 16 trunks agree with their pins), prints the manifest SHA-256 and the tag message.
6. PI review of the manifest; commit (trunks, pins, ledger, amendment, manifest; subject e.g. `docs(n10c): freeze fix-C configuration`); tag message:

   ```
   freeze n10c-freeze-<YYYYMMDD>

   N10-C, amendment 2026-10-25 (training fix C: own-coin reward,
   pre-trained trunk, Adam eps 1e-5, finished-agent mask, liveness
   guard with one restart). Supersedes tag n10c-freeze-20261024,
   whose campaign was stopped and is excluded.

   manifest: experiments/procgen_multiagent_gtm_comm/manifest/stage2-manifest-fixc.md
   manifest-sha256: <64 hex from freeze_prepare>
   ```

**Phase F — commitment (PI's final go only).**
7. `git tag -a n10c-freeze-<YYYYMMDD> -F <message>` on the freeze commit; push the commit and the tag to GitHub (`hypneum-lab/nerve-wml`; mirrored to Gitea); record the GitHub push time.
8. Software Heritage « Save Code Now »; record the SWHID of the tagged revision (`swh:1:rev:<commit>`).
9. OSF registration afterwards, referencing tag, commit and SWHID (optional for the launch).

**Phase L — launch (PI's go).**
10. Fresh campaign directories (the old ones are pinned to `n10c-freeze-20261024` and refuse other inputs): KXKM-AI `/ai/saisail/n10c-run2/{repo,campaign}`, macM1 `~/scratch/n10c-run2/{repo,campaign}`, repos checked out at the tag; copy `manifest/trunks/` of the tag into `campaign/trunks/` (same layout; the boundary checks each SHA-256 against `trunk_pins.py`).
11. `campaign.json` is written at the first start by the runner with: `freeze_tag`, `freeze_commit`, `swhid`, `manifest_path`, `manifest_sha256`, `amendment_path` (the 2026-10-25 amendment), `amendment_sha256`, `tag_push_time` (or its note), `registration: n10c`, `total_pair_steps: 50000000`, `trial: false`, `host`.
12. Dry run on both hosts (`--dry-run`, no `WOULD REFUSE`), then start: macM1 `HEAVY_LOCK=~/scratch/.heavy.lock BASE=~/scratch/n10c-run2 zsh experiments/procgen_multiagent_gtm_comm/campaign/macm1/start.sh --freeze-tag … --swhid …` (the runner then holds the heavy lock and the day-model pause for the whole campaign, about 22 days: other heavy macM1 jobs wait — a PI choice; without `HEAVY_LOCK` they would run next to it), KXKM-AI `BASE=/ai/saisail/n10c-run2 bash experiments/procgen_multiagent_gtm_comm/campaign/kxkm/start.sh --freeze-tag … --swhid …` after E1-small. Runs started before the tag's GitHub push time are excluded.
13. **macM1 → KXKM-AI hand-over.** The KXKM runner does not wait for the macM1 runs: when its own work is done it exits with `EXIT_AWAITING_SYNC` (6). As macM1 runs finish, sync them to the NEW KXKM campaign directory with `python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign sync --campaign-dir ~/scratch/n10c-run2/campaign --to electron@10.2.0.237:/ai/saisail/n10c-run2/campaign` (repeatable), then re-run the KXKM start command: it evaluates the synced runs (never trains them) and runs each cross-play cell once its 6 seeds are evaluated. The campaign is complete only when the KXKM runner records the event `complete` in `campaign/attempts.jsonl` (all 70 evaluations and the 10 cross-play cells done); the detached launcher's own exit status does not show it.
14. **Archive.** On electron-server: `bash <repo>/experiments/procgen_multiagent_gtm_comm/campaign/pull_archive.sh` (defaults: `n10c-run2/campaign` → `archives/n10c-fixc`, C8), repeatable; never the old destination.

## 4. Compute estimate (fix C, planning figures from the gate runs; manifest section 5)

| host | runs | per run (50M pair-steps) | training | other |
|---|---:|---:|---:|---|
| macM1 (seeds 0, 17, 42) | 30 | about 17.9 h (1,291 s per 1M) | about 538 h ≈ 22.4 days | day model paused throughout |
| KXKM-AI (seeds 73, 101, 131 + 10 control runs of seed 101) | 40 | about 10.3 h (740 s per 1M) | about 412 h ≈ 17.2 days | 70 held-out evaluations ≈ 7.8 h, cross-play 60 passes ≈ 3.4 h |

Liveness restarts (at most one per run) would add up to one run each.

## 5. Order on the shared hosts (PI decision 4)

1. **macM1:** T1 (about 35 min plus lock waits) at the go of phase T; its campaign right after the tag: macM1 is the bottleneck (about 22.4 days).
2. **KXKM-AI:** T2 + T3 (about 85 min) at the go of phase T (the lane waits for E1-small and our other jobs, tolerating foreign ones); then **E1-small (about 3.5 days)**; then the N10-C KXKM campaign (about 17.6 days with evaluations). KXKM then ends after about 21 days, before macM1 (about 22.4 days), so running E1-small first should not delay the end of N10-C — conditional on the E1-small duration (3.5 days, a planning figure) and on the fix-C throughput holding over 50M pair-steps (the gate measured 1M only). The synced macM1 runs are evaluated on KXKM-AI after the hand-over of step 13.
3. **N10-C-bis** after the N10-C KXKM training (strict serialisation, addendum decision 6): its freeze needs the new N10-C tag first (addendum A7.6: its tag must contain the N10-C training options and trunk pins), its own stage-2 manifest (`stage2-manifest-bis.md`; no writer yet — `freeze_prepare` refuses the `n10c-bis` registration until one exists), its smoke / throughput runs under fix C, then its tag `n10c-bis-freeze-<YYYYMMDD>`; 48 runs on KXKM-AI ≈ 21 days.

## 6. What this plan does not change

The 2026-10-23 amendment and the frozen `stage2-manifest.md` stay byte-identical (C1, C2, `FROZEN_PATHS`). The runs of `n10c-freeze-20261024` stay excluded. No result of a full-length run is read before the new tag.
