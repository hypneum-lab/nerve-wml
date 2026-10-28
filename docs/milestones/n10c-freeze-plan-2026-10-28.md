# N10-C — Freeze plan for the configuration of the 2026-10-25 amendment (fix C)

**Status:** PLAN, for the PI's go. Nothing here has been run: no registered pre-training, no GPU job, no tag, no Software Heritage save, no campaign start.
**Governs:** `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-25.md` (section 5, « New freeze, archive and manifest ») and the commitment rules of `docs/milestones/n10c-procgen-multiagent-gtm-comm-amendment-2026-10-23.md` (« Filing », items 1–6). Where this plan and the amendments differ, the amendments govern.
**Date:** 2026-10-28. **Base:** master `ddbe517` (PR #72 merged).

## 0. State at writing

- PR #72 merged: presets, pre-training, liveness guard and restart rule, per-run-pretraining control (seed 101, paired), gate, Adam decision (no bias correction), N10-C-bis alignment.
- Tests: all 516 `tests/unit/n10c` tests pass on macM1 (Metal). On KXKM-AI (MLX-CUDA) the strict reproducibility tests fail in the same way on the frozen tag and on master (backend non-determinism of the conv backward); the three new strict tests of PR #72 are made CUDA-aware on this branch (`tests/unit/n10c/_backend.py`: two skipped on CUDA with a stated reason, and the ratio-of-one assertion of the third not checked on CUDA; Metal and CPU stay strict).
- This branch sets `campaign.guards.REGISTERED_TRAIN_OPTIONS = "fix-c-coin-pretrain"` (amendment section 5, item 2). No registered run can start yet: the trunk pins (`REGISTERED_TRUNK_SHA256`, `REGISTERED_CONTROL_TRUNK_SHA256`) are empty and the new tag does not exist.

## 1. Decisions needed from the PI before the go

1. **How the trunk SHA-256 get under the tag (blocking).** Section 5 requires the trunk SHA-256 « produced by registered pre-training runs on the seed's fixed host » in the stage-2 manifest, which is committed by the tag. But `harness/pretrain.py --registered` requires a `RegistrationContext`, which only the guards issue, and they require the tag. Two ways out:
   - **A (recommended): pre-freeze registered pre-training.** A new mode `pretrain --pre-freeze` runs the exact registered procedure (registered seed or control unit, `PRETRAIN_BUDGET` = 1,000,000 steps, the seed's fixed host checked at runtime, that host's `HOST_SETTINGS`, real Procgen, training levels 0–15, seeding of section 3.2 / 3.4) from a clean tree at a commit pushed to origin (recorded in `pretrain.json`), and ledgers it under a new kind `pretrain_registered` (one per seed / control unit; not a development allowance). Pre-training never touches the coop task, a partner, a message or any held-out level, so running it before the tag reveals no outcome of the estimands. Its SHA-256 are then written into `guards.REGISTERED_TRUNK_SHA256` / `REGISTERED_CONTROL_TRUNK_SHA256` and into the manifest, and the tag commits them. *Code to write:* `--pre-freeze` in `pretrain.py` + `guards.check_pretrain_boundary` branch (no context, strict checks above) + ledger kind + tests.
   - **B: pre-train after the tag (not compatible with the amendment as written).** The pre-trainings would run under the new tag's context, so their SHA-256 cannot be in the tagged manifest, which section 5 item 3 requires and which the Filing rules make immutable after the tag. B therefore needs, before the tag, an **explicit amendment** of the pinning rule (trunk SHA-256 pinned by a later commit touching only the manifest folder, read by `guards` from a separate manifest-folder file, the tagged manifest left unchanged) plus its implementation; its checklist differs from Phase M below. Weaker evidence: the trunks are covered only by that later push time, not by the tag or its SWHID.
   - Either way: **an unusable pre-training** (liveness flag → `usable: false`) needs a rule fixed now. Proposal, mirroring the in-run rule: one restart from scratch (same seed and unit); a second failure stops the freeze and goes back to the PI.
2. **Where the trunk files live.** 16 files of 2.49 MB (6 seeds + 10 control units), 40 MB. **Proposal:** commit them in the freeze commit under `experiments/procgen_multiagent_gtm_comm/manifest/trunks/` so that the tag and the Software Heritage snapshot contain them (the campaign copies them into `<campaign_dir>/trunks/…` and the boundary checks the SHA-256). Alternative: keep them out of git, archived on electron-server and in the run archive, with only the SHA-256 under the tag.
3. **Order with E1-small on KXKM-AI** (section 5 below; proposal: E1-small first).
4. **Tag date:** `n10c-freeze-<YYYYMMDD>` of the day of the push.

## 2. Code changes still to make before the freeze (none run yet)

| # | where | change | why |
|---|---|---|---|
| C1 | `campaign/guards.py` `REGISTRATIONS["n10c"]` | `amendment_path` → the 2026-10-25 amendment; `manifest_name` → a NEW file (`stage2-manifest-fixc.md`) | the freeze must rewrite and check the Status line of the 2026-10-25 amendment, never the FROZEN 2026-10-23 one, and must not overwrite the frozen `stage2-manifest.md` (SHA-256 `1bddf9ca…`, pinned by the old tag) |
| C2 | `campaign/freeze_prepare.py` | resolve **both** the amendment path and the manifest name from the selected registration (it currently writes `guards.AMENDMENT_PATH`, the frozen 2026-10-23 amendment, independently of `REGISTRATIONS`, and defaults to `stage2-manifest.md`); refuse to write a file that exists in a pushed freeze tag; regression test that the frozen 2026-10-23 amendment and `stage2-manifest.md` keep their bytes | same; otherwise C1 alone would still rewrite the frozen amendment while the launch guard checks the 2026-10-25 one |
| C3 | `harness/manifest.py` | new sections: registered preset and its fields; pre-training budget, seeding (shared and per-run), trunk table (seed / unit, host, commit, `pretrain.json` and trunk SHA-256, coin curve, liveness); liveness rule (`MAX_LIVENESS_RESTARTS = 1`); control queue (run ids `prc-*`, paired order, 10 runs); gate and Adam-study results (by reference to `docs/superpowers/research/2026-10-25-n10c-gate/index.json` and its SHA-256); per-host throughput under fix C and the compute estimate (constants below); SHA-256 of both amendments | section 5, item 3 |
| C4 | `harness/manifest.py` constants | `KXKM_PAIR_STEPS_PER_S` and the macM1 planning figures re-measured under fix C (gate runs: KXKM-AI about 1,350 pair-steps/s with micro-batching, 733–747 s per 1M; macM1 about 775 pair-steps/s, 1,291 s per 1M with the day model paused) ; `N_RUNS` / per-host counts include the 10 control runs on KXKM-AI | compute estimate |
| C5 | `harness/pretrain.py`, `campaign/guards.py`, `harness/devledger.py` | option A of decision 1 | trunk pins before the tag |
| C6 | `campaign/macm1/pretrain.sh`, `start.sh` | optional `HEAVY_LOCK=~/scratch/.heavy.lock`: the detached wrapper runs `/usr/bin/lockf -k "$HEAVY_LOCK" zsh day_model_paused.sh …`, so the day model is paused only once the lock is held | macM1 is shared (e1 / Hypneum jobs hold the lock) |
| C7 | `harness/throughput.py` | a fix-C measurement path (preset, trunk, host settings) for the per-host throughput of C4, or take C4 from the gate run records | the throughput helper uses the frozen preset only |
| C8 | `campaign/pull_archive.sh` | defaults to the OLD campaign (`KXKM_SRC=…/n10c-run/campaign`, `DEST=/home/electron/archives/n10c`): change the defaults, or always pass `KXKM_SRC=electron@10.2.0.237:/ai/saisail/n10c-run2/campaign DEST=/home/electron/archives/n10c-fixc` (run ids repeat between the two campaigns, so the destination must differ) | archive |
| C9 | tests | C1–C8 (the campaign runner tests pin the frozen preset; fix-C runner tests are in `test_liveness_control.py`) | |

Note on the registered preset (already on this branch): `harness/train.py`'s CLI default `--preset` follows `REGISTERED_TRAIN_OPTIONS`, so a development invocation of the training CLI without `--preset` now selects fix C and needs `--init-trunk`; direct `train_run()` calls and the smoke / throughput helpers keep the frozen default (`TrainOptions()`).

## 3. Freeze checklist (in order)

**Phase P — preparation (no GPU).**
1. Merge this branch (C-changes above, CUDA-aware tests), CI green, 516+ tests on macM1.
2. PI decisions of section 1 recorded in the 2026-10-25 amendment (section 6).

**Phase T — registered pre-trainings (needs the go; GPU / Metal).** From a clean clone at the merge commit on each host (`/ai/saisail/n10c-freeze/repo`, macM1 `~/scratch/n10c-freeze/repo`), venvs with the pins of `KXKM_VERSIONS` / the macM1 record:

| job | host | command (option A) | estimate |
|---|---|---|---|
| T1 seeds 0, 17, 42 (shared trunks) | macM1 | `HEAVY_LOCK=~/scratch/.heavy.lock BASE=~/scratch/n10c-freeze zsh experiments/procgen_multiagent_gtm_comm/campaign/macm1/pretrain.sh ~/scratch/n10c-freeze/trunks/seed-<s> --seed <s> --pre-freeze` (day model paused by `day_model_paused.sh`, one seed at a time) | 3 × about 10.5 min (615 s per pre-training, rows 4 and 15 of section 4.1) + lock wait |
| T2 seeds 73, 101, 131 (shared trunks) | KXKM-AI | `python -m experiments.procgen_multiagent_gtm_comm.harness.pretrain --seed <s> --steps 1000000 --out /ai/saisail/n10c-freeze/trunks/seed-<s> --pre-freeze` (KXKM settings forced: micro-batch 512, clear_cache) | 3 × about 6.5 min (374 s, row 22) |
| T3 control units of seed 101 (10 own trunks) | KXKM-AI | same with `--per-run <arm> <condition>` for the 10 (arm, condition) of seed 101, out `…/trunks/seed-101-<condition>-<arm>` | 10 × about 6.5 min |

KXKM-AI total about 85 min of GPU, one job at a time, with the lane's wait rules (no other GPU job of ours, E1-small included; ComfyUI tolerated). Each `pretrain.json` must read `usable: true`; the rule of decision 1 applies otherwise.

**Phase M — manifest and freeze commit (macM1, MLX needed for the toolchain record).**
3. Copy the 16 trunks (decision 2) and their `pretrain.json`; write their SHA-256 into `guards.REGISTERED_TRUNK_SHA256` (6) and `REGISTERED_CONTROL_TRUNK_SHA256` (10).
4. `python -m experiments.procgen_multiagent_gtm_comm.campaign.freeze_prepare --tag n10c-freeze-<YYYYMMDD>` (after C1–C3): rewrites the Status line of the 2026-10-25 amendment to `**Status:** FROZEN by tag n10c-freeze-<YYYYMMDD> (stage 1 + stage 2 manifest); OSF registration to follow`, writes `manifest/stage2-manifest-fixc.md` with `Status: FROZEN (tag …)`, prints the manifest SHA-256 and the tag message.
5. Review the manifest (PI) ; commit (subject e.g. `docs(n10c): freeze fix-C configuration`); tag message:

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
6. `git tag -a n10c-freeze-<YYYYMMDD> -F <message>` on the freeze commit; push the commit and the tag to GitHub (`hypneum-lab/nerve-wml`; mirrored to Gitea); record the GitHub push time.
7. Software Heritage « Save Code Now » for the repository; record the SWHID of the tagged revision (`swh:1:rev:<commit>`).
8. OSF registration afterwards, referencing tag, commit and SWHID (optional for the launch).

**Phase L — launch (PI's go).**
9. Fresh campaign directories (the old ones are pinned to `n10c-freeze-20261024` and refuse other inputs): KXKM-AI `/ai/saisail/n10c-run2/{repo,campaign}`, macM1 `~/scratch/n10c-run2/{repo,campaign}`, repos checked out at the tag, trunks copied into `campaign/trunks/` (SHA-256 checked by the boundary).
10. `campaign.json` is written at the first start by the runner with: `freeze_tag`, `freeze_commit`, `swhid`, `manifest_path`, `manifest_sha256`, `amendment_path` (the 2026-10-25 amendment), `amendment_sha256`, `tag_push_time` (or its note), `registration: n10c`, `total_pair_steps: 50000000`, `trial: false`, `host`.
11. Dry run on both hosts (`--dry-run`, no `WOULD REFUSE`), then start: macM1 `HEAVY_LOCK=… BASE=~/scratch/n10c-run2 zsh experiments/procgen_multiagent_gtm_comm/campaign/macm1/start.sh --freeze-tag … --swhid …` (day model paused for the whole runner), KXKM-AI `BASE=/ai/saisail/n10c-run2 bash experiments/procgen_multiagent_gtm_comm/campaign/kxkm/start.sh --freeze-tag … --swhid …`. Runs started before the tag's GitHub push time are excluded.
12. **macM1 → KXKM-AI hand-over.** The KXKM runner does not wait for the macM1 runs: when its own work is done it exits with `EXIT_AWAITING_SYNC` (6). As macM1 runs finish, sync them to the NEW KXKM campaign directory with `python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign sync --campaign-dir ~/scratch/n10c-run2/campaign --to electron@10.2.0.237:/ai/saisail/n10c-run2/campaign` (repeatable), then re-run the KXKM start command: it evaluates the synced runs (never trains them) and runs each cross-play cell once its 6 seeds are evaluated. The campaign is complete only when the KXKM runner exits 0 with all 70 evaluations and the 10 cross-play cells done (`attempts.jsonl` event `complete`).
13. **Archive.** On electron-server: `KXKM_SRC=electron@10.2.0.237:/ai/saisail/n10c-run2/campaign DEST=/home/electron/archives/n10c-fixc bash pull_archive.sh` (repeatable), never the old destination (C8).

## 4. Compute estimate (fix C, from the gate runs; to be replaced by C4)

| host | runs | per run (50M pair-steps) | training | other |
|---|---:|---:|---:|---|
| macM1 (seeds 0, 17, 42) | 30 | about 17.9 h (1,291 s per 1M) | about 538 h ≈ 22.4 days | day model paused throughout |
| KXKM-AI (seeds 73, 101, 131 + 10 control runs of seed 101) | 40 | about 10.3 h (740 s per 1M) | about 412 h ≈ 17.2 days | 70 held-out evaluations ≈ 7.8 h, cross-play 60 passes ≈ 3.4 h |

Liveness restarts (at most one per run) would add up to one run each.

## 5. Proposed order on the shared hosts

1. **macM1 first:** T1 (about 35 min) as soon as the go is given (it only needs the heavy lock), then the macM1 campaign right after the tag: macM1 is the bottleneck (about 22.4 days).
2. **KXKM-AI:** T2 + T3 (about 85 min) in a gap of E1-small, or right before it; then **E1-small (about 3.5 days)**, then the N10-C KXKM campaign (about 17.6 days with evaluations). KXKM then ends after about 21 days, before macM1 (about 22.4 days), so running E1-small first should not delay the end of N10-C — conditional on the E1-small duration (3.5 days, a planning figure) and on the fix-C throughput holding over 50M pair-steps (the gate measured 1M only). The synced macM1 runs are evaluated on KXKM-AI after the hand-over of step 12.
3. **N10-C-bis** after the N10-C KXKM training (strict serialisation, addendum decision 6): its freeze needs the new N10-C tag first (addendum A7.6: its tag must contain the N10-C training options and trunk pins), its own stage-2 manifest (`stage2-manifest-bis.md`, with the B3 gate results and the B5 `vq` fix), its smoke / throughput runs under fix C, then its tag `n10c-bis-freeze-<YYYYMMDD>`; 48 runs on KXKM-AI ≈ 21 days.

## 6. What this plan does not change

The 2026-10-23 amendment and the frozen `stage2-manifest.md` stay byte-identical (C1, C2). The runs of `n10c-freeze-20261024` stay excluded. No result of a full-length run is read before the new tag.
