# N10-C training campaign runbook (option D, 60 runs, two hosts)

**Starting the campaign requires the PI's explicit go. Nothing in this
directory has been run against real hardware; `deploy.sh` and `start.sh`
must not be executed without that go.**

Scope (amendment 2026-10-23, option D, CoinRun-coop only, six seeds): registered
condition 6 arms {gtm, mlp, atoc, dial, eng, nocomm} x seeds
{0, 17, 42, 73, 101, 131} = 36 runs; closed condition {gtm, dial, eng,
nocomm} x 6 seeds = 24 runs. 60 runs of 50M pair-steps. Each run has a
`host` (seed block factor): **macM1 trains seeds 0, 17, 42; KXKM trains
73, 101, 131**. Evaluation on held-out levels 1000-1199 (intact + shuffled)
only after training and **only on KXKM** (x86 Procgen build); `--host
macm1` never evaluates (it raises). Each host has its own campaign dir and
campaign.json (host pinned).

Files: `queue.py` (frozen list, status files), `run_campaign.py` (runner
and guards), `kxkm/deploy.sh`, `kxkm/start.sh`.

## Pre-flight checklist (all must be ticked)

- [ ] Stage-2 manifest filed on OSF; DOI recorded. Runs started before
      filing are excluded by the registration.
- [ ] Manifest file committed under
      `experiments/procgen_multiagent_gtm_comm/manifest/`, with a
      `Status: FILED` line (no `DRAFT` anywhere), the exact OSF DOI
      (`10.17605/OSF.IO/<ID>`, placeholders refused) and the pinned
      commit in its text; its SHA-256 noted (`shasum -a 256 <file>`).
- [ ] Code commit pinned: the commit named in the manifest. HEAD of the
      deployed clone must equal it, or descend from it with every intervening commit
      touching only `docs/` or the manifest folder (the manifest cannot name its own commit). Pass it as
      `--expect-commit` (40 hex).
- [ ] Procgen builds: the macM1 arm64 build and the KXKM x86-64 wheel are
      known NOT to be bit-identical (disclosed in the amendment); nothing to
      equalise, but record both builds in the manifest.
- [ ] GPU reservation on KXKM: `gpu-worker` and `whisper-dex` stopped for
      the campaign (restore: `/ai/saisail/n10c-run/restore-services.sh`).
      `--min-free-gb` (default 8) must be free. The runner never kills or
      pauses other processes.
- [ ] Disk space: check `df -h /ai/saisail` (checkpoints every 1M
      pair-steps, about 50 per run kept; size one with a smoke run).
- [ ] Archive target exists on electron-server and the key works
      (`--archive-to user@host:/path`, rsync over ssh, BatchMode).
- [ ] `--dry-run` shows all guards passing and the expected duration.
- [ ] Both hosts ready (KXKM: micro_batch_pairs 512 + clear_cache from PR #53 merged).

## Deploy (KXKM, after go)

    bash kxkm/deploy.sh --dry-run     # prints commit, target, pins
    bash kxkm/deploy.sh               # bundle -> /ai/saisail/n10c-run/{repo,.venv}

Goes through the electron-server jump (`-J electron@100.78.191.52`,
target `electron@10.2.0.237`); no `tailscale switch`. Work only under
`/ai/saisail`. Needs `uv` already on the host (otherwise it stops: no
new install without review). Venv pins: Python 3.10 (uv managed),
mlx==0.31.1, mlx-cuda-12==0.31.1, numpy==1.26.4, gym3==0.3.3,
gym==0.26.2, procgen==0.10.7, pytest==9.0.3. These are exact pins, not
yet hash-locked: diff `pip freeze` against the Mac lock by hand (HITL).

## Start / stop / resume

Dry run (nothing trained; prints plan, guards, estimate):

    python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign \
      --campaign-dir /ai/saisail/n10c-run/campaign --dry-run \
      --throughput-json throughput.json --manifest experiments/procgen_multiagent_gtm_comm/manifest/<file>.md \
      --manifest-sha256 SHA --manifest-doi 10.17605/OSF.IO/XXXXX \
      --expect-commit COMMIT

Start (detached, tmux session `n10c` or nohup; pre-checks guards first):

    BASE=/ai/saisail/n10c-run bash kxkm/start.sh --manifest ... \
      --manifest-sha256 SHA --manifest-doi DOI --expect-commit COMMIT \
      --archive-to user@electron-server:/path --min-free-gb 8

- Stop gracefully: `touch <campaign-dir>/STOP`; the runner exits
  between runs (exit 4). Remove the file to resume.
- Hard stop (Ctrl-C, kill, reboot): the interrupted run is found
  `running` at resume and is restarted FROM SCRATCH; that counts as one
  infrastructure restart.
- The same guards protect the harness CLIs: `harness.train` refuses
  `--steps >= 50M` and `harness.evaluate` refuses levels >= 1000 unless
  the four guard flags are valid (dev budgets and training levels stay
  open). The campaign pins budget and trial flag in `campaign.json`;
  resuming with another budget or mixing trial and registered runs is
  refused.
- Resume: run the same `start.sh` command. The campaign is pinned to the
  manifest SHA, DOI and commit in `campaign.json`; different inputs are
  refused. Done runs are skipped; a trained run whose evaluation failed
  is re-evaluated without retraining.
- State: `<campaign-dir>/status/<run_id>.json`, event log
  `attempts.jsonl`, run outputs `runs/<run_id>/` (`run.json`,
  `eval.json`, checkpoints). Failed partials are kept as
  `<run_id>.failed-<n>`.
- Exit codes: 0 done, 2 guard refused, 3 GPU guard (state unchanged),
  4 STOP file, 5 finished with unresolved runs.

## Failure handling

| Failure | What the runner does | What you do |
|---|---|---|
| Crash, OOM, stub procgen, disk error, power loss (infrastructure) | Moves the partial aside, restarts from scratch; at most 2 restarts (3 attempts); every attempt in `attempts.jsonl` | Fix the cause between runs if you can. After 3 failures the run becomes a `training_failure` (flag `infra_exhausted`), evaluated at its last finite checkpoint (searched across the `.failed-<n>` dirs) or at initialisation, never replaced; report it per arm |
| Non-finite loss or parameter (`training_failure`) | Never retried or replaced; last finite checkpoint evaluated (untrained init if none) | Nothing; it is reported as a failure per the registration |
| Evaluation crash | Retried up to 3 times, no retraining | If still failing, debug and rerun evaluate by hand with the CLI |
| GPU guard (free memory below the minimum) | Stops before the run, state unchanged (exit 3) | Agree the GPU with the PI, then resume; `--gpu-wait-s` can wait |
| Archive failure | Logged, not fatal; retried at next resume | Check ssh/rsync target |
| Manifest/commit/tree mismatch | Refuses to start (exit 2) | Never edit the guards; fix the inputs |

## Expected duration (fill in from the benchmark)

| Arm | pair-steps/s | h per run (train) | eval h | runs | total h |
|---|---|---|---|---|---|
| gtm | TBD | TBD | TBD | 10 | TBD |
| nocomm | TBD | TBD | TBD | 10 | TBD |
| others (proxy: nocomm) | TBD | TBD | TBD | 30 | TBD |
| **Total** | | | | 50 | TBD (planning figure ~800 h) |

`--dry-run --throughput-json` computes this with the manifest
convention (gtm rate for gtm runs, nocomm rate for the rest).

## Hosts, sync and freeze-tag mode (2026-10-23, second round)

Run one process per host, each with its own `--campaign-dir`:

    # macM1 (trains seeds 0, 17, 42; no evaluation)
    ... run_campaign --host macm1 --campaign-dir ~/n10c/campaign ...
    # KXKM (trains 73, 101, 131; evaluates its own runs and the synced macM1 runs)
    ... run_campaign --host kxkm --campaign-dir /ai/saisail/n10c-run/campaign ...

macM1 runs end at status `trained` and are evaluated on KXKM after their
checkpoints are copied (through the electron-server jump):

    python -m experiments.procgen_multiagent_gtm_comm.campaign.run_campaign sync \
      --campaign-dir ~/n10c/campaign \
      --to electron@10.2.0.237:/ai/saisail/n10c-run/campaign

(`sync` copies `runs/<id>/{run.json,<last ckpt>}` first and
`status/<id>.json` second, never overwrites, marks the run `synced` on
macM1 only after both succeeded, and can be repeated. Equivalent manual
step: `rsync -aL --ignore-existing -e "ssh -J electron@100.78.191.52" ...`.)
Re-run the KXKM campaign command afterwards: it evaluates the newly
synced runs and never trains them. The finished event reports
`awaiting_sync`.

**Commitment point (PI decision 2026-10-23 22:17).** Instead of
`--manifest-doi`, start with `--freeze-tag <tag> [--swhid swh:1:rev:...]`:
the tag must exist on origin (`git ls-remote`), be annotated, point at HEAD
(or at an ancestor with only docs/ and manifest-folder commits since), the
tree must be clean, and the tag message must contain `manifest: <path>` and
`manifest-sha256: <hex>` equal to the committed manifest's SHA-256. The
manifest says `Status: FROZEN (tag <tag>)`. Tag, commit and SWHID are pinned
in `campaign.json`. The DOI mode is unchanged.
