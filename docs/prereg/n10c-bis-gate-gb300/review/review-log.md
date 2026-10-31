# Review log — G7 host annex

**Pass 1:** Codex CLI 0.162.1, model `gpt-6.1-sol`, reasoning effort `high`, 2026-10-31
(about 21:45–22:30 CET), blind on the repository (branch head `89fd478`), brief and
report in this folder (`codex-brief-pass1.md`, `codex-report-pass1.md`, in French). No
file of the repository was changed by the reviewer. Each finding was checked against the
source before a change; verdicts below. Counts: 2 BLOQUANT confirmed, 7 MAJEUR confirmed
(M5 as a disclosure and a PI decision), 3 MINEUR confirmed, 0 rejected.

| id | finding | verdict | change |
|---|---|---|---|
| B1 | 18 / 5 slots cannot give the one rerun per failed run of G2 | confirmed (`devledger.py` counts crashed attempts) | allowance 2k per arm: `eng` 21, `nocomm` 8; annex section 7 |
| B2 | the concurrent-arms option contradicts G5 | confirmed | option removed; one run at a time is closed, not a TBD |
| M1 | verdict after interruption, zero pairs, survivor selection | confirmed | PASS / FAIL / INCONCLUSIVE table, trunks checked before the first run, no substitution (section 3) |
| M2 | exit code 0 does not mean success; launch not guarded | confirmed (`train.py` returns a record on liveness abort) | committed lane `gate_g7_lane.sh` and a run-status rule (section 8) |
| M3 | `smoke_test` allows the stub and does not restrict levels | confirmed (`coop_env.py`, `num_levels=0`) | S4 replaced by an explicit wrapper check |
| M4 | environment not sealed; `filelock` 4.1.0 vs Procgen `filelock<4` | confirmed (Procgen `setup.py` at `5e1dbf3`) | TBD-6 and TBD-7 widened |
| M5 | carry-over of the GB300 verdict to KXKM-AI is an assumption | confirmed | wording of section 1 item 3; new TBD-2 |
| M6 | trainer stdout carries the success curve; seed choice informed | confirmed (`train.py` prints `success_curve`) | stdout sealed, monitoring list, seed rationale (sections 3, 9) |
| M7 | time rule not executable | confirmed (S5 gives two rates per arm) | formula with `pair_steps_per_s_incl_warmup`, factor 1.5, deadline kill |
| m1 | S5 `--out` is a file | confirmed | path fixed, pass condition added |
| m2 | archive contents, ledger snapshot, `SHA256SUMS` self-inclusion | confirmed | section 9 list |
| m3 | fixed arm order may confound load drift | confirmed | arm order alternates per seed |
