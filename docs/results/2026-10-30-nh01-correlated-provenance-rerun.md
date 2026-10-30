# NH-01 correlated patterns — provenance re-run 2026-10-30

**Why:** the review of PR #87 found that `run_meta.json` of the original
run records `git_commit = 8a13a33` (a working copy with untracked files),
that the SHA-256 check of the executed code against the tagged commit
was asserted but not archived, and that the tag push and the run start
carried the same second (16:48:06 UTC), so their order was not
established from the artefacts. PI decision 2026-10-30: re-run from a
clean clone at the tag and archive a full trace.

**Artefacts:** `results/nh01_correlated/provenance-rerun-20261030/`
(launch record, GitHub tag record, git status before and after, code
hashes, `uv pip freeze`, run log, re-run `run_meta.json` and
`summary.json`, output hashes, the wrapper and comparison scripts,
`comparison.json`, `SHA256SUMS`). The re-run cells (identical to the
committed ones, see below) are not committed a second time ; their
SHA-256 is in `outputs_sha256.txt`.

## Protocol

- Fresh clone of `https://github.com/hypneum-lab/nerve-wml.git` checked
  out at tag `prereg-nh01-correlated-20261027` (tag object `8fe96b8`,
  target commit `df9a1e7`) ; `git rev-parse HEAD` = `df9a1e7`, `git
  status --porcelain` empty before and after the run.
- SHA-256 of the four executed files (`run.py`, `correlated.py`,
  `confirmatory.py`, `__init__.py`) recomputed from
  `git show <tag>:<file>` and found equal before launch (the wrapper
  aborts otherwise).
- Tag record read from the GitHub API before launch : tagger date
  **2026-10-27 16:48:01 UTC**, API fetch **2026-10-30 12:10:34 UTC**,
  run start **12:10:34 UTC**, end **20:39:46 UTC** (30 547 s wall).
- Runner unchanged : `python -m experiments.nh01_hopfield_dream.correlated
  --workers 2 --out <dir>` (2 workers, as in the original run), same
  seeds 100–109, levels m ∈ {0, 0.15, 0.30} and configuration
  (constants of the tagged code). Host electron-server (8 cores,
  shared ; load 7.2 at launch, M5 kiki-flow and other jobs running),
  `nice 19`, BLAS and OpenMP threads = 1, Python 3.12.15, numpy 2.5.4
  (same venv as the original run ; full freeze archived).

## Result : exact reproduction

- **29 640 / 29 640 cells identical** to the committed
  `results/nh01_correlated/cells.jsonl.gz` on every field except the
  wall-clock field `task_seconds` (cells matched by key, NaN compared
  as NaN). The cells file differs byte-wise only by line order (tasks
  finish in a different order) and by `task_seconds`.
- **`summary.json` identical** to the committed one except its
  `run_meta` block (commit, start time, wall time) ; hence every number
  and label is unchanged.
- **`figure.png` byte-identical** (SHA-256 `de3ee720…`).

The original run is therefore reproduced exactly by the code of the
tagged commit, run after the tag from a clean tree. This settles the
provenance gap for #87 ; it does not change, and is not evidence for,
any interpretation of the results (in particular the reading of the
m = 0.30 level, left to the PI).
