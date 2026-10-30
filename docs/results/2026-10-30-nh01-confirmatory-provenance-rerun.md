# NH-01 confirmatory — provenance re-run 2026-10-30

**Why:** the review of PR #86 found that `run_meta.json` of the original
run records `git_commit = 8a13a33` (a working copy with untracked files)
and that the SHA-256 check of the executed code against the tagged
commit was asserted but not archived. PI decision 2026-10-30: re-run
from a clean clone at the tag and archive a full trace.

**Artefacts:** `results/nh01_confirmatory/provenance-rerun-20261030/`
(launch record, GitHub tag record, git status before and after, code
hashes, `uv pip freeze`, run log, re-run `run_meta.json` and
`summary.json`, output hashes, the wrapper and comparison scripts,
`comparison.json`, `SHA256SUMS`). The re-run cells (identical to the
committed ones, see below) are not committed a second time ; their
SHA-256 is in `outputs_sha256.txt`.

## Protocol

- Fresh clone of `https://github.com/hypneum-lab/nerve-wml.git` checked
  out at tag `prereg-nh01-confirmatory-20261025` (tag object `41135e4`,
  target commit `a000f73`) ; `git rev-parse HEAD` = `a000f73`, `git
  status --porcelain` empty before and after the run.
- SHA-256 of the three executed files (`run.py`, `confirmatory.py`,
  `__init__.py`) recomputed from `git show <tag>:<file>` and found
  equal before launch (the wrapper aborts otherwise).
- Tag record read from the GitHub API before launch : tagger date
  **2026-10-25 13:47:06 UTC**, API fetch **2026-10-30 12:10:33 UTC**,
  run start **12:10:34 UTC**, end **16:22:56 UTC** (15 136 s wall).
- Runner unchanged : `python -m experiments.nh01_hopfield_dream.confirmatory
  --workers 2 --out <dir>`, same seeds 0–9 and configuration (constants
  of the tagged code). Host electron-server (8 cores, shared ; load 7.4
  at launch, M5 kiki-flow and other jobs running), `nice 19`, BLAS and
  OpenMP threads = 1, Python 3.12.15, numpy 2.5.4 (same venv as the
  original run ; full freeze archived).
- One difference from the original launch : 2 worker processes instead
  of 4, to leave the cores of the concurrent M5 run free. The worker
  count changes only the scheduling ; every random stream is keyed by
  (seed, N, load, condition).

## Result : exact reproduction

- **9 120 / 9 120 cells identical** to the committed
  `results/nh01_confirmatory/cells.jsonl.gz` on every field except the
  wall-clock field `task_seconds` (cells matched by key ; `c_spec` is
  NaN in both, as the spectral factor is omitted by the
  pre-registration, and NaN is compared as NaN). The cells file differs
  byte-wise only by line order (tasks finish in a different order) and
  by `task_seconds`.
- **`summary.json` identical** to the committed one except its
  `run_meta` block (commit, start time, wall time, workers) ; hence every
  number and label of the verdict is unchanged.
- **`figure.png` byte-identical** (SHA-256 `3b64a94e…`).

The original run is therefore reproduced exactly by the code of the
tagged commit, run after the tag from a clean tree. This settles the
provenance gap for #86 ; it does not change, and is not evidence for,
any interpretation of the results.
