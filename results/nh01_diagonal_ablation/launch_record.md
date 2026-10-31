# Launch record — NH-01 diagonal ablation (2026-10-31)

Written after the run from the shell history and the host, as the
provenance trace required by `docs/results/2026-10-30-nh01-confirmatory-provenance-rerun.md`.

## Freeze (before any datum with seeds 200–209)

| Event | Time (UTC) | Evidence |
|---|---|---|
| PR #97 merged on green CI | commit `6e1b006` | GitHub |
| Annotated tag `prereg-nh01-diagonal-20261031` | 2026-10-31 16:38:17 | tag object `d1b9e0e`, target `6e1b006`, `git for-each-ref … taggerdate` |
| Software Heritage save request | 2026-10-31 16:38:19 | API `origin/save/git/url/https://github.com/hypneum-lab/nerve-wml/`, status `succeeded`, snapshot `swh:1:snp:2bb4ee0f98f91afbdb855b4000c54501f0bf6ed0` |
| Revision SWHID | — | `swh:1:rev:6e1b0069c49750f7cb2a01dff832a257c6157a6e` |
| Run launched | 2026-10-31 16:39:15 | `run_diag.started` (written by the launch command), `run_meta.json: started_utc` |
| Run finished | 2026-10-31 16:54:17 | mtime of `SHA256SUMS` (17:54:17 CET), `wall_seconds = 901.4` |

Tag message (verbatim fields): `file-sha256` of the pre-registration
`1aa53828032abe80f3ac7acfd4d8dcddd155333eb143b06254a5e7310c7f9e88`,
`code-sha256` of `diagonal_ablation.py`
`3b7fce417ee78d60a43acb7e5e17894b11e10dfb3ecc3c12c5b6b0d9b9f12824`,
thresholds 0.020 / 0.050 confirmed by the PI 2026-10-31 17:35 CET, go
given 17:37 CET.

## Host and code

- electron-server, Linux 6.8.0-136-generic, 8 × Intel Xeon X5680 (no AVX),
  shared host (load average ≈ 3 during the run).
- Fresh clone `~/scratch/nh01/nerve-wml-diag` at the tag:
  `git rev-parse HEAD` = `6e1b0069c49750f7cb2a01dff832a257c6157a6e`;
  `git status --porcelain` empty before the run and, after it, showing only
  the untracked `results/nh01_diagonal_ablation/`.
- SHA-256 of the executed files on the host, equal to
  `git show prereg-nh01-diagonal-20261031:<file>` (checked before launch,
  recomputed after the run):

| File | SHA-256 |
|---|---|
| `experiments/nh01_hopfield_dream/diagonal_ablation.py` | `3b7fce417ee78d60a43acb7e5e17894b11e10dfb3ecc3c12c5b6b0d9b9f12824` |
| `experiments/nh01_hopfield_dream/run.py` | `35f694771885c64f3fd24e2fa1c5b2277be2523e976d040126dc007020d53ecb` |
| `experiments/nh01_hopfield_dream/confirmatory.py` | `d19aa6d0d322a74e0b2c67984064a4cb684725208b4b25f713bce83f0a6972b9` |

- Environment: venv `~/scratch/nh01/venv` (the one used by the confirmatory
  and correlated runs), Python 3.12.15, numpy 2.5.4. Installed
  distributions: Pygments 2.21.0, contourpy 1.4.0, cycler 0.12.1,
  fonttools 4.67.0, iniconfig 2.3.1, kiwisolver 1.5.1, matplotlib 3.11.2,
  numpy 2.5.4, packaging 26.3, pillow 13.0.0, pluggy 1.7.0, pyparsing 3.3.3,
  pytest 9.1.1, python-dateutil 3.0.0, ruff 0.17.2, six 1.17.0,
  tzdata 2026.5.
- Timing probe (declared in the pre-registration): seed 997, two loads,
  23.6 s, projected 0.19 h for the full grid; output discarded.
- Launch command (from the clone directory):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nohup nice --adjustment=19 ../venv/bin/python \
  -m experiments.nh01_hopfield_dream.diagonal_ablation --workers 4 \
  > ../run_diag.log 2>&1
```

`run.log` in this directory is a copy of `../run_diag.log`.

## Outputs

`cells.jsonl` (4 800 lines), `summary.json`, `run_meta.json`, `SHA256SUMS`
(written by the runner, verified with `sha256sum -c` on the host and
`shasum -a 256 -c` after copy). 0 non-converged probe (max 68 sweeps),
0 right-censored estimate, per (seed, arm) task 291–306 s.
