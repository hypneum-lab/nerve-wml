# Contributing to nerve-wml

Research code of Hypneum Lab (programme dreamOfkiki, OSF
10.17605/OSF.IO/Q6JYN). Correctness and traceability come before
features. Please read this page before opening a pull request.

## Set up and run

```bash
git clone https://github.com/hypneum-lab/nerve-wml.git
cd nerve-wml
uv sync --all-extras            # Python 3.12+, torch CPU is enough
uv run pytest -m "not slow"     # fast tiers L1-L2 (what CI runs)
uv run pytest                   # full suite incl. L3-L4 golden/integration
uv run ruff check .
uv run mypy nerve_core track_p track_w
```

Optional extras: `uv sync --extra mnist` (torchvision), `--extra interpret`
(mne). Vendor SDKs (Loihi, Akida) are stubs; nothing downloads by default.

## Evidence contract

- Every numerical claim (paper, README, PR body, research note) must trace
  to a JSON cell under `docs/superpowers/research/` or a figure JSON under
  `papers/paper1/figures/`. `scripts/factcheck_audit.py --ci` enforces this
  and runs in `.github/workflows/factcheck.yml`.
- Real runs are registered through `harness/run_registry.py` (R1). A
  number without a committed artefact does not exist.
- New empirical claims use n = 50 seeds by default (`scripts/multi_seed.py`).
- Golden NPZs in `tests/golden/` are never regenerated to make a test
  pass; see the tolerance policy in `tests/golden/test_golden_regressions.py`.
- Invariants N-1..N-5 / W-1..W-4 (`nerve_core/invariants.py`) are never
  weakened without a spec update.

## Commits and branches

- Conventional commits, subject <= 50 ASCII characters, body lines <= 72,
  no underscore in the scope (`feat(track_p):` is rejected; drop the scope).
- Body required for anything beyond a typo: what changed, why, which
  artefact it moves.
- No `--no-verify`. No AI attribution trailer of any kind.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `renf/<name>`.
- Ship-impacting changes get an independent critic review before merge.

## Authorship and AI tools

- Byline for publications: single author, Clément Saillant (PI decision
  2026-10-25, roadmap S1). Contributors are credited in §Acknowledgments
  of each publication and listed in `CONTRIBUTORS.md`.
- LLMs (Claude, GPT, Qwen, ...) are tools. They are never co-authors and
  never appear in commit trailers. If an LLM helped, say so in the PR
  description so reviewers can weigh it.
## Issues and license

Use the org templates (`hypothese`, `correction`); new issues carry
`a-valider-pi` plus `phase:*`, `type:*`, `axe:nerve-wml`, `hote:*`.
One issue = one expected verdict. Contributions are MIT (see LICENSE).
