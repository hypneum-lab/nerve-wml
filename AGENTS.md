# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`nerve-wml` — substrate-agnostic nerve protocol for inter-WML (Weight-Memory-Language) communication. Research engine of **Hypneum Lab**, framework C. Master HEAD `75aaa17` (2026-05-21), v1.8.1 on PyPI. Design spec: `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`.

## Tech stack

- Language: Python 3.12+ (>=3.12 required in `pyproject.toml`)
- Runtime: `uv` (sync via `uv sync --all-extras`)
- Test: `pytest` — 507 tests (458 fast L1-L2, 49 slow L3-L4)
- Build: `hatchling`; wheels packaged from `nerve_core/`, `track_p/`, `track_w/`, `bridge/`, `harness/`, `interpret/`, `neuromorphic/`, `nerve_wml/`
- Core deps: `torch>=2.3`, `numpy>=1.26`, `scipy>=1.17.1`, `scikit-learn>=1.8.0`
- Optional extras: `[mnist]` (torchvision), `[interpret]` (mne — moved out of core in N5 Task 9)

## Commands

```bash
uv sync --all-extras
uv run pytest                 # full
uv run pytest -m "not slow"   # skip L3-L4
uv run ruff check .
uv run mypy nerve_core track_p
```

## Conventions

- Commits: subject ≤ 50 chars ASCII, body lines ≤ 72, no underscore in scope (`feat(track_p):` is rejected — drop scope for PEP-8-style packages), no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `prep/<name>`, `renf/<name>` (reinforcement-run results).
- PRs: critic-review mandatory for ship-impacting changes — see `~/.claude/projects/-Users-electron/memory/feedback_critic_before_ship.md` (validated saver at v1.7.0 and v1.8.0).
- Tests: keep the L1/L2/L3/L4 tiers separate; `-m "slow"` marks L3-L4.
- Multi-seed default `n=50` for new empirical claims (Renf 4 baseline). Wrapper: `scripts/multi_seed.py`.

## File layout

- `nerve_core/` — shared contracts: Neuroletter, Nerve/WML Protocol, invariants N-1..N-5 / W-1..W-4. Load-bearing.
- `track_p/` — protocol simulator (`SimNerve`, VQ, transducer, router).
- `track_w/` — WML lab (`MockNerve`, `MlpWML`, `LifWML` — future).
- `bridge/` — merge trainer (future).
- `interpret/` — MNE-backed interpretability (separate extra, do not import unconditionally).
- `neuromorphic/` — Norse/LIF substrates.
- `harness/` — eval harness, run registry, matrix config.
- `tests/` — `L1` unit / `L2` info-theoretic / `L3` integration / `L4` golden.
- `docs/superpowers/research/` — factcheck JSON cells (every numerical claim must trace to one).
- `docs/invariants/` — N/W invariant definitions.

## Domain-specific gotchas

- **Neuroletter is dimensionless**: do not attach physical units in protocol code or papers.
- **Every numerical claim in papers/PRs must trace** to a JSON cell in `docs/superpowers/research/` (enforced by `scripts/factcheck_audit.py --ci` + `.github/workflows/factcheck.yml`).
- **Cross-host bit-exactness**: PyTorch deterministic path is bit-exact across Apple Silicon archs; Kuramoto-Euler iterators are NOT. `mx.random.normal` in MLX is non-bit-exact M1 vs M3+ (ml-explore/mlx#3568). Validate empirical claims on ≥ 2 archs (macm1 g13s + macM3 g15s / M5 g17g).
- **Do not weaken N-1..N-5 or W-1..W-4** without a matching spec update — these are the protocol invariants the paper rests on.
- **`mne` is an extra**, not a core dep. Code in `interpret/` must lazy-import; tests outside `tests/interpret/` must not require it.
- **GammaThetaMultiplexer (issue #1)** is merged on master but empirical validation lives downstream in `bouba_sens` Sprint 1+. Don't claim empirical validation here.

## When in doubt

- Read `CLAUDE.md` (project-specific guidance) and `docs/superpowers/specs/2026-04-18-nerve-wml-design.md` (spec).
- Recent commits: `git log --oneline -20`.
- Cluster context: `~/CLAUDE.md`.
- Run `uv run pytest -m "not slow"` before non-trivial commits; full suite before tag/release.
