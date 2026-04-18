# CLAUDE.md — nerve-wml

Research engine for substrate-agnostic inter-WML nerve protocol. Python 3.12 + uv + torch. Design spec at `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`.

## Structure

- `nerve_core/` — shared contracts (Neuroletter, Nerve/WML Protocol, invariants)
- `track_p/` — protocol simulator (SimNerve, VQ, transducer, router)
- `track_w/` — WML lab (MockNerve, MlpWML, LifWML) — future plan
- `bridge/` — merge trainer — future plan
- `tests/` — unit (L1), info-theoretic (L2), integration (L3), golden (L4)

## Commands

```bash
uv sync --all-extras        # install
uv run pytest               # all tests
uv run pytest -m "not slow" # skip long tests
uv run ruff check .
uv run mypy nerve_core track_p
```

## Invariants load-bearing

See `docs/invariants/` and the spec. Never weaken N-1..N-5 or W-1..W-4 without a spec update.
