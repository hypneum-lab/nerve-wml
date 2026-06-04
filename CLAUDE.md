# CLAUDE.md — nerve-wml

Research engine for substrate-agnostic inter-WML nerve protocol. Python 3.12 + uv + torch. Design spec at `docs/superpowers/specs/2026-04-18-nerve-wml-design.md`.

## Structure

- `nerve_core/` — shared contracts (Neuroletter, Nerve/WML Protocol, invariants)
- `track_p/` — protocol simulator (SimNerve, VQ, transducer, router)
- `track_w/` — WML lab — **6 substrates implemented**: `MlpWML`, `LifWML`,
  `TransformerWML`, `BioWML`, `BioFieldWML`, `SpikingKikiWML`; plus
  `track_w/continual/` sub-package (`ewc.py` — diagonal Fisher EWC,
  `rehearsal.py` — fixed-replay `RehearsalBuffer`) for gate-W continual
  hardening (PR #46)
- `bridge/` — merge trainer — future plan
- `tests/` — unit (L1), info-theoretic (L2), integration (L3), golden (L4)

## Commands

```bash
uv sync --all-extras        # install
uv run pytest               # 507 tests (458 fast L1–L2, 49 slow L3–L4)
uv run pytest -m "not slow" # skip long tests
uv run ruff check .
uv run mypy nerve_core track_p
```

## Scientific protocol

**Every number traces to code/log:**
Every numerical claim in papers, research notes, or PR bodies must be **traceable to a JSON cell** in `docs/superpowers/research/` via the same session. Enforced by `scripts/factcheck_audit.py --ci` and `.github/workflows/factcheck.yml`. See `docs/superpowers/research/2026-05-20-factcheck-audit-report.md` for claim registry.

**Multi-seed default (n=50):**
New empirical claims use n=50 seed ensemble by default (Renf 4 baseline). See `scripts/multi_seed.py` wrapper.

**Cross-host validation:**
Science claims tested on ≥2 Apple Silicon archs (macm1 g13s + macM3 g15s / M5 g17g). PyTorch deterministic path is cross-arch bit-exact; Kuramoto-Euler-style iterators are NOT. MLX `mx.random.normal` is non-bit-exact M1 vs M3+ (ml-explore/mlx#3568).

## Invariants load-bearing

See `docs/invariants/` and the spec. Never weaken N-1..N-5 or W-1..W-4 without a spec update.
