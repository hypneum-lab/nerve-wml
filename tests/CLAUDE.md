# Testing

## Framework

pytest — 507 tests: 458 fast (L1–L2 unit/invariant), 49 slow (L3–L4 integration/golden).

## Running

```bash
uv run pytest tests/
uv run pytest tests/ -v -k <name>  # filter by name
uv run pytest -m "not slow"        # L1–L2 only (~60 s)
```

## Structure

- **L1 (unit):** nerve_core contracts, encode/decode round-trips, protocol bounds checks.
- **L2 (invariant):** info-theoretic properties (synchrony, spectral_entropy), N-1..N-5 and W-1..W-4 hold.
- **L3 (integration):** Renf pipeline (Renf 1–13), end-to-end VQ/transducer/router chains. Slow (marked `@pytest.mark.slow`).
- **L4 (golden):** Artefact reproduction (golden JSONs `docs/superpowers/research/`).

## Conventions

- Test files: `test_*.py` or `*_test.py`
- Mirror module structure: `tests/<module>/test_<thing>.py`
- One assertion focus per test when practical
- Fixtures in `conftest.py`, not duplicated across files

## Numerical assertions

If a test asserts on a numerical value (accuracy, loss, metric threshold), the ground truth **must trace to a JSON cell** in `docs/superpowers/research/` via `scripts/factcheck_audit.py`. Golden artefacts (Renf 1–13 outputs) are stored as `*.json` per session and enumerated in the audit registry.

## Mocking

- Prefer real implementations
- Mock only at boundaries: network, time, randomness, filesystem

## Anti-patterns

- Don't mock what you can test directly
- Don't test implementation details (private methods, internal state)
- Don't share mutable state between tests
- Don't write tests that pass when production code is broken
