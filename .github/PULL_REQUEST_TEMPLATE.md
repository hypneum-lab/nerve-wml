## Summary

<!-- What changed and why. Mention any LLM assistance. -->

Closes #

## Checklist

- [ ] Tests pass: `uv run pytest -m "not slow"` (plus affected slow tests)
- [ ] `uv run ruff check .` and `uv run mypy nerve_core track_p track_w` clean
- [ ] No frozen run, result artefact or pre-registration file modified
      (post-freeze changes go through a PI-signed amendment)
- [ ] Every new number traces to a committed artefact
      (`scripts/factcheck_audit.py --ci`)
- [ ] Docs / CHANGELOG updated
- [ ] Commit subjects <= 50 chars, bodies <= 72 cols, no underscore in scope
