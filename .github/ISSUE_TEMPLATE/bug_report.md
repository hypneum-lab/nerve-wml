---
name: Bug report
about: Something in the code, tests or CI does not behave as documented
labels: bug
---

## What happened

<!-- Expected vs actual behaviour. -->

## How to reproduce

```bash
uv sync --all-extras
uv run pytest path/to/test.py::test_name
```

## Environment

- OS / CPU architecture:
- Python version (3.12 expected):
- Commit or release:
- Seed(s), if the failure is stochastic:

## Scope check

- [ ] This does not concern a frozen run or a pre-registered result
      (for those, open a `correction` issue instead).
- [ ] Logs / traceback attached.
