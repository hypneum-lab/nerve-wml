# Wiring `NerveWmlAdvisor` into `micro-kiki`

Self-contained recipe for the downstream PR on [micro-kiki](https://github.com/electron-rare/micro-kiki). A subagent can implement this without reading any other nerve-wml documentation.

## What this adds

An advisory-only router that micro-kiki's `MetaRouter` calls before its sigmoid decision. The advisor:

- Returns a dict `{domain_idx: weight}` summing to 1.0, OR `None` on any failure.
- Is **env-gated** — when `NERVE_WML_ENABLED=0` (default), it is a no-op adding < 1 ms per call.
- **Never raises** — every exception is caught and returns `None`.

## Prereqs on the micro-kiki side

1. **Install nerve-wml as editable dep**:
   ```bash
   cd /path/to/micro-kiki
   uv add --dev --editable /path/to/nerve-wml
   ```
2. **Generate a test checkpoint** from a trained WML pool:
   ```bash
   uv run python -c "
   from bridge.checkpoint import save_advisor_checkpoint
   from bridge.sim_nerve_adapter import SimNerveAdapter
   from track_w.mlp_wml import MlpWML
   pool = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
   nerve = SimNerveAdapter(n_wmls=2, k=1, seed=0)
   save_advisor_checkpoint(pool, nerve, '/tmp/nerve-wml-checkpoint')
   "
   ```

## Env variables

| Variable | Default | Effect |
|---|---|---|
| `NERVE_WML_ENABLED` | `0` | `1` to enable the advisor, anything else keeps it off |
| `NERVE_WML_CHECKPOINT_PATH` | unset | absolute path to the checkpoint dir produced by `save_advisor_checkpoint` |
| `NERVE_WML_ALPHA` | `0.1` | mixing weight for the advisor's contribution |

## The diff — expected ~8 lines in `src/routing/meta_router.py`

Assumes the MetaRouter exposes `existing_logits` (pre-sigmoid tensor of shape `[35]`). If your codebase names differently, substitute.

```python
# Near module-level imports:
import os
from bridge.kiki_nerve_advisor import NerveWmlAdvisor

_advisor = NerveWmlAdvisor()  # reads NERVE_WML_ENABLED + NERVE_WML_CHECKPOINT_PATH
_alpha = float(os.environ.get("NERVE_WML_ALPHA", "0.1"))

# Inside the routing decision function, just BEFORE the existing sigmoid call:
def route(query_tokens):
    # ... existing code that produces `existing_logits: Tensor[35]` ...
    advice = _advisor.advise(query_tokens)
    if advice is not None:
        import torch
        advisor_logits = torch.tensor(
            [advice[i] for i in range(35)], dtype=existing_logits.dtype,
        )
        existing_logits = (1 - _alpha) * existing_logits + _alpha * advisor_logits

    # ... rest of the routing decision (sigmoid, argmax, etc.) ...
```

## Latency expectations

- **Disabled** (`NERVE_WML_ENABLED=0`): < 1 ms overhead per call.
- **Enabled, warm path**: < 50 ms per call on M-series Apple Silicon.
- **Cold path (first call)**: checkpoint load adds ~20-100 ms once; subsequent calls are warm.

## Verification on micro-kiki side

```bash
# With advisor off (default behaviour preserved):
unset NERVE_WML_ENABLED
uv run pytest tests/routing/ -v          # existing routing tests pass

# With advisor on:
export NERVE_WML_ENABLED=1
export NERVE_WML_CHECKPOINT_PATH=/tmp/nerve-wml-checkpoint
uv run pytest tests/routing/ -v          # still pass — advisor is additive
```

Optional: add a micro-kiki-side test that patches `_advisor.advise` to return a fake dict and asserts the mixing formula behaves as expected.

## What the advisor returns

```python
>>> advice = _advisor.advise(query_tokens)
>>> print(advice)
{0: 0.03, 1: 0.02, 2: 0.07, ..., 34: 0.04}  # sums to 1.0
```

If `None`, just skip the mixing step — the existing MetaRouter handles the request alone.

## Notes

- `query_tokens` must be a `Tensor[1, 16]` float. The checkpoint dictates the token dim; v0 uses `d_hidden=16`.
- Mixing formula `(1 - α) * existing + α * advisor` is one reasonable choice; log-space mix or product-of-experts are alternatives for v1.
- The advisor only reads the checkpoint — it never writes. Safe to share one checkpoint across multiple worker processes.

## Rollback

Set `NERVE_WML_ENABLED=0` and restart. No state is persisted on the micro-kiki side.
