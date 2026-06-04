# Bio-substrate access — prerequisites and degradation contract

This note records the external prerequisite for the `BioWML` substrate
(Plan C — Biological Substrate WML Implementation) so a later subagent
never blocks on it. No code lands in this task — only documentation.

## Providers

Real biological-culture access requires an account and an API key from
one of two providers:

- **Cortical Labs CL1** — the "CL API" exposes real-time closed-loop
  read/stimulate access to a CL1 biocomputer unit. Commercial; request
  access at <https://corticallabs.com>. Round-trip latency: tens of ms
  per closed-loop step.
- **FinalSpark Neuroplatform** — a remote REST API to living human
  brain organoids, **free for research groups**; apply at
  <https://finalspark.com/neuroplatform>. Round-trip: queued batch,
  seconds to minutes per request depending on load.

## Environment variables

Three env vars govern all real-adapter behaviour:

| Variable | Required | Purpose |
|---|---|---|
| `NERVE_WML_BIO_API_KEY` | **yes (for real adapters)** | Auth token. If unset, the real adapter constructors raise `BioApiKeyMissing`. Mock client ignores this. |
| `NERVE_WML_BIO_PROVIDER` | no, default `finalspark` | Selects the adapter: `cl1` or `finalspark`. |
| `NERVE_WML_BIO_ENDPOINT` | no | Overrides the provider base URL (for staging / proxies). |

## Degradation contract

This contract **must hold for every Plan C task** that follows:

- `MockBioCultureClient` needs **no** env var and always works.
  Deterministic numpy spike simulation with realistic latency/jitter —
  the cross-substrate pool and CI fast suite use this exclusively.
- `CL1Adapter` and `FinalSparkAdapter` constructors read
  `NERVE_WML_BIO_API_KEY`. If it is unset they raise `BioApiKeyMissing`
  (a subclass of `RuntimeError`).
- Any `pytest` test that constructs a real adapter must first do
  `if not os.environ.get("NERVE_WML_BIO_API_KEY"): pytest.skip(...)` and
  be marked `@pytest.mark.slow`. CI runs `uv run pytest -m "not slow"`,
  so the network is never touched in CI.
- The hot path of `BioWML` never raises on missing credentials — if the
  caller passes a real adapter and the API key is unset, construction
  fails *before* `BioWML` ever holds a reference; the substrate itself
  is never built with a broken client.

This mirrors the existing env-gated precedent
`bridge/kiki_nerve_advisor.py` (`NERVE_WML_ENABLED`,
`NERVE_WML_CHECKPOINT_PATH`) — the substrate is opt-in, falls back to a
local mock, and never raises in the inference hot path.

## Rate-limit expectations

- **CL1**: ~tens of ms per closed-loop round-trip; reasonable for
  inline `BioWML.step()` calls if the carrier batch is small.
- **FinalSpark**: queued batched access; aggregate calls and treat the
  adapter as high-latency / low-throughput. The mock simulates this via
  configurable latency parameters.

## Status (2026-05-19)

No API key has been obtained yet — Plan C tasks 1–4 build the substrate
on the mock client; Task 5 implements the real adapters and gates them
behind `NERVE_WML_BIO_API_KEY`. The cross-substrate integration test in
Task 6 uses only the mock; CI never reaches the network.
