# Obtaining bio-substrate API credentials

This runbook covers how to obtain and validate a real API key for `nerve-wml`'s BioWML substrate. The substrate ships (PR #18) validated only against the offline `MockBioCultureClient`. To claim cross-substrate generality including a real biological culture, one end-to-end roundtrip against either Cortical Labs CL1 or FinalSpark Neuroplatform must be recorded in your environment.

## Why this matters

PR #18 establishes the BioWML substrate (encode, decode, stimulus/activity wire protocol), but exercises only a deterministic numpy mock. To earn credibility in the broader AI-bio substrate literature, at least one validated roundtrip against real spiking activity is necessary. This document guides you to obtain credentials and run the validation script.

## Two providers: cost and latency trade-off

| Provider | Cost | Access | Latency | Notes |
|---|---|---|---|---|
| **Cortical Labs CL1** | Commercial (contact for pricing) | Real-time closed-loop | 10–50 ms/roundtrip | Live biocomputer units; lowest latency; requires approval + payment. |
| **FinalSpark Neuroplatform** | Free for research | REST API batch queue | 1–10 seconds/roundtrip | Human brain organoids; free tier for academics; queued, not real-time. |

**Recommendation**: Start with FinalSpark. It is free for research groups, and batch latency is acceptable for offline validation. Apply at <https://finalspark.com/neuroplatform>.

## Obtaining FinalSpark credentials

1. **Visit the application portal**: <https://finalspark.com/neuroplatform>
2. **Complete the research application form** with:
   - Your institutional affiliation (university, research lab, company)
   - Project description (e.g., "Validating inter-substrate WML nerve protocol on human brain organoids")
   - Expected API usage (e.g., "< 10 requests/day for research validation")
   - Ethics statement: "I confirm that my institution's IRB or equivalent has reviewed organoid research, or I am exempt from such review."
3. **Wait for approval**: Typically weeks to months (FinalSpark reviews research applications to ensure ethical use). Confirm timeline at the application portal.
4. **Receive API key by email**: Once approved, FinalSpark will send you a `NERVE_WML_BIO_API_KEY` token.
5. **Store securely**: Add to your shell profile, dotenv, or secret-management tool (e.g., Infisical):
   ```bash
   export NERVE_WML_BIO_API_KEY=fs_...yourkeyhere
   ```

## Obtaining Cortical Labs CL1 credentials

CL1 requires direct contact and commercial terms. Proceed if you have existing infrastructure or internal funding.

1. **Request access**: <https://corticallabs.com> → Contact sales/research partnerships.
2. **Negotiate access**: CL will discuss hardware access, API tier, and cost.
3. **Receive API key**: Once terms are agreed, you will receive `NERVE_WML_BIO_API_KEY`.
4. **Set provider**: When running validation scripts, set `NERVE_WML_BIO_PROVIDER=cl1`:
   ```bash
   export NERVE_WML_BIO_API_KEY=cl_...yourkeyhere
   export NERVE_WML_BIO_PROVIDER=cl1
   ```

## Validating end-to-end connectivity

Once you have an API key, validate that it works:

```bash
# Set your credentials
export NERVE_WML_BIO_API_KEY=fs_...yourkeyhere

# (optional) Confirm provider (default is finalspark)
export NERVE_WML_BIO_PROVIDER=finalspark

# Navigate to the nerve-wml repo root
cd /path/to/nerve-wml-wt-gap

# Run the smoke test (1 roundtrip, ~10–30 seconds for FinalSpark)
uv run python -m scripts.bio_smoke_test
```

The smoke test will:
- Instantiate the real adapter (CL1 or FinalSpark).
- Send a small stimulus code list `[7, 17, 42]` to the culture.
- Read back the spiking activity.
- Decode the activity to recover codes.
- Print latency, decoded codes, and round-trip fidelity (how many codes matched).
- Exit 0 if successful, non-zero on error.

## Failure modes and remediation

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP 401 Unauthorized | API key is invalid or truncated. | Re-check the key in your env; paste it fresh from email/secret manager. |
| HTTP 429 Too Many Requests | Rate limit exceeded (FinalSpark queues requests). | Wait 1–5 minutes and retry. Contact FinalSpark for rate-limit tier if you exceed research limits. |
| HTTP 5xx | Provider server error. | Check provider status page; wait 10 minutes and retry. If persistent, contact support. |
| Connection refused / DNS error | Wrong endpoint URL. | Unset `NERVE_WML_BIO_ENDPOINT` (defaults are baked into adapters); retry. |
| 200 OK but `spikes.shape` mismatch | Provider API changed wire format. | Check the provider's API changelog; open an issue in `nerve-wml` with the actual response. |
| Decoded codes ≠ sent codes | Encoder/decoder mismatch or noisy activity. | Fidelity < 50% is concerning; check with provider if the culture is responsive. Fidelity > 50% is acceptable for research. |

## Integration tests

If you want to run the full integration suite against your real adapter:

```bash
# Mark `@pytest.mark.slow` tests will now run
export NERVE_WML_BIO_API_KEY=fs_...yourkeyhere

# Run integration tests (skips mock-only tests)
uv run pytest tests/integration/track_w/test_bio_adapters.py -v
uv run pytest tests/integration/track_w/test_bio_cross_substrate.py -v
```

## Privacy and ethics

If using FinalSpark, you are interacting with **human-derived brain organoids** — mini-brains grown from human cells in the lab. FinalSpark's published [ethics framework](https://finalspark.com/ethics) covers consent, minimization of harm, and transparency. **Confirm that your institution's IRB or ethics board has reviewed your use** (or you are exempt under your local research exemption policy). If you are publishing results, cite FinalSpark's organoid sourcing and ethics statement in your methodology.

## Environment variable reference

- **`NERVE_WML_BIO_API_KEY`** (required): Your API token from CL1 or FinalSpark. If unset, real adapters raise `BioApiKeyMissingError` and the smoke test exits 0 (skip).
- **`NERVE_WML_BIO_PROVIDER`** (optional, default `finalspark`): `cl1` or `finalspark`.
- **`NERVE_WML_BIO_ENDPOINT`** (optional): Override base URL (e.g., for staging or local proxy). Leave unset to use provider defaults.

## Next steps

1. Apply for FinalSpark access (or contact CL1).
2. Wait for approval (weeks to months).
3. Receive API key.
4. Run `uv run python -m scripts.bio_smoke_test` to validate.
5. (Optional) Run integration tests with `@pytest.mark.slow`.
6. Document your results in your own research notes or PR comments.
