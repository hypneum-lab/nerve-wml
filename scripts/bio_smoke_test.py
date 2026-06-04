"""CLI smoke for the real bio adapters.

One-shot roundtrip validation against Cortical Labs CL1 or FinalSpark.
Requires NERVE_WML_BIO_API_KEY in the environment.

Usage:
    export NERVE_WML_BIO_API_KEY=fs_...yourkeyhere
    export NERVE_WML_BIO_PROVIDER=finalspark    # or cl1
    uv run python -m scripts.bio_smoke_test

If NERVE_WML_BIO_API_KEY is unset, prints a runbook pointer and exits 0
(skipping is the expected default for CI).
"""
from __future__ import annotations

import os
from time import perf_counter

from track_w.bio_clients import (
    BioApiKeyMissingError,
    CL1Adapter,
    FinalSparkAdapter,
)

_ENV_KEY = "NERVE_WML_BIO_API_KEY"
_ENV_PROVIDER = "NERVE_WML_BIO_PROVIDER"
_RUNBOOK = "docs/superpowers/research/2026-05-20-bio-api-credentials.md"


def main() -> int:
    """Run one roundtrip smoke test.

    Exit 0 if key is unset (skip); 0 if success; non-zero on error.
    """
    if not os.environ.get(_ENV_KEY):
        print(f"[skip] {_ENV_KEY} is unset.")
        print(f"       See {_RUNBOOK} for how to obtain credentials.")
        return 0

    provider = os.environ.get(_ENV_PROVIDER, "finalspark").lower()
    print(f"[bio-smoke] provider = {provider}")

    client: CL1Adapter | FinalSparkAdapter
    try:
        if provider == "cl1":
            client = CL1Adapter()
        elif provider == "finalspark":
            client = FinalSparkAdapter()
        else:
            print(f"[error] unknown provider '{provider}' (use 'cl1' or 'finalspark')")
            return 2
    except BioApiKeyMissingError as exc:
        print(f"[error] {exc}")
        return 2

    codes = [7, 17, 42]
    print(f"[bio-smoke] roundtrip codes = {codes}")
    t0 = perf_counter()
    try:
        frame = client.roundtrip(codes)
    except Exception as exc:
        wall_ms = (perf_counter() - t0) * 1e3
        print(f"[fail] roundtrip raised after {wall_ms:.1f} ms: {exc!r}")
        return 3
    wall_ms = (perf_counter() - t0) * 1e3
    print(f"[ok]   spikes.shape = {frame.spikes.shape}")
    print(f"[ok]   reported latency_ms = {frame.latency_ms:.2f}")
    print(f"[ok]   wall-clock roundtrip = {wall_ms:.2f} ms")

    decoded = client.decode_activity(frame)
    print(f"[ok]   decoded codes = {decoded}")
    hits = sum(1 for d, s in zip(decoded, codes, strict=False) if d == s)
    fidelity = hits / len(codes) if codes else 0.0
    print(f"[ok]   round-trip fidelity = {hits}/{len(codes)} = {fidelity:.0%}")

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
