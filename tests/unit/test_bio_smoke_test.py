"""Unit test for scripts.bio_smoke_test.

Minimal coverage: verify that the smoke test skips cleanly when the
API key env var is unset (expected default behavior for CI).
"""
from __future__ import annotations

import os
import subprocess


def test_bio_smoke_unset_env_skips_cleanly() -> None:
    """Smoke test exits 0 and prints skip message when NERVE_WML_BIO_API_KEY is unset."""
    env = os.environ.copy()
    env.pop("NERVE_WML_BIO_API_KEY", None)
    result = subprocess.run(
        ["uv", "run", "python", "-m", "scripts.bio_smoke_test"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
    assert "[skip]" in result.stdout
    assert "NERVE_WML_BIO_API_KEY" in result.stdout
