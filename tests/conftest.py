"""Top-level shared fixtures for nerve-wml tests.

Originally extracted in N3 Task 3 (commit e713863) to host the
``LINUX_CI_312`` xfail sentinel. N3 Task 9 (2026-05-10) removed the
sentinel after fixing the 3 platform-fragile tests at the test/source
level rather than skipping them. Kept as the canonical location for
future cross-suite fixtures.
"""
from __future__ import annotations
