"""Record/replay ε traces to NPZ for offline consolidation.

Plan 7 Task 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_eps_replay(
    trace: np.ndarray,
    metadata: dict,
    path: str | Path,
) -> None:
    """Write a schema v0 trace + metadata to disk.

    path/trace.npz: the encoded trace.
    path/metadata.json: commit_sha, seed, n_wmls, schema_version, etc.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "trace.npz", trace=trace)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_eps_replay(path: str | Path) -> tuple[np.ndarray, dict]:
    """Inverse of save_eps_replay. Returns (trace, metadata)."""
    src = Path(path)
    trace = np.load(src / "trace.npz")["trace"]
    metadata = json.loads((src / "metadata.json").read_text())
    return trace, metadata
