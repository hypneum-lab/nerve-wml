"""Advisor checkpoint — save/load a SimNerveAdapter + pool as a portable artefact.

Uses torch.save (pickle-backed) for simplicity instead of safetensors: nerve-wml
ships without safetensors as a runtime dep, and the artefact is small enough
that pickle overhead is negligible. Signed-scheme migration to safetensors is
a v1 follow-up.

Plan 4d Task 2.
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_advisor_checkpoint(
    pool: list,
    nerve,
    path: str | Path,
) -> None:
    """Serialise WML pool + SimNerveAdapter state to a directory.

    Path layout:
        path/pool.pt      — list of per-WML state_dict()s
        path/nerve.pt     — nerve.state_dict() (router logits + transducer logits)
        path/manifest.pt  — n_wmls, k, per-WML class name, alphabet_size
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    pool_state = [
        {
            "class_name": type(wml).__name__,
            "id":         wml.id,
            "state_dict": wml.state_dict(),
        }
        for wml in pool
    ]
    torch.save(pool_state, out / "pool.pt")
    torch.save(nerve.state_dict(), out / "nerve.pt")

    manifest = {
        "n_wmls":        nerve.n_wmls,
        "k":             nerve.router.k,
        "alphabet_size": nerve.ALPHABET_SIZE,
        "schema_version": "v0",
    }
    torch.save(manifest, out / "manifest.pt")


def load_advisor_checkpoint(path: str | Path) -> dict:
    """Inverse of save_advisor_checkpoint. Returns a dict of raw state.

    The caller must rebuild the WML pool and SimNerveAdapter by name
    (class_name field) and call .load_state_dict() on each.
    """
    src = Path(path)
    pool_state = torch.load(src / "pool.pt", weights_only=False)
    nerve_state = torch.load(src / "nerve.pt", weights_only=False)
    manifest = torch.load(src / "manifest.pt", weights_only=False)
    return {
        "pool_state":  pool_state,
        "nerve_state": nerve_state,
        "manifest":    manifest,
    }
