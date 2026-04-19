"""NerveWmlAdvisor — advisory-only router for micro-kiki.

Loads a trained pool + SimNerveAdapter from a checkpoint, encodes
incoming token embeddings to neuroletters, drives the pool through
the nerve for n_ticks, and returns a soft weight dict over 35
micro-kiki domains. Never raises — every error path returns None
so the host runtime is unaffected.

Env gate: NERVE_WML_ENABLED. When "0" or unset → all calls return None.
Env var:  NERVE_WML_CHECKPOINT_PATH — directory produced by save_advisor_checkpoint.

Plan 4d Tasks 4-6.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch import Tensor

from bridge.checkpoint import load_advisor_checkpoint
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML

logger = logging.getLogger(__name__)


class NerveWmlAdvisor:
    """Lazy advisor. Returns None on any failure — never raises."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        checkpoint_path: str | Path | None = None,
        n_domains: int = 35,
    ) -> None:
        if enabled is None:
            enabled = os.environ.get("NERVE_WML_ENABLED", "0") == "1"
        self.enabled = enabled
        self.n_domains = n_domains

        if checkpoint_path is None:
            checkpoint_path = os.environ.get("NERVE_WML_CHECKPOINT_PATH")
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )

        self._loaded = False
        self._pool: list = []
        self._nerve: SimNerveAdapter | None = None

    def _lazy_load(self) -> bool:
        """Load checkpoint on first advise(). Returns False on any failure."""
        if self._loaded:
            return True
        if not self.enabled or self.checkpoint_path is None:
            return False
        try:
            loaded = load_advisor_checkpoint(self.checkpoint_path)
            manifest = loaded["manifest"]
            nerve = SimNerveAdapter(
                n_wmls=manifest["n_wmls"],
                k=manifest["k"],
                seed=0,
            )
            nerve.load_state_dict(loaded["nerve_state"])

            pool: list = []
            for entry in loaded["pool_state"]:
                # Only MlpWML supported in the v0 advisor.
                if entry["class_name"] != "MlpWML":
                    continue
                wml = MlpWML(id=entry["id"], d_hidden=16, seed=0)
                wml.load_state_dict(entry["state_dict"])
                pool.append(wml)

            self._pool = pool
            self._nerve = nerve
            self._loaded = True
            return True
        except Exception as exc:  # noqa: BLE001 — advisor must never raise
            logger.debug("NerveWmlAdvisor lazy load failed: %s", exc)
            return False

    def advise(
        self,
        query_tokens: Tensor,
        current_route: dict | None = None,
    ) -> dict | None:
        """Return a soft-weight dict over n_domains, or None on any failure.

        Args:
            query_tokens: [batch, token_dim] float tensor. Only the first row is used.
            current_route: unused by v0 — reserved for a mixing strategy in v1.

        Returns:
            {domain_idx: float_weight} for all domains, summing to 1, OR None.
        """
        del current_route  # v0 doesn't consume the existing route
        if not self.enabled:
            return None
        if not self._lazy_load():
            return None
        if self._nerve is None or not self._pool:
            return None

        try:
            with torch.no_grad():
                if torch.isnan(query_tokens).any():
                    return None
                # Take first row, flatten to [token_dim].
                q = query_tokens[0].float()
                if q.shape[0] != 16:  # must match pool's d_hidden
                    return None
                # Run WML 0's π head on the query directly.
                pi_logits = self._pool[0].emit_head_pi(
                    self._pool[0].core(q.unsqueeze(0))
                ).squeeze(0)
                # Take the first n_domains codes as domain scores.
                scores = pi_logits[: self.n_domains]
                weights = torch.softmax(scores, dim=-1)
                return {i: float(weights[i].item()) for i in range(self.n_domains)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("NerveWmlAdvisor advise() failed: %s", exc)
            return None
