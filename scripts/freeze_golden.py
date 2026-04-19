"""Freeze golden-NPZ regressions for L4 tests.

Run once after gate-m-passed. Produces:
  tests/golden/cycle_trace_4wmls_seed0.npz  — neuroletter codes per cycle
  tests/golden/transducers_merged.npz       — logits of every nerve transducer

Subsequent runs must reproduce these exactly (bit-stable).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bridge.sim_nerve_adapter import SimNerveAdapter
from nerve_core.neuroletter import Neuroletter, Phase, Role


def _emit_cycle(nerve: SimNerveAdapter, n_cycles: int) -> np.ndarray:
    """Drive a deterministic 2-letter-per-cycle stimulus; record delivered codes."""
    codes: list[list[int]] = []
    for i in range(n_cycles):
        nerve.set_phase_active(gamma=True, theta=False)
        nerve.send(Neuroletter((i * 3) % 64, Role.PREDICTION, Phase.GAMMA, 0, 1, i * 1e-3))
        received_g = [letter.code for letter in nerve.listen(wml_id=1)]

        nerve.set_phase_active(gamma=False, theta=True)
        nerve.send(Neuroletter((i * 7) % 64, Role.ERROR, Phase.THETA, 2, 1, i * 1e-3))
        received_t = [letter.code for letter in nerve.listen(wml_id=1)]

        codes.append(received_g + received_t)

    # Pad to fixed width for NPZ storage.
    max_len = max((len(row) for row in codes), default=1) or 1
    matrix = np.full((len(codes), max_len), -1, dtype=np.int32)
    for i, row in enumerate(codes):
        matrix[i, : len(row)] = row
    return matrix


def main(out_dir: str = "tests/golden") -> None:
    torch.manual_seed(0)
    nerve = SimNerveAdapter(n_wmls=4, k=2, seed=0)

    trace = _emit_cycle(nerve, n_cycles=1000)

    # Transducer logits — one tensor per nerve key.
    transducers = {
        key: t.logits.detach().cpu().numpy()
        for key, t in nerve._transducers.items()
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "cycle_trace_4wmls_seed0.npz", codes=trace)
    np.savez(out / "transducers_merged.npz", **transducers)
    print(f"Frozen golden artefacts in {out.resolve()}")


if __name__ == "__main__":
    main()
