"""N-3 invariant guard validation via violation injection.

Tests whether the gamma/theta invariant (N-3: role==ERROR iff
phase==THETA) functions as a formal correctness contract by
injecting malformed Neuroletters (role=ERROR, phase=GAMMA --
violating N-3) into a MockNerve with strict_n3 toggled.

Unlike scripts/ablation_n3.py which tests the canonical pipeline
(where invariants are satisfied by construction and ablation
yields Delta=0), this script stresses the gate with deliberate
violations and measures:

  * strict=True : AssertionError raised per malformed letter
    (gate catches -> formal correctness preserved).
  * strict=False: malformed letter passes through send()
    without crash (silent -- downstream consumers may mis-behave).

For each condition we sweep the violation rate
v in {0.00, 0.05, 0.10, 0.25, 0.50} across 3 seeds, emitting
N=1000 letters total per cell, and report:

  * n_caught        : count of malformed letters that raised.
  * n_silent_passed : count of malformed letters that passed.
  * n_received      : count of letters visible at listen().

Expected outcome under correct gate semantics:
  * strict=True  -> n_caught ~= violation_rate * N,  n_silent = 0
  * strict=False -> n_caught = 0,  n_silent ~= violation_rate * N

Deviation from that shape reveals a gate bug (or semantic drift
in MockNerve).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from track_w.mock_nerve import MockNerve
from nerve_core.neuroletter import Neuroletter, Role, Phase


def _run_cell(
    strict: bool,
    violation_rate: float,
    n_letters: int,
    seed: int,
) -> dict:
    """Run one ablation cell: strict x violation_rate x seed."""
    rng = random.Random(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed, strict_n3=strict)
    nerve.set_phase_active(gamma=False, theta=True)

    n_caught = 0
    n_silent = 0
    n_wellformed = 0
    received_letters: list = []

    for i in range(n_letters):
        is_violation = rng.random() < violation_rate
        if is_violation:
            letter = Neuroletter(
                code=i % 64,
                role=Role.ERROR,
                phase=Phase.GAMMA,
                src=0,
                dst=1,
                timestamp=float(i),
            )
        else:
            letter = Neuroletter(
                code=i % 64,
                role=Role.ERROR,
                phase=Phase.THETA,
                src=0,
                dst=1,
                timestamp=float(i),
            )

        try:
            nerve.send(letter)
            if is_violation:
                n_silent += 1
            else:
                n_wellformed += 1
        except AssertionError:
            n_caught += 1

    try:
        nerve.tick(1e-3)
    except Exception:
        pass

    try:
        received = nerve.listen(dst=1)
        received_letters = list(received)
    except Exception:
        received_letters = []

    return {
        "strict":            strict,
        "violation_rate":    violation_rate,
        "seed":              seed,
        "n_letters_sent":    n_letters,
        "n_caught":          n_caught,
        "n_silent_passed":   n_silent,
        "n_wellformed_sent": n_wellformed,
        "n_received":        len(received_letters),
    }


def main() -> None:
    import numpy as np

    n_letters = 1000
    violation_rates = [0.0, 0.05, 0.10, 0.25, 0.50]
    seeds = [0, 1, 2]

    all_cells = []
    for strict in (True, False):
        for v in violation_rates:
            for s in seeds:
                cell = _run_cell(strict, v, n_letters, s)
                all_cells.append(cell)

    aggregated = []
    for strict in (True, False):
        for v in violation_rates:
            cells = [c for c in all_cells if c["strict"] is strict and c["violation_rate"] == v]
            aggregated.append({
                "strict":            strict,
                "violation_rate":    v,
                "n_letters":         n_letters,
                "n_caught_mean":     float(np.mean([c["n_caught"] for c in cells])),
                "n_silent_mean":     float(np.mean([c["n_silent_passed"] for c in cells])),
                "n_received_mean":   float(np.mean([c["n_received"] for c in cells])),
                "expected_violate":  v * n_letters,
            })

    out_dir = Path("papers/paper1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_n3_guard.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "n_letters": n_letters,
                    "violation_rates": violation_rates,
                    "seeds": seeds,
                },
                "cells": all_cells,
                "aggregated": aggregated,
            },
            indent=2,
        )
    )

    print(f"N-3 guard ablation -- N={n_letters} letters, {len(seeds)} seeds, "
          f"{len(violation_rates)} violation rates.")
    print()
    header = (
        f"{'strict':>7}{'vrate':>8}{'expected':>10}"
        f"{'caught':>10}{'silent':>10}{'received':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in aggregated:
        print(
            f"{str(row['strict']):>7}{row['violation_rate']:>8.2f}"
            f"{row['expected_violate']:>10.1f}"
            f"{row['n_caught_mean']:>10.1f}"
            f"{row['n_silent_mean']:>10.1f}"
            f"{row['n_received_mean']:>10.1f}"
        )

    print()
    print("Expected shape under correct gate semantics:")
    print("  strict=True  -> caught ~= expected, silent = 0")
    print("  strict=False -> caught = 0, silent ~= expected")
    print()
    strict_rows = [r for r in aggregated if r["strict"] is True and r["violation_rate"] > 0]
    open_rows   = [r for r in aggregated if r["strict"] is False and r["violation_rate"] > 0]
    def _tol(expected: float) -> float:
        return max(10.0, 0.15 * expected)

    strict_ok = all(
        abs(r["n_caught_mean"] - r["expected_violate"]) < _tol(r["expected_violate"])
        and r["n_silent_mean"] == 0
        for r in strict_rows
    )
    open_ok = all(
        r["n_caught_mean"] == 0
        and abs(r["n_silent_mean"] - r["expected_violate"]) < _tol(r["expected_violate"])
        for r in open_rows
    )
    print(f"strict mode conforms: {strict_ok}")
    print(f"open   mode conforms: {open_ok}")
    print()
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
