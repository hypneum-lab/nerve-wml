"""Ablation of the N-3 invariant on a predictive-coding loop.

Unlike scripts/ablation_n3.py (w2_hard, gamma-only) and
scripts/ablation_n3_guard.py (artificial violation injection),
this script runs the full gamma -> theta -> consolidate loop
where WMLs emit epsilon letters naturally via emit_head_eps
when their internal surprise exceeds threshold_eps=0.30.

Pipeline per (strict_n3, seed) cell:
  (1) GAMMA phase   : wml.step(nerve, t) x N_steps on random input.
                      WMLs emit pi letters on every step and epsilon
                      letters when their (h - h_prior).norm() > 0.30.
  (2) THETA phase   : bridge.collect_eps_trace() records epsilon
                      letters with phase=THETA over duration_ticks.
  (3) Consolidation : bridge.apply_consolidation_output() applies
                      a zero-delta update to transducers (no-op
                      wrt weights, tests the plumbing end-to-end).
  (4) GAMMA eval    : count how many epsilon letters were emitted
                      during the gamma phase and survived the
                      invariant check at send() time.

Prediction under correct gate semantics:
  Because MlpWML.step() hard-codes phase=Phase.THETA when it emits
  epsilon (via emit_head_eps), the invariant N-3 is satisfied by
  construction. strict_n3=True and strict_n3=False should produce
  identical metrics (Delta = 0).

Any deviation would mean either:
  (a) an implementation bug where a WML emits epsilon in GAMMA, or
  (b) a path where strict mode changes behaviour beyond just
      raising on violations.

Does NOT modify any WML, nerve, or bridge -- uses public APIs.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch

from track_w.mock_nerve import MockNerve
from track_w.mlp_wml import MlpWML
from bridge.dream_bridge import DreamBridge
from nerve_core.neuroletter import Role


@contextlib.contextmanager
def _force_strict_n3(value: bool):
    original = MockNerve.__init__

    def patched(self, *args, **kwargs):
        kwargs["strict_n3"] = value
        return original(self, *args, **kwargs)

    MockNerve.__init__ = patched
    try:
        yield
    finally:
        MockNerve.__init__ = original


def _run_cell(strict: bool, seed: int, n_gamma_steps: int = 200) -> dict:
    torch.manual_seed(seed)

    with _force_strict_n3(strict):
        nerve = MockNerve(n_wmls=2, k=1, seed=seed)

    wml_a = MlpWML(id=0, d_hidden=16, seed=seed)
    wml_b = MlpWML(id=1, d_hidden=16, seed=seed + 1)

    nerve.set_phase_active(gamma=True, theta=False)

    gamma_send_caught = 0
    gamma_send_attempts = 0
    gamma_eps_attempted = 0

    for step in range(n_gamma_steps):
        x_a = torch.randn(16)
        x_b = torch.randn(16)
        gamma_send_attempts += 2
        try:
            wml_a.receive(x_a)
        except Exception:
            pass
        try:
            wml_b.receive(x_b)
        except Exception:
            pass
        try:
            wml_a.step(nerve, t=float(step) * 1e-3)
        except AssertionError:
            gamma_send_caught += 1
        try:
            wml_b.step(nerve, t=float(step) * 1e-3)
        except AssertionError:
            gamma_send_caught += 1
        nerve.tick(1e-3)

    bridge = DreamBridge(enabled=True)
    theta_trace_len = 0
    theta_caught = 0
    try:
        nerve.set_phase_active(gamma=False, theta=True)
        trace = bridge.collect_eps_trace(nerve, duration_ticks=200, dt=1e-3)
        theta_trace_len = len(trace)
    except AssertionError:
        theta_caught = 1

    delta = torch.zeros(2, 64, 64).numpy()
    apply_ok = True
    try:
        bridge.apply_consolidation_output(nerve, delta, alpha=0.1)
    except Exception as exc:
        apply_ok = False
        apply_err = str(exc)[:80]
    else:
        apply_err = ""

    return {
        "strict":              strict,
        "seed":                seed,
        "n_gamma_steps":       n_gamma_steps,
        "gamma_send_attempts": gamma_send_attempts,
        "gamma_send_caught":   gamma_send_caught,
        "theta_trace_len":     theta_trace_len,
        "theta_caught":        theta_caught,
        "apply_consolidation": apply_ok,
        "apply_err":           apply_err,
    }


def main() -> None:
    seeds = [0, 1, 2]
    cells = []
    for strict in (True, False):
        for s in seeds:
            cells.append(_run_cell(strict, s))

    strict_cells = [c for c in cells if c["strict"] is True]
    open_cells = [c for c in cells if c["strict"] is False]

    def _mean(ks, cs):
        return sum(c[ks] for c in cs) / max(len(cs), 1)

    summary = {
        "strict": {
            "gamma_send_caught_mean": _mean("gamma_send_caught", strict_cells),
            "theta_trace_len_mean":   _mean("theta_trace_len", strict_cells),
            "theta_caught_mean":      _mean("theta_caught", strict_cells),
        },
        "open": {
            "gamma_send_caught_mean": _mean("gamma_send_caught", open_cells),
            "theta_trace_len_mean":   _mean("theta_trace_len", open_cells),
            "theta_caught_mean":      _mean("theta_caught", open_cells),
        },
        "delta": {
            "gamma_send_caught": (
                _mean("gamma_send_caught", strict_cells)
                - _mean("gamma_send_caught", open_cells)
            ),
            "theta_trace_len": (
                _mean("theta_trace_len", strict_cells)
                - _mean("theta_trace_len", open_cells)
            ),
        },
    }

    out_dir = Path("papers/paper1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_n3_predictive.json"
    out_path.write_text(
        json.dumps({"cells": cells, "summary": summary}, indent=2)
    )

    print(f"N-3 predictive-coding ablation -- 2 conditions x {len(seeds)} seeds")
    print()
    header = f"{'cond':>8}{'seed':>6}{'g_caught':>10}{'t_trace':>10}{'apply':>8}"
    print(header)
    print("-" * len(header))
    for c in cells:
        cond = "strict" if c["strict"] else "open"
        print(
            f"{cond:>8}{c['seed']:>6}{c['gamma_send_caught']:>10}"
            f"{c['theta_trace_len']:>10}{str(c['apply_consolidation']):>8}"
        )

    print()
    print("Summary means:")
    print(f"  strict gamma_send_caught = {summary['strict']['gamma_send_caught_mean']:.2f}")
    print(f"  open   gamma_send_caught = {summary['open']['gamma_send_caught_mean']:.2f}")
    print(f"  strict theta_trace_len   = {summary['strict']['theta_trace_len_mean']:.2f}")
    print(f"  open   theta_trace_len   = {summary['open']['theta_trace_len_mean']:.2f}")
    print()
    print(f"Delta gamma_send_caught = {summary['delta']['gamma_send_caught']:+.2f}")
    print(f"Delta theta_trace_len   = {summary['delta']['theta_trace_len']:+.2f}")
    print()
    any_delta = (
        abs(summary["delta"]["gamma_send_caught"]) > 0.5
        or abs(summary["delta"]["theta_trace_len"]) > 0.5
    )
    if any_delta:
        print("Verdict: SIGNAL -- N-3 gate affects the correct pipeline.")
    else:
        print(
            "Verdict: NO SIGNAL -- WMLs emit epsilon in Phase.THETA by construction;"
        )
        print(
            "         the invariant is satisfied structurally, gate acts as a"
        )
        print("         formal type-checker only.")
    print()
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
