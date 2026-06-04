"""L1a: Full-LIF sim training + scaling sweep on HardFlowProxyTask.

Usage:
    uv run python scripts/lif_l1a_train.py            # primary run n_neurons=1024
    uv run python scripts/lif_l1a_train.py --sweep    # sweep {128,256,512,1024}
    uv run python scripts/lif_l1a_train.py --n 256    # single run with n_neurons=256

Outputs:
    docs/superpowers/research/2026-05-30-lif-l1a-train.jsonl  (per-step log)
    docs/superpowers/research/2026-05-30-lif-l1a-gate.json    (gate evidence)
    docs/superpowers/research/2026-05-30-lif-l1a-scaling.json (sweep results)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F  # noqa: N812

# Ensure project root is importable when invoked directly.
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402

_RESEARCH = _ROOT / "docs" / "superpowers" / "research"

# L1a constants
TASK_DIM = 16
N_CLASSES = 12
INPUT_ENCODER_DIM = 64   # 16 → 64 projection (spec §3.2 updated)
T_TICKS = 8
N_STEPS = 2000
BATCH = 64
LR = 3e-3
GRAD_CLIP = 1.0
SEED = 0
LOG_EVERY = 100
SWEEP_SIZES = [128, 256, 512, 1024]
PRIMARY_N = 1024


def _eval_acc(
    lif: LifWML,
    input_encoder: torch.nn.Linear,
    task: HardFlowProxyTask,
    *,
    batch: int = 512,
) -> float:
    """Evaluate accuracy via T-tick forward pass (no grad)."""
    with torch.no_grad():
        x, y = task.sample(batch=batch)
        v_mem = torch.zeros(batch, lif.n_neurons)
        spikes_acc = torch.zeros(batch, lif.n_neurons)
        for _ in range(T_TICKS):
            i_in = lif.input_proj(input_encoder(x))
            v_mem = v_mem + (1e-3 / lif.tau_mem) * (-v_mem + i_in)
            spikes = (v_mem > lif.v_thr).float()
            v_mem = v_mem * (1.0 - spikes)
            spikes_acc = spikes_acc + spikes
        logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
        preds = logits.argmax(-1)
    return (preds == y).float().mean().item()


def train_one(
    n_neurons: int,
    seed: int = SEED,
    *,
    log_path: pathlib.Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train a single LifWML of size n_neurons.

    Returns a dict with training results suitable for the gate and sweep JSON.
    """
    torch.manual_seed(seed)

    task = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=seed)
    lif = LifWML(
        id=0,
        n_neurons=n_neurons,
        alphabet_size=64,
        input_dim=INPUT_ENCODER_DIM,
        seed=seed + 10,  # match historical offset from run_w2_hard
    )
    input_encoder = torch.nn.Linear(TASK_DIM, INPUT_ENCODER_DIM)
    torch.nn.init.kaiming_normal_(input_encoder.weight, nonlinearity="relu")
    torch.nn.init.zeros_(input_encoder.bias)

    # Baseline (untrained) accuracy on a fresh eval task instance.
    task_eval = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=seed)
    acc_untrained = _eval_acc(lif, input_encoder, task_eval)

    optimizer = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=LR,
    )

    loss_step_log: list[dict] = []
    loss_100: float | None = None
    loss_final: float = math.nan
    acc_final: float = 0.0

    if verbose:
        print(f"[L1a] n_neurons={n_neurons}  seed={seed}  "
              f"acc_untrained={acc_untrained:.3f}")

    log_file = None
    if log_path is not None:
        log_file = open(log_path, "a")

    for step in range(1, N_STEPS + 1):
        x, y = task.sample(batch=BATCH)

        v_mem = torch.zeros(BATCH, n_neurons)
        spikes_acc = torch.zeros(BATCH, n_neurons)
        for _ in range(T_TICKS):
            i_in = lif.input_proj(input_encoder(x))
            v_mem = v_mem + (1e-3 / lif.tau_mem) * (-v_mem + i_in)
            spikes = spike_with_surrogate(v_mem, v_thr=lif.v_thr)
            v_mem = v_mem * (1.0 - spikes.detach())
            spikes_acc = spikes_acc + spikes

        logits = lif.emit_head_pi(spikes_acc)[:, :N_CLASSES]
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(lif.parameters()) + list(input_encoder.parameters()),
            GRAD_CLIP,
        )
        optimizer.step()

        loss_val = loss.item()
        with torch.no_grad():
            acc_val = (logits.argmax(-1) == y).float().mean().item()

        if step == LOG_EVERY:
            loss_100 = loss_val

        if step % LOG_EVERY == 0:
            row = {"step": step, "n_neurons": n_neurons, "loss": loss_val, "acc": acc_val}
            loss_step_log.append(row)
            if log_file is not None:
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
            if verbose:
                print(f"  step={step:4d}  loss={loss_val:.4f}  acc={acc_val:.3f}")

        if step == N_STEPS:
            loss_final = loss_val

    if log_file is not None:
        log_file.close()

    # Final eval on fresh task instance.
    task_eval2 = HardFlowProxyTask(dim=TASK_DIM, n_classes=N_CLASSES, seed=seed)
    acc_final = _eval_acc(lif, input_encoder, task_eval2)

    if verbose:
        print(f"[L1a] n_neurons={n_neurons}  acc_final={acc_final:.3f}  "
              f"loss_100={loss_100:.4f}  loss_final={loss_final:.4f}")

    return {
        "n_neurons": n_neurons,
        "input_dim": INPUT_ENCODER_DIM,
        "seed": seed,
        "T_ticks": T_TICKS,
        "n_steps": N_STEPS,
        "acc_untrained": acc_untrained,
        "loss_step100": loss_100,
        "loss_step2000": loss_final,
        "acc_final": acc_final,
        "loss_descent": (loss_final < loss_100) if loss_100 is not None else None,
        "gate_passed": acc_final >= 0.20 and (loss_100 is not None and loss_final < loss_100),
    }


def run_sweep(verbose: bool = True) -> list[dict]:
    """Sweep over SWEEP_SIZES and collect results."""
    _RESEARCH.mkdir(parents=True, exist_ok=True)
    results = []
    for n in SWEEP_SIZES:
        r = train_one(n, seed=SEED, verbose=verbose)
        results.append(r)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="L1a LIF training")
    parser.add_argument("--sweep", action="store_true", help="Run scaling sweep")
    parser.add_argument("--n", type=int, default=PRIMARY_N,
                        help="n_neurons for single run (default: 1024)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _RESEARCH.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    log_path = _RESEARCH / "2026-05-30-lif-l1a-train.jsonl"
    # Truncate per-run (fresh run).
    log_path.write_text("")

    if args.sweep:
        results = run_sweep(verbose=verbose)

        # Write sweep JSON.
        sweep_path = _RESEARCH / "2026-05-30-lif-l1a-scaling.json"
        sweep_path.write_text(json.dumps(results, indent=2))
        print(f"\nScaling sweep written to {sweep_path}")

        # Print table.
        print("\nn_neurons | acc_untrained | acc_final | loss_100 | loss_final | gate")
        print("-" * 75)
        for r in results:
            print(
                f"  {r['n_neurons']:5d}   |"
                f"   {r['acc_untrained']:.3f}       |"
                f"  {r['acc_final']:.3f}    |"
                f" {r['loss_step100']:.4f}   |"
                f" {r['loss_step2000']:.4f}     |"
                f" {'PASS' if r['gate_passed'] else 'FAIL'}"
            )

        # Gate for primary size (1024).
        primary = next((r for r in results if r["n_neurons"] == PRIMARY_N), results[-1])
        gate_path = _RESEARCH / "2026-05-30-lif-l1a-gate.json"
        gate_data = {
            "task": "HardFlowProxyTask",
            "n_classes": N_CLASSES,
            **primary,
        }
        gate_path.write_text(json.dumps(gate_data, indent=2))
        print(f"\nGate evidence written to {gate_path}")
        gate_status = "PASSED" if primary["gate_passed"] else "FAILED"
        print(
            f"Gate {gate_status}: acc={primary['acc_final']:.3f} >= 0.20"
            f" and loss descends = {primary['loss_descent']}"
        )

    else:
        r = train_one(args.n, seed=args.seed, log_path=log_path, verbose=verbose)

        gate_path = _RESEARCH / "2026-05-30-lif-l1a-gate.json"
        gate_data = {"task": "HardFlowProxyTask", "n_classes": N_CLASSES, **r}
        gate_path.write_text(json.dumps(gate_data, indent=2))
        print(f"\nGate evidence written to {gate_path}")
        print(f"Gate {'PASSED' if r['gate_passed'] else 'FAILED'}: "
              f"acc={r['acc_final']:.3f} >= 0.20 and loss descends = {r['loss_descent']}")


if __name__ == "__main__":
    main()
