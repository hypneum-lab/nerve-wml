"""Train the GammaThetaMultiplexer constellation as a code-carrying channel.

Part 1 of the GTM-channel experiment. The γ/θ PAC multiplexer's only
trainable parameter is the ``[alphabet_size, 2]`` ``constellation`` (PSK init).
Untrained it round-trips codes at ~chance (≈1/64 ≈ 1.6 %). This script trains
``constellation`` for round-trip code reconstruction (channel coding):

    codes  ──forward──▶ carrier ──demodulate(soft)──▶ logits ──CE loss──▶ ∂/∂constellation

and reports the *hard* round-trip fidelity trajectory + plateau, robustness
under AWGN, and saves the trained ``state_dict``.

Usage:
    python scripts/train_gtm_channel.py \
        --steps 4000 --batch 256 --out runs/gtm_channel

Outputs (under ``--out``):
    constellation.pt        trained GTM state_dict
    gtm_channel_training.json  trajectory + plateau + noise sweep + metadata
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from track_p.multiplexer import AWGN, GammaThetaConfig, GammaThetaMultiplexer


def hard_fidelity(
    gtm: GammaThetaMultiplexer,
    codes: Tensor,
    *,
    noise: AWGN | None = None,
) -> float:
    """Fraction of codes recovered exactly by demodulate(forward(codes), hard=True)."""
    with torch.no_grad():
        carrier = gtm.forward(codes, noise=noise)
        recovered = gtm.demodulate(carrier, hard=True)
    return float((recovered == codes).float().mean().item())


def sample_codes(batch: int, k: int, alphabet: int, gen: torch.Generator) -> Tensor:
    return torch.randint(0, alphabet, (batch, k), generator=gen)


def train(
    *,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    eval_every: int,
    eval_batch: int,
    tau: float,
    train_sigma: float = 0.0,
    k_active: int | None = None,
) -> dict:
    """Train the constellation for round-trip code reconstruction.

    ``train_sigma`` injects AWGN on the carrier *during training* so the
    constellation can learn a noise-robust channel code (the clean K=7
    round-trip is already exact under PSK init — zero headroom — so noise is
    where any learnable capacity lives). ``k_active`` restricts the number of
    occupied symbol slots (K=1 is the live-loop regime, where the demod design
    matrix is over-provisioned and fidelity degrades).
    """
    cfg = GammaThetaConfig()
    k = k_active if k_active is not None else cfg.symbols_per_theta
    alphabet = cfg.alphabet_size

    gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)
    opt = torch.optim.Adam(gtm.parameters(), lr=lr)
    train_noise = AWGN(train_sigma) if train_sigma > 0.0 else None

    gen = torch.Generator()
    gen.manual_seed(seed)
    eval_gen = torch.Generator()
    eval_gen.manual_seed(seed + 10_000)
    # Fixed held-out eval batch so the trajectory is comparable across steps.
    eval_codes = sample_codes(eval_batch, k, alphabet, eval_gen)

    # Eval noise matches train noise so the trajectory tracks the objective
    # the optimiser actually sees (clean eval is reported separately too).
    eval_noise = AWGN(train_sigma) if train_sigma > 0.0 else None

    trajectory: list[dict] = []
    # Step 0 baseline (untrained), evaluated under the training noise level.
    fid0 = hard_fidelity(gtm, eval_codes, noise=eval_noise)
    fid0_clean = hard_fidelity(gtm, eval_codes)
    trajectory.append({"step": 0, "fidelity": fid0, "loss": None})

    t_start = time.time()
    for step in range(1, steps + 1):
        codes = sample_codes(batch, k, alphabet, gen)
        opt.zero_grad()
        carrier = gtm.forward(codes, noise=train_noise)
        logits = gtm.demodulate(carrier, hard=False, tau=tau)  # [B, K, alphabet]
        # gumbel_softmax(hard=False) returns a probability simplex; take log for NLL.
        loss = torch.nn.functional.nll_loss(
            torch.log(logits.reshape(-1, alphabet) + 1e-9),
            codes.reshape(-1),
        )
        loss.backward()
        opt.step()
        gtm.step()

        if step % eval_every == 0 or step == steps:
            fid = hard_fidelity(gtm, eval_codes, noise=eval_noise)
            trajectory.append(
                {"step": step, "fidelity": fid, "loss": float(loss.item())}
            )

    elapsed = time.time() - t_start

    # Plateau = mean fidelity over the last 5 logged points (robust to noise).
    tail = [p["fidelity"] for p in trajectory[-5:]]
    plateau = sum(tail) / len(tail)
    final_fid = trajectory[-1]["fidelity"]

    # Noise robustness sweep: trained constellation vs a fresh untrained PSK
    # baseline (same seed/init), so any gap is attributable to training.
    baseline_gtm = GammaThetaMultiplexer(cfg=cfg, seed=seed)
    noise_sweep: list[dict] = []
    for sigma in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        n_draws = 5
        fids_t = [hard_fidelity(gtm, eval_codes, noise=AWGN(sigma)) for _ in range(n_draws)]
        fids_b = [
            hard_fidelity(baseline_gtm, eval_codes, noise=AWGN(sigma))
            for _ in range(n_draws)
        ]
        noise_sweep.append(
            {
                "sigma": sigma,
                "fidelity_trained": sum(fids_t) / len(fids_t),
                "fidelity_untrained_psk": sum(fids_b) / len(fids_b),
            }
        )

    # Clean fidelity at the trained K plus a K=1 probe (live-loop regime), so
    # the report records that the demod is over-provisioned for K<k_cap.
    clean_final = hard_fidelity(gtm, eval_codes)
    k1_codes = sample_codes(eval_batch, 1, alphabet, eval_gen)
    clean_k1_trained = hard_fidelity(gtm, k1_codes)
    clean_k1_untrained = hard_fidelity(baseline_gtm, k1_codes)

    return {
        "config": {
            "gamma_hz": cfg.gamma_hz,
            "theta_hz": cfg.theta_hz,
            "sample_rate_hz": cfg.sample_rate_hz,
            "alphabet_size": alphabet,
            "symbols_per_theta": k,
            "modulation": cfg.modulation,
        },
        "hparams": {
            "steps": steps,
            "batch": batch,
            "lr": lr,
            "seed": seed,
            "tau": tau,
            "eval_batch": eval_batch,
            "train_sigma": train_sigma,
            "k_active": k,
        },
        "baseline_fidelity_untrained": fid0,
        "baseline_fidelity_untrained_clean": fid0_clean,
        "chance_fidelity": 1.0 / alphabet,
        "final_fidelity": final_fid,
        "final_fidelity_clean": clean_final,
        "plateau_fidelity": plateau,
        "k1_clean_trained": clean_k1_trained,
        "k1_clean_untrained": clean_k1_untrained,
        "trajectory": trajectory,
        "noise_sweep": noise_sweep,
        "elapsed_sec": elapsed,
        "_gtm_state_dict": gtm.state_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batch", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument(
        "--train-sigma",
        type=float,
        default=0.0,
        help="AWGN sigma injected during training (0 = clean; clean K=7 is "
        "already exact under PSK init, so >0 is where learnable capacity lives)",
    )
    ap.add_argument(
        "--k-active",
        type=int,
        default=None,
        help="number of occupied symbol slots (default = symbols_per_theta=7)",
    )
    ap.add_argument("--out", type=str, default="runs/gtm_channel")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train(
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_batch=args.eval_batch,
        tau=args.tau,
        train_sigma=args.train_sigma,
        k_active=args.k_active,
    )

    state_dict = result.pop("_gtm_state_dict")
    torch.save(state_dict, out_dir / "constellation.pt")

    json_path = out_dir / "gtm_channel_training.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"train_sigma                   : {args.train_sigma}")
    print(f"baseline untrained (eval@σ)   : {result['baseline_fidelity_untrained']:.4f}")
    print(f"baseline untrained (clean)    : {result['baseline_fidelity_untrained_clean']:.4f}")
    print(f"chance (1/alphabet)           : {result['chance_fidelity']:.4f}")
    print(f"final fidelity (eval@σ)       : {result['final_fidelity']:.4f}")
    print(f"final fidelity (clean K=7)    : {result['final_fidelity_clean']:.4f}")
    print(f"plateau fidelity (eval@σ)     : {result['plateau_fidelity']:.4f}")
    print(f"K=1 clean trained/untrained   : "
          f"{result['k1_clean_trained']:.4f} / {result['k1_clean_untrained']:.4f}")
    print("noise sweep  sigma : trained vs untrained-PSK:")
    for row in result["noise_sweep"]:
        print(f"  sigma={row['sigma']:.2f}  "
              f"trained={row['fidelity_trained']:.4f}  "
              f"untrained_psk={row['fidelity_untrained_psk']:.4f}")
    print(f"saved state_dict -> {out_dir / 'constellation.pt'}")
    print(f"saved results    -> {json_path}")


if __name__ == "__main__":
    main()
